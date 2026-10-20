"""Rerank bằng cross-encoder (bge-reranker-base) trên CPU, cắt 128 token, tối đa 24 cặp."""
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import httpx

from retrieval.search import Hit, interleave

RERANK_ID = "BAAI/bge-reranker-base"
MAX_LEN = 128
MAX_CAPTION_CHARS = 300


class Scorer(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...
    # tuỳ chọn: score_pairs(pairs) -> list[float] để chấm gộp nhiều query trong một lần chạy


class Reranker:
    max_length = MAX_LEN

    def __init__(self, model_id: str = RERANK_ID, device: str = "cpu", quantize: bool = True,
                 threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        if quantize and device == "cpu":  # INT8 động: ~1,5x nhanh hơn fp32 trên CPU (ADR-002)
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.device = device

    def tokenize(self, query: str, docs: list[str]):
        return self._tokenize_pairs([(query, d) for d in docs])

    def _tokenize_pairs(self, pairs: list[tuple[str, str]]):
        return self.tokenizer([q for q, _ in pairs], [d for _, d in pairs], padding=True,
                              truncation="only_second", max_length=self.max_length, return_tensors="pt")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Chấm một loạt cặp (query, tài liệu) trong MỘT lần chạy mô hình, mỗi cặp có query riêng."""
        if not pairs:
            return []
        enc = {k: v.to(self.device) for k, v in self._tokenize_pairs(pairs).items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits.view(-1)
        return self.torch.sigmoid(logits).float().cpu().tolist()

    def score(self, query: str, docs: list[str]) -> list[float]:
        return self.score_pairs([(query, d) for d in docs])


class HttpRerankClient:
    """Gọi endpoint rerank tương thích OpenAI/Jina (`POST /v1/rerank`). Lỗi HTTP hoặc mạng thì ném ngoại lệ."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 0.8,
                 transport: httpx.BaseTransport | None = None):
        self.url = base_url.rstrip("/") + "/v1/rerank"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model
        self.client = httpx.Client(timeout=timeout_s, transport=transport)

    def __call__(self, query: str, docs: list[str]) -> list[float]:
        body = json.dumps({"model": self.model, "query": query, "documents": docs}, ensure_ascii=False)
        r = self.client.post(self.url, content=body.encode("utf-8"), headers=self.headers)
        r.raise_for_status()
        out = [0.0] * len(docs)
        for item in r.json()["results"]:
            out[item["index"]] = float(item["relevance_score"])
        return out


class RemoteFallbackScorer:
    """Ưu tiên rerank ở endpoint ngoài (nhanh hơn, nhưng hạn mức dùng chung và địa chỉ có thể chết);
    hết lượt, lỗi, quá hạn hoặc kết quả sai độ dài thì dùng bộ rerank local.

    Sau một lỗi, endpoint ngoài bị bỏ qua `cooldown_s` giây để không chờ timeout lặp lại ở mọi truy vấn."""

    def __init__(self, local, remote_call: Callable[[str, list[str]], list[float]], rpm: int = 40,
                 cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.local, self.remote_call = local, remote_call
        self.rpm, self.cooldown_s, self.clock = rpm, cooldown_s, clock
        self._stamps: deque[float] = deque()
        self._blocked_until = 0.0
        self._lock = threading.Lock()
        self.remote_ok = self.fallbacks = 0

    def _reserve(self, n: int) -> bool:
        with self._lock:
            now = self.clock()
            if now < self._blocked_until:
                return False
            while self._stamps and now - self._stamps[0] >= 60.0:
                self._stamps.popleft()
            if len(self._stamps) + n > self.rpm:
                return False
            self._stamps.extend([now] * n)
            return True

    def score(self, query: str, docs: list[str]) -> list[float]:
        return self.score_pairs([(query, d) for d in docs])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        by_query: dict[str, list[int]] = defaultdict(list)
        for i, (q, _) in enumerate(pairs):
            by_query[q].append(i)
        if self._reserve(len(by_query)):
            try:
                out = [0.0] * len(pairs)
                for q, idx in by_query.items():
                    scores = self.remote_call(q, [pairs[i][1] for i in idx])
                    if len(scores) != len(idx):
                        raise ValueError("endpoint ngoài trả sai số điểm")
                    for i, sc in zip(idx, scores, strict=True):
                        out[i] = sc
                self.remote_ok += 1
                return out
            except Exception:  # noqa: BLE001 - mọi lỗi của endpoint ngoài (mạng, 4xx/5xx, JSON) đều quay về local
                with self._lock:
                    self._blocked_until = self.clock() + self.cooldown_s
        self.fallbacks += 1
        return self.local.score_pairs(pairs)


def doc_text(h: Hit) -> str:
    names = " / ".join(h.entity_names or [])
    cap = (h.caption or "")[:MAX_CAPTION_CHARS]
    return f"{names}: {cap}".strip(": ")


def rerank_hits(query: str | list[str], hits: list[Hit], reranker: Scorer, top_k: int = 10,
                max_pairs: int = 24, max_per_entity: int = 3) -> list[Hit]:
    """Chấm lại tối đa max_pairs ứng viên đầu (đã xen kẽ theo ý), mỗi hit so với query của ý của nó.

    Sắp xếp lại trong từng ý rồi xen kẽ tiếp, nên không ý nào bị đẩy khỏi top_k."""
    cand = [replace(h) for h in hits[:max_pairs]]  # bản sao: luồng chạy muộn không được ghi đè điểm của người khác
    by_sub: dict[int, list[Hit]] = defaultdict(list)
    for h in cand:
        by_sub[h.source_sub].append(h)

    def q_of(h: Hit) -> str:
        return query[h.source_sub] if isinstance(query, list) else query

    if hasattr(reranker, "score_pairs"):  # một lần chạy mô hình cho mọi ý (nhanh hơn nhiều lần gọi nhỏ)
        scores = reranker.score_pairs([(q_of(h), doc_text(h)) for h in cand])
        for h, s in zip(cand, scores, strict=True):
            h.score = float(s)
    else:
        for lst in by_sub.values():
            scores = reranker.score(q_of(lst[0]), [doc_text(h) for h in lst])
            for h, s in zip(lst, scores, strict=True):
                h.score = float(s)
    for lst in by_sub.values():
        lst.sort(key=lambda h: -h.score)
    return interleave([by_sub[s] for s in sorted(by_sub)], top_k=top_k, max_per_entity=max_per_entity)
