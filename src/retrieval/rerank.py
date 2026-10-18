"""Rerank bằng cross-encoder (bge-reranker-base) trên CPU, cắt 128 token, tối đa 24 cặp."""
from collections import defaultdict
from typing import Protocol

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


def doc_text(h: Hit) -> str:
    names = " / ".join(h.entity_names or [])
    cap = (h.caption or "")[:MAX_CAPTION_CHARS]
    return f"{names}: {cap}".strip(": ")


def rerank_hits(query: str | list[str], hits: list[Hit], reranker: Scorer, top_k: int = 10,
                max_pairs: int = 24, max_per_entity: int = 3) -> list[Hit]:
    """Chấm lại tối đa max_pairs ứng viên đầu (đã xen kẽ theo ý), mỗi hit so với query của ý của nó.

    Sắp xếp lại trong từng ý rồi xen kẽ tiếp, nên không ý nào bị đẩy khỏi top_k."""
    cand = hits[:max_pairs]
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
