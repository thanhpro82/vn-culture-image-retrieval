"""So sánh rerank: không rerank / bge-reranker-base (local) / jina-reranker-v2 (endpoint ngoài).

Chạy trong cùng môi trường với API (cần model, Qdrant, Gemini, biến EXT_*), ví dụ:
  docker compose cp eval/compare_rerank.py api:/tmp/ && docker compose cp <testset> api:/tmp/t.jsonl
  docker compose exec -T api python /tmp/compare_rerank.py /tmp/t.jsonl /tmp/out.json
Mỗi truy vấn dùng chung một danh sách ứng viên, nên chênh lệch chỉ đến từ bộ rerank.
Endpoint ngoài có hạn mức chung: script tự giãn tốc độ (RPM) và ghi lại số lần gọi.
"""
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from retrieval.api import CANDIDATES
from retrieval.grounding import ground
from retrieval.main import build_services
from retrieval.qparse import parse_query
from retrieval.rerank import rerank_hits
from retrieval.search import BRANCHES, _units, search

RPM = int(os.environ.get("EXT_RPM", "40"))


class RemoteScorer:
    """Scorer tương thích rerank_hits: gọi /v1/rerank, giữ nhịp tối đa RPM yêu cầu mỗi phút."""

    def __init__(self):
        self.base, self.key = os.environ["EXT_BASE_URL"].rstrip("/"), os.environ["EXT_API_KEY"]
        self.model = os.environ.get("EXT_RERANK_MODEL", "jina-reranker-v2")
        self.client = httpx.Client(timeout=30)
        self.calls = self.failures = 0
        self.latencies: list[float] = []
        self._next = 0.0

    def score(self, query: str, docs: list[str]) -> list[float]:
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + 60.0 / RPM
        body = json.dumps({"model": self.model, "query": query, "documents": docs}, ensure_ascii=False).encode()
        for attempt in range(3):
            t = time.perf_counter()
            self.calls += 1
            r = self.client.post(self.base + "/v1/rerank", content=body,
                                 headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            self.latencies.append((time.perf_counter() - t) * 1000)
            if r.status_code == 200:
                out = [0.0] * len(docs)
                for item in r.json()["results"]:
                    out[item["index"]] = float(item["relevance_score"])
                return out
            self.failures += 1
            time.sleep(5 * (attempt + 1))  # 429/5xx: lùi lại, không dồn dập
        raise RuntimeError("endpoint ngoài lỗi sau 3 lần thử")


def recall(results: list[list[str]], gold: list[set[str]]) -> float:
    return statistics.fmean(1.0 if set(r[:10]) & g else 0.0 for r, g in zip(results, gold, strict=True))


async def main(testset: str, out: str, limit: int | None) -> None:
    svc = build_services()
    remote = RemoteScorer()
    tests = [json.loads(line) for line in Path(testset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        tests = tests[:limit]
    names = svc.catalog.names if svc.catalog else []
    res: dict[str, dict[str, list]] = {k: {"none": [], "local": [], "remote": [], "gold": []}
                                       for k in {t["type"] for t in tests}}
    cache_path = Path(out).with_suffix(".cache.jsonl")  # lưu từng truy vấn: chạy lại thì bỏ qua phần đã xong
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    t0 = time.time()
    for i, t in enumerate(tests):
        if t["id"] in done:
            row = done[t["id"]]
            bucket = res[t["type"]]
            for m in ("none", "local", "remote"):
                bucket[m].append(row[m])
            bucket["gold"].append(set().union(*map(set, t["intents"])))
            continue
        parsed = await parse_query(t["query"], svc.llm, timeout_ms=svc.settings.gemini_timeout_ms, entity_names=names)
        if svc.catalog:
            parsed = ground(parsed, svc.catalog)
        for attempt in range(4):  # Qdrant có thể ngắt kết nối thoáng qua
            try:
                hits = search(parsed, svc.embedder, svc.qdrant, CANDIDATES, svc.collection, 3, BRANCHES, {})
                break
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
        queries = [u[1] for u in _units(parsed)]
        bucket = res[t["type"]]
        bucket["none"].append([h.image_id for h in hits[:10]])
        bucket["local"].append([h.image_id for h in rerank_hits(queries, hits, svc.reranker, 10, CANDIDATES)])
        try:
            bucket["remote"].append([h.image_id for h in rerank_hits(queries, hits, remote, 10, CANDIDATES)])
        except RuntimeError:
            bucket["remote"].append([h.image_id for h in hits[:10]])  # đếm là không rerank, ghi nhận qua failures
        bucket["gold"].append(set().union(*map(set, t["intents"])))
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": t["id"], **{m: bucket[m][-1] for m in ("none", "local", "remote")}}) + "\n")
        if i < 5 or i % 20 == 0:
            print(f"{i}/{len(tests)} {time.time() - t0:.0f}s remote_calls={remote.calls} fail={remote.failures} subs={len(queries)}", flush=True)
    report = {"n": len(tests), "candidates": CANDIDATES, "rpm": RPM, "remote_calls": remote.calls,
              "remote_failures": remote.failures,
              "remote_latency_ms_p50": round(statistics.median(remote.latencies)) if remote.latencies else None,
              "by_type": {k: {"n": len(v["gold"]), **{m: round(recall(v[m], v["gold"]), 4) for m in ("none", "local", "remote")}}
                          for k, v in res.items()}}
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None))
