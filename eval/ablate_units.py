"""So sánh các cách ghép "ý cần tìm" (retrieval.search.POLICIES) trên cả bộ test, chỉ truy xuất (chưa rerank).

Chạy trong môi trường có model, Qdrant và khoá Gemini (xem eval/compare_rerank.py để biết cách chạy trong container):
  python /tmp/ablate_units.py /tmp/t.jsonl /tmp/ablate.json
Kết quả phân tích của Gemini được lưu cạnh file đầu ra (`.parsed.json`) để chạy lại không gọi Gemini nữa.
"""
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from retrieval import search as S
from retrieval.api import CANDIDATES
from retrieval.grounding import ground
from retrieval.main import build_services
from retrieval.qparse import ParsedQuery, SubQuery, parse_query


class CachedEmbedder:
    """Các cách ghép dùng chung phần lớn văn bản, nên nhúng mỗi văn bản một lần."""

    def __init__(self, inner):
        self.inner, self._sig, self._bge = inner, {}, {}

    def text_siglip(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self._sig]
        if miss:
            for t, v in zip(miss, self.inner.text_siglip(miss), strict=True):
                self._sig[t] = v
        return [self._sig[t] for t in texts]

    def text_bge(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self._bge]
        if miss:
            dense, sparse = self.inner.text_bge(miss)
            for t, d, s in zip(miss, dense, sparse, strict=True):
                self._bge[t] = (d, s)
        return [self._bge[t][0] for t in texts], [self._bge[t][1] for t in texts]


def to_parsed(d: dict) -> ParsedQuery:
    return ParsedQuery(qt=d["qt"], subs=[SubQuery(**s) for s in d["subs"]])


def metrics(rows: list[tuple[list[str], list[set[str]]]]) -> dict:
    rec = [1.0 if set(ids[:10]) & set().union(*golds) else 0.0 for ids, golds in rows]
    cov = [sum(1 for g in golds if set(ids[:10]) & g) / len(golds) for ids, golds in rows]
    return {"n": len(rows), "recall@10": round(statistics.fmean(rec), 4), "intent_cov@10": round(statistics.fmean(cov), 4)}


async def main(testset: str, out: str) -> None:
    svc = build_services()
    emb = CachedEmbedder(svc.embedder)
    tests = [json.loads(line) for line in Path(testset).read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_path = Path(out).with_suffix(".parsed.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    names = svc.catalog.names if svc.catalog else []
    for i, t in enumerate(tests):
        if t["id"] in cache:
            continue
        p = await parse_query(t["query"], svc.llm, timeout_ms=svc.settings.gemini_timeout_ms, entity_names=names)
        if svc.catalog:
            p = ground(p, svc.catalog)
        cache[t["id"]] = asdict(p)
        if i % 25 == 0:
            print(f"parse {i}/{len(tests)}", flush=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    report: dict = {"candidates": CANDIDATES, "policies": {}}
    for pol in S.POLICIES:
        S.UNIT_POLICY = pol
        by_type: dict[str, list] = {}
        t0 = time.time()
        for t in tests:
            parsed = to_parsed(cache[t["id"]])
            hits = S.search(parsed, emb, svc.qdrant, CANDIDATES, svc.collection, 3, S.BRANCHES, {})
            by_type.setdefault(t["type"], []).append(([h.image_id for h in hits[:10]], [set(g) for g in t["intents"]]))
        report["policies"][pol] = {k: metrics(v) for k, v in sorted(by_type.items())}
        print(pol, json.dumps(report["policies"][pol], ensure_ascii=False), f"{time.time() - t0:.0f}s", flush=True)
    S.UNIT_POLICY = "current"
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
