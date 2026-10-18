"""Chỉ số Recall@k, MRR@k, độ phủ ý; và chạy bộ test qua /search."""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def recall_at_k(results: list[list[str]], gold: list[set[str]], k: int = 10) -> float:
    """Trúng nếu có ít nhất 1 ảnh trong top-k thuộc đáp án."""
    return statistics.fmean(1.0 if set(r[:k]) & g else 0.0 for r, g in zip(results, gold, strict=True))


def mrr_at_k(results: list[list[str]], gold: list[set[str]], k: int = 10) -> float:
    def rr(r, g):
        return next((1.0 / (i + 1) for i, x in enumerate(r[:k]) if x in g), 0.0)
    return statistics.fmean(rr(r, g) for r, g in zip(results, gold, strict=True))


def intent_coverage(results: list[list[str]], gold_intents: list[list[set[str]]], k: int = 10) -> float:
    """Tỉ lệ ý (của mỗi query) có ít nhất 1 ảnh trong top-k, trung bình theo query."""
    return statistics.fmean(
        sum(1 for g in intents if set(r[:k]) & g) / len(intents)
        for r, intents in zip(results, gold_intents, strict=True))


def _call(base: str, key: str, query: str, mode: str, rerank: bool) -> tuple[list[str], float]:
    body = json.dumps({"query": query, "top_k": 10, "mode": mode, "rerank": rerank}).encode()
    req = urllib.request.Request(base + "/search", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return [x["image_id"] for x in data["results"]], (time.perf_counter() - t) * 1000


def run(testset: Path, base: str, key: str, mode: str, rerank: bool, workers: int = 2) -> dict:
    tests = [json.loads(line) for line in testset.read_text(encoding="utf-8").splitlines() if line.strip()]
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda t: _call(base, key, t["query"], mode, rerank), tests))
    report: dict = {"mode": mode, "rerank": rerank, "n": len(tests), "by_type": {}}
    for kind in sorted({t["type"] for t in tests}):
        idx = [i for i, t in enumerate(tests) if t["type"] == kind]
        res = [outs[i][0] for i in idx]
        gold = [set().union(*map(set, tests[i]["intents"])) for i in idx]
        lat = sorted(outs[i][1] for i in idx)
        row = {"n": len(idx), "recall@10": round(recall_at_k(res, gold), 4), "mrr@10": round(mrr_at_k(res, gold), 4),
               "latency_ms": {"p50": round(lat[len(lat) // 2]), "p95": round(lat[int(0.95 * len(lat)) - 1]),
                              "max": round(lat[-1])}}
        if kind == "multi_intent":
            row["intent_coverage@10"] = round(intent_coverage(res, [[set(g) for g in tests[i]["intents"]] for i in idx]), 4)
        report["by_type"][kind] = row
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "image", "text"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = run(Path(a.testset), a.base, a.key, a.mode, not a.no_rerank)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
