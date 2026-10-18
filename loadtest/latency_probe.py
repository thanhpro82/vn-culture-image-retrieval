"""Đo độ trễ tuần tự của /search với danh sách query cho trước (một yêu cầu tại một thời điểm).

Dùng 127.0.0.1 thay cho localhost: trên Windows localhost thử IPv6 trước và thêm ~2 s mỗi yêu cầu.
"""
import json
import statistics
import sys
import time
import urllib.request


def probe(base: str, key: str, queries: list[str]) -> list[tuple[str, float, dict]]:
    out = []
    for q in queries:
        req = urllib.request.Request(
            base + "/search", data=json.dumps({"query": q, "top_k": 10}).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": key})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        out.append((q, (time.perf_counter() - t) * 1000, d))
    return out


if __name__ == "__main__":
    base, key = sys.argv[1], sys.argv[2]
    with open(sys.argv[3], encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    rows = probe(base, key, queries)
    for q, ms, d in rows:
        t = d["timings_ms"]
        print(f"{ms:6.0f} ms | parse {t['parse_ms']:5.0f} retr {t['retrieve_ms']:5.0f} rerank {t['rerank_ms']:5.0f} "
              f"| {'R' if d['reranked'] else '-'} {d['query_type']:12s} {q}")
    lat = sorted(r[1] for r in rows)
    print(f"n={len(lat)} p50={statistics.median(lat):.0f} p95={lat[int(0.95 * len(lat)) - 1]:.0f} max={lat[-1]:.0f}"
          f" under3s={sum(x < 3000 for x in lat)}/{len(lat)}")
