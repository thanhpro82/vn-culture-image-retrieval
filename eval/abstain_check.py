"""Đo khả năng cảnh báo `low_confidence` của /search: truy vấn ngoài dữ liệu nên bị gắn cờ, truy vấn trong dữ liệu thì không.

Dùng: python -m eval.abstain_check <API_KEY> [base_url] [testset_v2.jsonl] [oos.jsonl] [số_mẫu_trong_dữ_liệu=45]
Gọi /search với rerank=false (cờ không phụ thuộc rerank) nên không tốn hạn mức rerank ngoài.
"""
import json
import random
import sys
import urllib.request
from pathlib import Path


def flagged(base: str, key: str, query: str) -> bool:
    body = json.dumps({"query": query, "top_k": 10, "rerank": False}).encode()
    req = urllib.request.Request(base + "/search", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return bool(json.load(r)["low_confidence"])


def rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


if __name__ == "__main__":
    key = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    inscope = [json.loads(line) for line in Path(sys.argv[3] if len(sys.argv) > 3 else "eval/datasets/testset-v2-seed1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    oos = [json.loads(line) for line in Path(sys.argv[4] if len(sys.argv) > 4 else "eval/datasets/testset-v2-oos.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 45
    sample = random.Random(0).sample(inscope, min(n, len(inscope)))
    oos_flags = [flagged(base, key, t["query"]) for t in oos]
    in_flags = [flagged(base, key, t["query"]) for t in sample]
    out = {"oos": {"n": len(oos), "flagged_rate": rate(oos_flags),
                   "not_flagged": [t["query"] for t, f in zip(oos, oos_flags, strict=True) if not f]},
           "in_scope": {"n": len(sample), "flagged_rate": rate(in_flags),
                        "wrongly_flagged": [t["query"] for t, f in zip(sample, in_flags, strict=True) if f][:25]}}
    print(json.dumps(out, ensure_ascii=False, indent=2))
