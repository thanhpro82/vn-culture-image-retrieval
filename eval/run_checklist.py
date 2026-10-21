"""Chạy bảng câu hỏi kiểm tra tay (eval/datasets/manual-checklist-v1.json) qua /search và tự chấm theo từ khoá mong đợi.

Mỗi câu có `must`: danh sách nhóm từ khoá; một nhóm được thoả nếu BẤT KỲ từ nào của nhóm xuất hiện trong tên thực thể,
chú thích hoặc tiêu đề trang của ảnh trả về (không phân biệt dấu và hoa thường).
  strict: mọi nhóm đều xuất hiện trong top 10.   loose: ít nhất một nhóm xuất hiện trong top 5.
Cũng ghi cờ `low_confidence` (câu nằm trong dữ liệu thì đáng lẽ không bị cảnh báo) và thời gian.

Dùng: python -m eval.run_checklist <API_KEY> [base_url] [checklist.json] [out.json]
Từ khoá chỉ là phép chấm gần đúng: các câu trượt cần xem tay (kết quả đầy đủ nằm ở file out).
"""
import json
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").casefold().split())


def group_hits(texts: list[str], must: list[list[str]]) -> list[bool]:
    folded = [norm(t) for t in texts]
    return [any(norm(w) in t for w in group for t in folded) for group in must]


def grade(texts: list[str], must: list[list[str]]) -> dict:
    top10, top5 = group_hits(texts[:10], must), group_hits(texts[:5], must)
    return {"strict": bool(texts) and all(top10), "loose": bool(texts) and any(top5), "groups": top10}


def hit_text(r: dict) -> str:
    return " ".join([*r.get("entities", []), r.get("caption", "") or "", r.get("page_title", "") or ""])


def call(base: str, key: str, query: str) -> dict:
    body = json.dumps({"query": query, "top_k": 10}).encode("utf-8")
    req = urllib.request.Request(base + "/search", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    d["_ms"] = round((time.perf_counter() - t) * 1000)
    return d


if __name__ == "__main__":
    key = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    items = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else "eval/datasets/manual-checklist-v1.json").read_text(encoding="utf-8"))
    out = []
    for it in items:
        d = call(base, key, it["query"])
        texts = [hit_text(r) for r in d["results"]]
        g = grade(texts, it["must"])
        out.append({**it, **g, "low_confidence": d["low_confidence"], "query_type": d["query_type"], "ms": d["_ms"],
                    "subs": [s["vi"] for s in d["subs"]],
                    "top": [(r["entities"] or ["?"])[0] for r in d["results"]]})
        print(f'{it["id"]} {"OK  " if g["strict"] else ("gan " if g["loose"] else "TRUOT")} {"CO " if d["low_confidence"] else "   "} {d["_ms"]:5d}ms', flush=True)
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
