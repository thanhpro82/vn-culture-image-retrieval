"""So sánh có kiểm soát các quy tắc cho cờ `low_confidence`: cùng kết quả phân tích của Gemini và cùng ảnh trả về.

Cờ cũ ("old"): Gemini đề xuất thực thể nhưng không thực thể nào khớp danh mục.
Các quy tắc còn lại bỏ cờ khi thực thể đề xuất "được hậu thuẫn" bởi tên thực thể của các ảnh đầu
(retrieval.grounding.entities_supported với `threshold`, `pooled` và số ảnh khác nhau).

Chạy trong môi trường có model, Qdrant và khoá Gemini (xem eval/compare_rerank.py):
  python /tmp/flag_ablation.py /tmp/in_scope.jsonl /tmp/oos.jsonl /tmp/flags.json
`in_scope.jsonl`: mỗi dòng có `query` và (tuỳ chọn) `group`. Truy vấn trong dữ liệu nên KHÔNG bị cờ; ngoài dữ liệu nên bị cờ.
"""
import asyncio
import json
import sys
from pathlib import Path

from retrieval import search as S
from retrieval.api import CANDIDATES
from retrieval.grounding import entities_supported, ground
from retrieval.main import build_services
from retrieval.qparse import parse_query

# tên quy tắc -> (số ảnh đầu xét, ngưỡng độ phủ, gộp từ của các tên)
RULES = {
    "single5_full": (5, 1.0, False),   # quy tắc đang dùng
    "single10_full": (10, 1.0, False),
    "pooled10_full": (10, 1.0, True),
    "single10_075": (10, 0.75, False),
    "single10_060": (10, 0.6, False),
    "pooled10_075": (10, 0.75, True),
    "pooled10_060": (10, 0.6, True),
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


async def evaluate(svc, query: str) -> dict:
    names = svc.catalog.names if svc.catalog else []
    parsed = await parse_query(query, svc.llm, timeout_ms=svc.settings.gemini_timeout_ms, entity_names=names)
    proposed = [e for s in parsed.subs for e in s.ents]
    grounded = ground(parsed, svc.catalog) if svc.catalog else parsed
    old = bool(proposed) and not any(s.ents for s in grounded.subs)
    flags = {"old": old}
    if old:
        hits = S.search(grounded, svc.embedder, svc.qdrant, CANDIDATES, svc.collection, 3, S.BRANCHES, {})
        for name, (k, thr, pooled) in RULES.items():
            top = [n for h in hits[:k] for n in (h.entity_names or [])]
            flags[name] = not entities_supported(proposed, top, thr, pooled)
    else:
        flags.update({name: False for name in RULES})
    return {"query": query, "proposed": proposed, "flags": flags}


async def main(in_path: str, oos_path: str, out: str) -> None:
    svc = build_services()
    rows = {"__oos__": []}
    for t in load(in_path):
        rows.setdefault(t.get("group", "in_scope"), []).append(await evaluate(svc, t["query"]))
    for t in load(oos_path):
        rows["__oos__"].append(await evaluate(svc, t["query"]))
    report: dict = {}
    for group, items in rows.items():
        report[group] = {"n": len(items), **{rule: sum(r["flags"][rule] for r in items) for rule in ["old", *RULES]},
                         "wrongly_flagged_by_old": [r["query"] for r in items if r["flags"]["old"]][:40]}
        print(group, json.dumps({k: v for k, v in report[group].items() if k != "wrongly_flagged_by_old"}), flush=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
