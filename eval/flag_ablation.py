"""So sánh có kiểm soát cờ `low_confidence` cũ và mới: cùng kết quả phân tích của Gemini và cùng ảnh trả về cho mỗi truy vấn.

Cờ cũ: Gemini đề xuất thực thể nhưng không thực thể nào khớp danh mục.
Cờ mới: như cũ, trừ khi một thực thể đề xuất nằm trọn trong một tên thực thể của 5 ảnh đầu (retrieval.grounding.entities_supported).

Chạy trong môi trường có model, Qdrant và khoá Gemini (xem eval/compare_rerank.py):
  python /tmp/flag_ablation.py /tmp/in_scope.jsonl /tmp/oos.jsonl /tmp/flags.json
Truy vấn trong dữ liệu nên KHÔNG bị gắn cờ; truy vấn ngoài dữ liệu nên bị gắn cờ.
"""
import asyncio
import json
import sys
from pathlib import Path

from retrieval import search as S
from retrieval.api import CANDIDATES, LOW_CONF_CHECK
from retrieval.grounding import entities_supported, ground
from retrieval.main import build_services
from retrieval.qparse import parse_query


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


async def flags_for(svc, query: str) -> tuple[bool, bool, list[str]]:
    names = svc.catalog.names if svc.catalog else []
    parsed = await parse_query(query, svc.llm, timeout_ms=svc.settings.gemini_timeout_ms, entity_names=names)
    proposed = [e for s in parsed.subs for e in s.ents]
    grounded = ground(parsed, svc.catalog) if svc.catalog else parsed
    old = bool(proposed) and not any(s.ents for s in grounded.subs)
    hits = S.search(grounded, svc.embedder, svc.qdrant, CANDIDATES, svc.collection, 3, S.BRANCHES, {})
    new = old and not entities_supported(proposed, [n for h in hits[:LOW_CONF_CHECK] for n in (h.entity_names or [])])
    return old, new, proposed


async def main(in_path: str, oos_path: str, out: str) -> None:
    svc = build_services()
    report: dict = {}
    for label, path in (("in_scope", in_path), ("out_of_scope", oos_path)):
        rows = []
        for t in load(path):
            old, new, proposed = await flags_for(svc, t["query"])
            rows.append({"query": t["query"], "old": old, "new": new, "proposed": proposed})
        n = len(rows)
        report[label] = {"n": n, "old_flagged": sum(r["old"] for r in rows), "new_flagged": sum(r["new"] for r in rows),
                         "cleared_by_evidence": [r["query"] for r in rows if r["old"] and not r["new"]][:30]}
        print(label, json.dumps({k: v for k, v in report[label].items() if k != "cleared_by_evidence"}), flush=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
