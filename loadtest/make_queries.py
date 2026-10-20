"""Xuất danh sách query cho k6 từ bộ test (trộn đều 3 loại, xoay vòng)."""
import json
import sys
from itertools import zip_longest
from pathlib import Path

src = Path(sys.argv[1])
tests = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
by_type: dict[str, list[str]] = {}
for t in tests:
    by_type.setdefault(t["type"], []).append(t["query"])
mixed = [q for group in zip_longest(*by_type.values()) for q in group if q]
Path("loadtest/queries.json").write_text(json.dumps(mixed, ensure_ascii=False), encoding="utf-8")
print(len(mixed), "queries")
