"""Tạo bảng duyệt tay một mẫu bộ test: mỗi truy vấn kèm các thực thể đáp án để người duyệt đánh giá.

Dùng: python -m eval.make_review_sheet [testset.jsonl] [manifest.parquet] [số_mỗi_loại=15] [seed=0]
Ghi `eval/datasets/review-sheet-seed<seed>.csv` (UTF-8 có BOM để Excel đọc đúng tiếng Việt).
Cột `verdict` do người duyệt điền: dung / sai / mo_ho (xem eval/datasets/README.md).
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

TOP_ENTITIES = 5


def gold_entities(image_ids: list[str], names_by_image: dict[str, list[str]]) -> list[tuple[str, int]]:
    """Thực thể thuộc các ảnh đáp án, theo số ảnh giảm dần."""
    c: Counter[str] = Counter()
    for i in image_ids:
        c.update(names_by_image.get(i, []))
    return c.most_common()


def build_sheet(tests: list[dict], names_by_image: dict[str, list[str]], per_type: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for kind in ("single", "multi_intent", "multi_hop"):
        pool = sorted((t for t in tests if t["type"] == kind), key=lambda t: t["id"])
        for t in rng.sample(pool, min(per_type, len(pool))):
            intents = []
            for k, ids in enumerate(t["intents"], 1):
                ents = gold_entities(ids, names_by_image)
                top = "; ".join(f"{n} ({c})" for n, c in ents[:TOP_ENTITIES])
                label = f"Ý {k}" if len(t["intents"]) > 1 else "Đáp án"
                intents.append(f"{label} [{len(ids)} ảnh, {len(ents)} thực thể]: {top}")
            rows.append({"id": t["id"], "type": kind, "query": t["query"], "gold": " | ".join(intents),
                         "verdict": "", "note": ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    testset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/datasets/testset-v1-seed0.jsonl")
    manifest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Data/manifest.parquet")
    per_type = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tests = [json.loads(line) for line in testset.read_text(encoding="utf-8").splitlines() if line.strip()]
    m = pd.read_parquet(manifest, columns=["image_id", "entity_names"])
    names = {r.image_id: [str(x) for x in (r.entity_names if r.entity_names is not None else [])] for r in m.itertuples()}
    out = Path("eval/datasets") / f"review-sheet-seed{seed}.csv"
    build_sheet(tests, names, per_type, seed).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{per_type * 3} dòng -> {out}")
