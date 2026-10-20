"""Sinh bộ test 3 loại query từ manifest (không dùng caption đã index để tránh rò rỉ).

single: tên thực thể. multi_intent: hai thực thể cùng địa bàn ghép bằng "và".
multi_hop (khái niệm): thể loại văn hoá + địa bàn, đáp án là mọi ảnh thuộc nhóm đó.
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

MAX_PHRASE = 40
MIN_IMAGES = 3
MIN_GROUP_ENTITIES = 5


def _strip(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()


def _explode(manifest: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in manifest.itertuples(index=False):
        for name in (r.entity_names if r.entity_names is not None else []):
            recs.append({"image_id": r.image_id, "name": _strip(name), "region": r.region,
                         "culture": r.culture_type})
    return pd.DataFrame(recs)


def _short(v) -> str | None:
    return v if isinstance(v, str) and 2 <= len(v) <= MAX_PHRASE else None


def build_testset(manifest: pd.DataFrame, n_per_type: int = 100, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    ex = _explode(manifest)
    captions = {str(c).casefold() for c in manifest["caption"].dropna()}
    by_name = {n: sorted(set(g["image_id"])) for n, g in ex.groupby("name") if len(n) >= 3}
    by_name = {n: ids for n, ids in by_name.items() if len(ids) >= MIN_IMAGES}
    names = sorted(by_name)
    out: list[dict] = []

    def add(kind, query, intents):
        if query.casefold() in captions or any(t["query"] == query for t in out):
            return False
        out.append({"id": f"{kind[:2]}-{len(out):04d}", "type": kind, "query": query, "intents": intents})
        return True

    for n in rng.sample(names, min(n_per_type, len(names))):
        add("single", n, [by_name[n]])

    region_names: dict[str, set[str]] = {}
    for n, g in ex[ex["name"].isin(by_name)].groupby("name"):
        for reg in {_short(x) for x in g["region"]} - {None}:
            region_names.setdefault(reg, set()).add(n)
    regions = sorted(r for r, s in region_names.items() if len(s) >= 2)
    made, tries = 0, 0
    while regions and made < n_per_type and tries < n_per_type * 20:
        tries += 1
        a, b = rng.sample(sorted(region_names[rng.choice(regions)]), 2)
        if add("multi_intent", f"{a} và {b}", [by_name[a], by_name[b]]):
            made += 1

    groups = []
    for (cul, reg), g in ex.groupby(["culture", "region"]):
        if _short(cul) and _short(reg) and g["name"].nunique() >= MIN_GROUP_ENTITIES:
            groups.append((f"{cul} {reg}".lower(), sorted(set(g["image_id"]))))
    rng.shuffle(groups)
    for q, ids in groups[:n_per_type]:
        add("multi_hop", q, [ids])
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ts = build_testset(pd.read_parquet("Data/manifest.parquet"), n, seed)
    out = Path("eval/datasets") / f"testset-v1-seed{seed}.jsonl"
    out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in ts), encoding="utf-8")
    pd.DataFrame([{"id": t["id"], "type": t["type"], "query": t["query"], "n_gold": sum(map(len, t["intents"])),
                   "verdict": ""} for t in ts]).to_csv(Path("eval/datasets") / f"review-seed{seed}.csv", index=False)
    print(len(ts), "queries ->", out)
