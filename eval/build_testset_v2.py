"""Bộ test v2: truy vấn viết bằng mô tả tự nhiên, không chứa tên thực thể (bộ v1 dùng chính tên thực thể nên dễ).

single: Gemini viết một câu mô tả đối tượng (từ mô tả thực thể) mà không nhắc tên.
multi_intent: hai đối tượng khác thể loại, một câu tự nhiên gồm hai ý, không nhắc tên.
multi_hop: yêu cầu mức khái niệm cho một nhóm (thể loại + địa bàn), người dùng không biết tên cụ thể.
Đáp án: ảnh của thực thể (hoặc của nhóm). Truy vấn do LLM sinh nên cần người duyệt (eval/make_review_sheet.py).
Truy vấn ngoài dữ liệu (không có đáp án) nằm ở file riêng, xem OUT_OF_SCOPE.

Dùng: python -m eval.build_testset_v2 [số_mỗi_loại=40] [seed=1]   (cần GEMINI_API_KEY)
"""
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

MIN_IMAGES = 3
MIN_DESC = 40
MIN_GROUP_ENTITIES = 5
MIN_GROUP_IMAGES = 8
SCHEMA = {"type": "OBJECT", "properties": {"q": {"type": "STRING"}}, "required": ["q"]}
BASE = "Bạn giúp tạo câu truy vấn tìm ảnh bằng tiếng Việt cho hệ thống tìm ảnh di sản văn hoá. "
ONLY_JSON = ' Chỉ trả JSON {"q": "..."}.'

# Chủ đề không có trong dữ liệu di sản văn hoá: hệ thống nên cảnh báo độ tin cậy thấp.
OUT_OF_SCOPE = [
    "ảnh con hổ trong rừng", "bãi biển Nha Trang", "hồ bơi và công viên nước ở Phú Quốc",
    "ảnh con trai chủ tịch Phạm Nhật Vượng", "xe điện VinFast", "đội tuyển bóng đá Việt Nam ăn mừng bàn thắng",
    "ca sĩ biểu diễn trên sân khấu hiện đại", "tòa nhà chọc trời ở Thành phố Hồ Chí Minh về đêm",
    "máy bay đang hạ cánh xuống sân bay", "chó và mèo thú cưng trong nhà", "sân golf ở Đà Lạt",
    "cầu thủ Quang Hải", "cảnh quay phim siêu anh hùng", "con voi châu Phi ở thảo nguyên",
    "ảnh vệ tinh Trái Đất", "xe đua công thức một", "người mẫu trình diễn thời trang cao cấp",
    "khu trượt tuyết ở châu Âu", "robot công nghiệp trong nhà máy", "tàu vũ trụ phóng lên quỹ đạo",
]


def _strip(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()


def leaks(query: str, names: list[str]) -> bool:
    """Truy vấn có lộ tên thực thể không: chứa nguyên văn tên (bỏ phần trong ngoặc) hoặc 3 từ liên tiếp của tên."""
    q = " ".join(query.casefold().split())
    for raw in names:
        n = " ".join(_strip(raw).casefold().split())
        if not n:
            continue
        if n in q:
            return True
        w = n.split()
        if len(w) >= 3 and any(" ".join(w[i:i + 3]) in q for i in range(len(w) - 2)):
            return True
    return False


def _explode(manifest: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in manifest.itertuples(index=False):
        for name in (r.entity_names if r.entity_names is not None else []):
            recs.append({"image_id": r.image_id, "name": _strip(name), "region": r.region, "culture": r.culture_type,
                         "desc": r.entity_desc if isinstance(r.entity_desc, str) else ""})
    return pd.DataFrame(recs)


async def _ask(llm, prompt: str, names: list[str], retries: int = 3) -> str | None:
    """Hỏi LLM một câu truy vấn; nếu lộ tên thì nhắc lại, tối đa `retries` lần."""
    extra = ""
    for _ in range(retries):
        try:
            q = json.loads(await llm.generate(prompt + extra, SCHEMA)).get("q", "").strip()
        except (json.JSONDecodeError, KeyError, TimeoutError, OSError):
            continue
        if q and not leaks(q, names):
            return q
        extra = "\nLần trước bạn đã nhắc tên đối tượng. Viết lại, tuyệt đối không dùng tên (kể cả một phần)."
    return None


def _entity_block(e: dict) -> str:
    return f"- Tên: {e['name']}\n- Thể loại: {e['culture']}\n- Địa bàn: {e['region']}\n- Mô tả: {e['desc'][:400]}"


async def build_v2(manifest: pd.DataFrame, llm, per_type: int = 40, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    ex = _explode(manifest)
    ents: dict[str, dict] = {}
    for name, g in ex.groupby("name"):
        ids = sorted(set(g["image_id"]))
        desc = max(g["desc"], key=len)
        if len(name) >= 3 and len(ids) >= MIN_IMAGES and len(desc) >= MIN_DESC:
            ents[name] = {"name": name, "ids": ids, "desc": desc, "culture": g["culture"].iloc[0], "region": g["region"].iloc[0]}
    names = sorted(ents)
    out: list[dict] = []

    def add(kind: str, query: str, intents: list[list[str]]) -> None:
        out.append({"id": f"v2-{kind[:2]}-{len(out):04d}", "type": kind, "query": query, "intents": intents, "set": "v2"})

    made = 0
    for n in rng.sample(names, min(len(names), per_type * 3)):
        if made >= per_type:
            break
        e = ents[n]
        q = await _ask(llm, BASE + f"Đối tượng cần tìm ảnh:\n{_entity_block(e)}\nViết 1 câu truy vấn ngắn (5 đến 15 từ) như một người "
                       "dùng bình thường sẽ gõ để tìm ảnh về đối tượng này, mô tả bằng đặc điểm, hoạt động, đồ vật hoặc bối cảnh. "
                       "TUYỆT ĐỐI không chứa tên đối tượng (kể cả một phần tên)." + ONLY_JSON, [n])
        if q:
            add("single", q, [e["ids"]])
            made += 1

    made = 0
    for a in rng.sample(names, len(names)):
        if made >= per_type:
            break
        others = [b for b in rng.sample(names, min(len(names), 30)) if ents[b]["culture"] != ents[a]["culture"]]
        if not others:
            continue
        b = others[0]
        q = await _ask(llm, BASE + f"Hai đối tượng cần tìm ảnh:\nĐối tượng 1:\n{_entity_block(ents[a])}\nĐối tượng 2:\n{_entity_block(ents[b])}\n"
                       "Viết 1 câu truy vấn tự nhiên gồm HAI ý (mỗi ý mô tả một đối tượng bằng đặc điểm hoặc hoạt động, nối bằng "
                       "\"và\" hoặc cách nói tự nhiên). TUYỆT ĐỐI không chứa tên của cả hai đối tượng (kể cả một phần tên)." + ONLY_JSON, [a, b])
        if q:
            add("multi_intent", q, [ents[a]["ids"], ents[b]["ids"]])
            made += 1

    groups = []
    for (cul, reg), g in ex[ex["name"].isin(ents)].groupby(["culture", "region"]):
        gn, gi = sorted(set(g["name"])), sorted(set(g["image_id"]))
        if isinstance(cul, str) and isinstance(reg, str) and len(gn) >= MIN_GROUP_ENTITIES and len(gi) >= MIN_GROUP_IMAGES:
            groups.append((cul, reg, gn, gi))
    rng.shuffle(groups)
    made = 0
    for cul, reg, gn, gi in groups:
        if made >= per_type:
            break
        sample = rng.sample(gn, min(5, len(gn)))
        q = await _ask(llm, BASE + f"Nhóm ảnh: thể loại \"{cul}\", địa bàn \"{reg}\". Ví dụ thực thể trong nhóm: {', '.join(sample)}.\n"
                       "Viết 1 yêu cầu tìm ảnh ở mức khái niệm (6 đến 15 từ), như người dùng không biết tên cụ thể và muốn xem các "
                       "đối tượng thuộc nhóm này. Không nhắc tên các thực thể ví dụ và không chép nguyên văn cụm thể loại." + ONLY_JSON, sample)
        if q:
            add("multi_hop", q, [gi])
            made += 1
    return out


async def _main(per_type: int, seed: int) -> None:
    from retrieval.config import get_settings
    from retrieval.qparse import GeminiClient
    s = get_settings()
    llm = GeminiClient(s.gemini_api_key, s.gemini_model)
    try:
        m = pd.read_parquet("Data/manifest.parquet", columns=["image_id", "entity_names", "region", "culture_type", "entity_desc"])
        tests = await build_v2(m, llm, per_type, seed)
    finally:
        await llm.close()
    d = Path("eval/datasets")
    (d / f"testset-v2-seed{seed}.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tests), encoding="utf-8")
    oos = [{"id": f"v2-oos-{i:02d}", "type": "out_of_scope", "query": q, "intents": [], "set": "v2"} for i, q in enumerate(OUT_OF_SCOPE)]
    (d / "testset-v2-oos.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in oos), encoding="utf-8")
    print(len(tests), "truy vấn +", len(oos), "ngoài dữ liệu ->", d)


if __name__ == "__main__":
    asyncio.run(_main(int(sys.argv[1]) if len(sys.argv) > 1 else 40, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
