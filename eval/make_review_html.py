"""Tạo trang HTML duyệt nhanh bộ test từ bảng duyệt CSV (xem eval/make_review_sheet.py), kèm ảnh thu nhỏ của đáp án.

Dùng: python -m eval.make_review_html [review-sheet.csv] [review.html] [testset.jsonl] [manifest.parquet] [thư_mục_ảnh]
Trang tự chứa (ảnh nhúng dạng base64, không cần mạng), tự lưu tiến độ trong trình duyệt, xuất lại CSV có cột `verdict`.
"""
import base64
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

TEMPLATE = Path(__file__).with_name("review_template.html")


def pick_images(image_ids: list[str], names_by_image: dict[str, list[str]], k: int) -> list[str]:
    """Chọn tối đa k ảnh đáp án, mỗi thực thể một ảnh trước khi lấy ảnh thứ hai (thực thể nhiều ảnh nhất đi trước)."""
    by_entity: dict[str, list[str]] = defaultdict(list)
    for i in sorted(set(image_ids)):
        by_entity[(names_by_image.get(i) or ["?"])[0]].append(i)
    order = [n for n, _ in Counter({n: len(v) for n, v in by_entity.items()}).most_common()]
    picked: list[str] = []
    depth = 0
    while len(picked) < k and any(len(by_entity[n]) > depth for n in order):
        for n in order:
            if len(by_entity[n]) > depth and len(picked) < k:
                picked.append(by_entity[n][depth])
        depth += 1
    return picked


def thumb(path: Path, side: int = 200) -> str:
    """Ảnh thu nhỏ dạng data URI JPEG; chuỗi rỗng nếu thiếu file hoặc file hỏng."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
    except (OSError, ValueError):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_rows(sheet: list[dict], tests: dict[str, dict], names_by_image: dict[str, list[str]], images_dir: Path,
               per_intent: int = 5) -> list[dict]:
    rows = []
    for r in sheet:
        intents = tests[r["id"]]["intents"] if r["id"] in tests else []
        imgs = []
        for ids in intents:
            uris = [thumb(Path(images_dir) / f"{i}.webp") for i in pick_images(ids, names_by_image, per_intent)]
            imgs.append([u for u in uris if u])
        rows.append({**r, "imgs": imgs})
    return rows


def render(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<" + chr(92) + "/")  # không để dữ liệu đóng thẻ <script>
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/[]", data)


if __name__ == "__main__":
    a = sys.argv[1:]
    src = Path(a[0]) if len(a) > 0 else Path("eval/datasets/review-sheet-seed0.csv")
    dst = Path(a[1]) if len(a) > 1 else src.with_name(src.stem.replace("sheet", "form") + ".html")
    testset = Path(a[2]) if len(a) > 2 else Path("eval/datasets/testset-v1-seed0.jsonl")
    manifest = Path(a[3]) if len(a) > 3 else Path("Data/manifest.parquet")
    images = Path(a[4]) if len(a) > 4 else Path("Data/images")
    df = pd.read_csv(src, encoding="utf-8-sig", dtype=str).fillna("")
    tests = {t["id"]: t for t in (json.loads(line) for line in testset.read_text(encoding="utf-8").splitlines() if line.strip())}
    m = pd.read_parquet(manifest, columns=["image_id", "entity_names"])
    names = {r.image_id: [str(x) for x in (r.entity_names if r.entity_names is not None else [])] for r in m.itertuples()}
    out_rows = build_rows(df.to_dict("records"), tests, names, images)
    dst.write_text(render(out_rows), encoding="utf-8")
    n_img = sum(len(g) for r in out_rows for g in r["imgs"])
    print(f"{len(df)} dòng, {n_img} ảnh -> {dst} ({dst.stat().st_size // 1024} KB)")
