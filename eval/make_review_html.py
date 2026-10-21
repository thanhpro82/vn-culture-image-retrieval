"""Tạo trang HTML duyệt nhanh bộ test từ bảng duyệt CSV (xem eval/make_review_sheet.py).

Dùng: python -m eval.make_review_html [review-sheet.csv] [review.html]
Trang tự chứa (không cần mạng), tự lưu tiến độ trong trình duyệt, xuất lại CSV có cột `verdict`.
"""
import json
import sys
from pathlib import Path

import pandas as pd

TEMPLATE = Path(__file__).with_name("review_template.html")


def render(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<" + chr(92) + "/")  # không để dữ liệu đóng thẻ <script>
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/[]", data)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/datasets/review-sheet-seed0.csv")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem.replace("sheet", "form") + ".html")
    df = pd.read_csv(src, encoding="utf-8-sig", dtype=str).fillna("")
    dst.write_text(render(df.to_dict("records")), encoding="utf-8")
    print(f"{len(df)} dòng -> {dst}")
