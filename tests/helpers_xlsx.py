"""Tạo workbook giả giống file của team 3 để test manifest."""
import re
import zipfile
from pathlib import Path

import openpyxl

SUMMARY = "Thực thể (Summary)"
APPRAISED = "Ảnh đã thẩm định"
SUMMARY_COLS = ["STT (Batch)", "Entity ID", "Tên Thực thể", "Nhánh (Branch)", "Loại Thực thể",
                "Địa bàn", "Thể loại Văn hóa", "Mô tả sơ qua về thực thể"]
APPRAISED_COLS = ["STT (Batch)", "Tên Thực thể", "Entity ID", "Thứ tự ảnh", "Đại diện?",
                  "Điểm khớp", "Mô tả ảnh (Gemini Vision)", "URL Ảnh", "Độ phân giải",
                  "Trang nguồn", "Tiêu đề trang nguồn", "Tên miền", "Loại nguồn", "Whitelist",
                  "Lý do thẩm định", "Trích đoạn nội dung", "pHash", "dHash", "Người verify",
                  "Giữ/Bỏ", "Verify note"]


def entity(eid, name, branch="B", region="Nam Bộ", culture="Trang phục", desc="mô tả"):
    return [1, eid, name, branch, "Danh thức", region, culture, desc]


def image(eid, name, url, label=None, caption="cap", phash="aa"):
    row = dict.fromkeys(APPRAISED_COLS)
    row.update({"Tên Thực thể": name, "Entity ID": eid, "URL Ảnh": url,
                "Mô tả ảnh (Gemini Vision)": caption, "Tiêu đề trang nguồn": "tiêu đề",
                "Tên miền": "ex.vn", "Độ phân giải": "800x600", "pHash": phash, "dHash": "bb",
                "Giữ/Bỏ": label})
    return [row[c] for c in APPRAISED_COLS]


def write_book(path: Path, entities: list, images: list, strip_dimension: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SUMMARY
    ws.append(SUMMARY_COLS)
    for e in entities:
        ws.append(e)
    ws2 = wb.create_sheet(APPRAISED)
    ws2.append(APPRAISED_COLS)
    for i in images:
        ws2.append(i)
    wb.save(path)
    if strip_dimension:  # file thật của team không có <dimension>, dòng bị cắt ô cuối
        tmp = path.with_suffix(".tmp")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.startswith("xl/worksheets/"):
                    data = re.sub(rb"<dimension[^>]*/>", b"", data)
                zout.writestr(item, data)
        tmp.replace(path)
