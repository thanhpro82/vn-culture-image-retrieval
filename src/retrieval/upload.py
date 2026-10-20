"""Đẩy thư mục ảnh WebP lên S3 song song, bỏ qua ảnh đã có (chạy lại được)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from retrieval.storage import ImageStore, make_store


def upload_dir(src: Path, store: ImageStore, workers: int = 32) -> dict:
    files = sorted(Path(src).glob("*.webp"))
    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    def one(p: Path) -> str:
        image_id = p.stem
        try:
            if store.exists(image_id):
                return "skipped"
            store.put(image_id, p.read_bytes())
            return "uploaded"
        except Exception:  # noqa: BLE001 - đếm lỗi từng file rồi chạy tiếp, báo ở cuối
            return "failed"

    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, files), 1):
            counts[r] += 1
            if i % 5000 == 0:
                print(i, "/", len(files), counts, flush=True)
    return counts


if __name__ == "__main__":
    print(upload_dir(Path(sys.argv[1] if len(sys.argv) > 1 else "Data/images"), make_store()))
