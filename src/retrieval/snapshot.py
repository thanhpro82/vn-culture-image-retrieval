"""Tạo snapshot Qdrant, đẩy lên S3 cùng danh mục thực thể, và khôi phục ở máy khác.

Dùng:
  python -m retrieval.snapshot publish            # tạo snapshot, tải lên S3, in link ký sẵn (7 ngày)
  python -m retrieval.snapshot restore SNAP_URL ENTITIES_URL
"""
import sys
import urllib.request
from pathlib import Path

import boto3
from qdrant_client import QdrantClient

from retrieval.config import get_settings

WEEK = 7 * 24 * 3600


def create_snapshot(qdrant_url: str, collection: str, dest_dir: Path) -> Path:
    client = QdrantClient(url=qdrant_url, timeout=900)
    name = client.create_snapshot(collection_name=collection).name
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{qdrant_url}/collections/{collection}/snapshots/{name}", dest)
    return dest


def upload_index(s3, bucket: str, prefix: str, files: dict[str, Path]) -> dict[str, str]:
    """Tải các file lên S3 và trả link ký sẵn cho từng file (tối đa 7 ngày với khoá IAM)."""
    urls = {}
    for label, path in files.items():
        key = f"{prefix}{Path(path).name}"
        s3.upload_file(str(path), bucket, key)
        urls[label] = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                                ExpiresIn=WEEK)
    return urls


def restore(qdrant_url: str, collection: str, snapshot_url: str, entities_url: str, entities_dest: Path) -> None:
    Path(entities_dest).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(entities_url, entities_dest)
    QdrantClient(url=qdrant_url, timeout=1800).recover_snapshot(collection_name=collection,
                                                                location=snapshot_url, wait=True)


if __name__ == "__main__":
    s = get_settings()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "publish":
        snap = create_snapshot(s.qdrant_url, "images", Path("Data/snapshots"))
        s3 = boto3.client("s3", region_name=s.aws_region)
        urls = upload_index(s3, s.s3_bucket, "index/", {"snapshot": snap, "entities": Path("Data/entities.parquet")})
        print("SNAPSHOT_URL=" + urls["snapshot"])
        print("ENTITIES_URL=" + urls["entities"])
    elif cmd == "restore" and len(sys.argv) == 4:
        restore(s.qdrant_url, "images", sys.argv[2], sys.argv[3], Path("Data/entities.parquet"))
        print("restored")
    else:
        sys.exit(__doc__)
