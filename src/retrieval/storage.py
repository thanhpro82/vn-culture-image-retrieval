"""Lưu ảnh đã chuẩn hoá: ổ đĩa cục bộ hoặc S3. Cùng một giao diện."""
import sys
from pathlib import Path
from typing import Protocol

import boto3
from botocore.exceptions import ClientError

from retrieval.config import Settings, get_settings


class ImageStore(Protocol):
    def put(self, image_id: str, data: bytes) -> None: ...
    def exists(self, image_id: str) -> bool: ...
    def url(self, image_id: str, expires: int = 3600) -> str: ...


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, image_id: str, data: bytes) -> None:
        (self.root / f"{image_id}.webp").write_bytes(data)

    def exists(self, image_id: str) -> bool:
        return (self.root / f"{image_id}.webp").exists()

    def url(self, image_id: str, expires: int = 3600) -> str:
        return f"/media/{image_id}.webp"


class S3Store:
    def __init__(self, bucket: str, prefix: str, region: str):
        self.bucket, self.prefix = bucket, prefix
        self.client = boto3.client("s3", region_name=region)

    def _key(self, image_id: str) -> str:
        return f"{self.prefix}{image_id}.webp"

    def put(self, image_id: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(image_id), Body=data,
                               ContentType="image/webp")

    def exists(self, image_id: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(image_id))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def url(self, image_id: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(image_id)},
            ExpiresIn=expires)


def make_store(settings: Settings | None = None, local_root: Path = Path("Data/images")) -> ImageStore:
    s = settings or get_settings()
    if s.s3_bucket:
        return S3Store(s.s3_bucket, s.s3_prefix, s.aws_region)
    return LocalStore(local_root)


if __name__ == "__main__" and sys.argv[1:] == ["selftest"]:
    st = make_store()
    if not isinstance(st, S3Store):
        sys.exit("S3_BUCKET chưa đặt")
    st.put("_selftest", b"x" * 1024)
    assert st.exists("_selftest")
    st.client.delete_object(Bucket=st.bucket, Key=st._key("_selftest"))
    print("ok")
