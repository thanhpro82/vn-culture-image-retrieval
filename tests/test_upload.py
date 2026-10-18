import boto3
from moto import mock_aws

from retrieval.storage import S3Store
from retrieval.upload import upload_dir


@mock_aws
def test_upload_dir_uploads_all_and_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "t")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "t")
    boto3.client("s3", region_name="ap-southeast-1").create_bucket(
        Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "ap-southeast-1"})
    for i in range(5):
        (tmp_path / f"{i:016x}.webp").write_bytes(b"x" * 10)
    (tmp_path / "ignore.txt").write_bytes(b"no")
    store = S3Store("test-bucket", "images/", "ap-southeast-1")
    first = upload_dir(tmp_path, store, workers=4)
    assert first == {"uploaded": 5, "skipped": 0, "failed": 0}
    second = upload_dir(tmp_path, store, workers=4)
    assert second == {"uploaded": 0, "skipped": 5, "failed": 0}
    assert store.exists(f"{3:016x}")
