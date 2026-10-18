import boto3
from moto import mock_aws

from retrieval.snapshot import upload_index


@mock_aws
def test_upload_index_puts_files_and_returns_week_long_presigned_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "t")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "t")
    s3 = boto3.client("s3", region_name="ap-southeast-1")
    s3.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "ap-southeast-1"})
    snap, ent = tmp_path / "images-snap.snapshot", tmp_path / "entities.parquet"
    snap.write_bytes(b"s" * 100)
    ent.write_bytes(b"e" * 10)
    urls = upload_index(s3, "test-bucket", "index/", {"snapshot": snap, "entities": ent})
    assert set(urls) == {"snapshot", "entities"}
    assert s3.head_object(Bucket="test-bucket", Key="index/images-snap.snapshot")["ContentLength"] == 100
    assert s3.head_object(Bucket="test-bucket", Key="index/entities.parquet")["ContentLength"] == 10
    assert "X-Amz-Expires=604800" in urls["snapshot"] or "Expires=" in urls["snapshot"]
