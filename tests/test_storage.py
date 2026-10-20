import boto3
from moto import mock_aws

from retrieval.config import Settings
from retrieval.storage import LocalStore, S3Store, make_store


def test_local_roundtrip(tmp_path):
    st = LocalStore(tmp_path)
    assert not st.exists("abc")
    st.put("abc", b"data")
    assert st.exists("abc")
    assert st.url("abc") == "/media/abc.webp"
    assert (tmp_path / "abc.webp").read_bytes() == b"data"


@mock_aws
def test_s3_roundtrip_and_url_is_presigned(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    boto3.client("s3", region_name="ap-southeast-1").create_bucket(
        Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "ap-southeast-1"})
    st = S3Store("test-bucket", "images/", "ap-southeast-1")
    assert not st.exists("abc")
    st.put("abc", b"data")
    assert st.exists("abc")
    url = st.url("abc", expires=600)
    assert "images/abc.webp" in url and "Signature" in url


def test_make_store_without_bucket_is_local(tmp_path):
    st = make_store(Settings(_env_file=None, s3_bucket=""), local_root=tmp_path)
    assert isinstance(st, LocalStore)


def test_make_store_with_bucket_is_s3():
    st = make_store(Settings(_env_file=None, s3_bucket="test-bucket", aws_region="ap-southeast-1"))
    assert isinstance(st, S3Store)
