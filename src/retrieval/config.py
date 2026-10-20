from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-southeast-1"
    s3_bucket: str = ""
    s3_prefix: str = "images/"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_ms: int = 1600
    api_key: str = ""
    # Endpoint rerank ngoài (mentor cấp, hạn mức chung): tắt mặc định, bật bằng EXT_RERANK_ENABLED=true
    ext_rerank_enabled: bool = False
    ext_base_url: str = ""
    ext_api_key: str = ""
    ext_rerank_model: str = "jina-reranker-v2"
    ext_rpm: int = 40
    ext_timeout_ms: int = 800
    qdrant_url: str = "http://127.0.0.1:6333"  # 127.0.0.1: localhost trên Windows thử IPv6 trước


@lru_cache
def get_settings() -> Settings:
    return Settings()
