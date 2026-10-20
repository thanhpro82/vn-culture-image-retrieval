"""Khởi chạy thật: uvicorn retrieval.main:create --factory."""
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from qdrant_client import QdrantClient

from retrieval.api import Services, create_app
from retrieval.config import get_settings
from retrieval.embed import Embedder
from retrieval.grounding import EntityCatalog
from retrieval.qparse import GeminiClient
from retrieval.rerank import HttpRerankClient, RemoteFallbackScorer, Reranker
from retrieval.storage import make_store

CATALOG_PATHS = (Path("Data/entities.parquet"), Path("Data/manifest.parquet"))


def build_services() -> Services:
    s = get_settings()
    path = next((p for p in CATALOG_PATHS if p.exists()), None)
    catalog = EntityCatalog.from_manifest(pd.read_parquet(path, columns=["entity_names"])) if path else None
    embedder, reranker = Embedder(device="cpu"), Reranker(threads=int(os.environ.get("TORCH_THREADS", os.cpu_count() or 8)))
    # làm nóng: nạp model lười lần đầu mất ~30 s, không để người dùng đầu tiên chịu
    embedder.text_siglip(["khởi động"])
    embedder.text_bge(["khởi động"])
    reranker.score("khởi động", ["khởi động"])
    search_reranker = None
    if s.ext_rerank_enabled and s.ext_base_url and s.ext_api_key:
        search_reranker = RemoteFallbackScorer(
            reranker, HttpRerankClient(s.ext_base_url, s.ext_api_key, s.ext_rerank_model, s.ext_timeout_ms / 1000),
            rpm=s.ext_rpm)
    return Services(settings=s, embedder=embedder, reranker=reranker, search_reranker=search_reranker,
                    qdrant=QdrantClient(url=s.qdrant_url, timeout=30),
                    llm=GeminiClient(s.gemini_api_key, s.gemini_model), catalog=catalog,
                    store=make_store(s))


def create() -> FastAPI:
    return create_app(build_services())
