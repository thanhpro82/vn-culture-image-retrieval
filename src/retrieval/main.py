"""Khởi chạy thật: uvicorn retrieval.main:create --factory."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from qdrant_client import QdrantClient

from retrieval.api import Services, create_app
from retrieval.config import get_settings
from retrieval.embed import Embedder
from retrieval.grounding import EntityCatalog
from retrieval.qparse import GeminiClient
from retrieval.rerank import Reranker
from retrieval.storage import make_store

CATALOG_PATHS = (Path("Data/entities.parquet"), Path("Data/manifest.parquet"))


def build_services() -> Services:
    s = get_settings()
    path = next((p for p in CATALOG_PATHS if p.exists()), None)
    catalog = EntityCatalog.from_manifest(pd.read_parquet(path, columns=["entity_names"])) if path else None
    return Services(settings=s, embedder=Embedder(device="cpu"), reranker=Reranker(),
                    qdrant=QdrantClient(url=s.qdrant_url, timeout=30),
                    llm=GeminiClient(s.gemini_api_key, s.gemini_model), catalog=catalog,
                    store=make_store(s))


def create() -> FastAPI:
    return create_app(build_services())
