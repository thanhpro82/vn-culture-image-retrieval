import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from fakes import FakeEmbedder, hash_vec
from retrieval.api import Services, create_app
from retrieval.config import Settings
from retrieval.grounding import EntityCatalog
from retrieval.index import build_doc, create_collection, upsert_points
from retrieval.storage import LocalStore

ROWS = [
    {"image_id": f"{i:016x}", "entity_ids": [f"e{i}"], "entity_names": [name], "caption": cap}
    for i, (name, cap) in enumerate([("Áo bà ba", "người mặc áo bà ba"), ("Múa hổ", "điệu múa hổ"),
                                     ("Chợ nổi", "chợ nổi miền tây")] * 6, start=1)
]


class FakeLLM:
    def __init__(self, fail=False):
        self.fail = fail

    async def generate(self, prompt, schema):
        if self.fail:
            raise TimeoutError
        return '{"qt":"single","subs":[{"vi":"áo bà ba","en":"ao ba ba","ents":["Áo bà ba"]}]}'


class FakeReranker:
    def score(self, query, docs):
        return [float(len(d)) for d in docs]


@pytest.fixture
def client(tmp_path):
    q = QdrantClient(":memory:")
    create_collection(q, "images")
    emb = FakeEmbedder()
    dense, sparse = emb.text_bge([build_doc(r) for r in ROWS])
    upsert_points(q, "images", ROWS, {"img": np.stack([hash_vec(r["caption"], 768) for r in ROWS]),
                                      "txt": dense, "sparse": sparse})
    import pandas as pd
    cat = EntityCatalog.from_manifest(pd.DataFrame({"entity_names": [r["entity_names"] for r in ROWS]}))
    svc = Services(settings=Settings(_env_file=None, api_key="k"), embedder=emb, reranker=FakeReranker(),
                   qdrant=q, llm=FakeLLM(), catalog=cat, store=LocalStore(tmp_path))
    return TestClient(create_app(svc)), svc


H = {"X-API-Key": "k"}


def test_missing_or_wrong_api_key_401(client):
    c, _ = client
    assert c.post("/search", json={"query": "áo bà ba"}).status_code == 401
    assert c.post("/search", json={"query": "áo bà ba"}, headers={"X-API-Key": "sai"}).status_code == 401


def test_healthz_no_key_needed(client):
    assert client[0].get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("q", ["", "   ", "x" * 501])
def test_bad_query_422(client, q):
    assert client[0].post("/search", json={"query": q}, headers=H).status_code == 422


def test_search_returns_at_most_10_with_urls_and_timings(client):
    r = client[0].post("/search", json={"query": "áo bà ba"}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["results"]) <= 10
    assert body["results"][0]["url"].startswith("/media/")
    assert {"parse_ms", "retrieve_ms", "rerank_ms", "total_ms"} <= set(body["timings_ms"])
    assert body["query_type"] == "single"


def test_search_survives_llm_failure(client):
    c, svc = client
    svc.llm = FakeLLM(fail=True)
    r = c.post("/search", json={"query": "áo bà ba"}, headers=H)
    assert r.status_code == 200 and len(r.json()["results"]) >= 1


def test_embed_image_space_and_text_space(client):
    c, _ = client
    v = c.post("/embed", json={"texts": ["a", "b"]}, headers=H).json()["vectors"]
    assert len(v) == 2 and len(v[0]) == 768
    t = c.post("/embed", json={"texts": ["a"], "space": "text"}, headers=H).json()["vectors"]
    assert len(t[0]) == 1024


def test_rerank_orders_documents(client):
    r = client[0].post("/rerank", json={"query": "q", "documents": ["ngắn", "dài hơn nhiều", "vừa đủ"], "top_n": 2},
                       headers=H).json()["results"]
    assert [x["index"] for x in r] == [1, 2]


def test_api_key_unset_rejects_everything(tmp_path):
    svc = Services(settings=Settings(_env_file=None, api_key=""), embedder=None, reranker=None, qdrant=None,
                   llm=None, catalog=None, store=LocalStore(tmp_path))
    assert TestClient(create_app(svc)).post("/embed", json={"texts": ["a"]}, headers={"X-API-Key": ""}).status_code == 401


def test_root_serves_html_with_search_form(client):
    r = client[0].get("/")
    assert r.status_code == 200 and "<form" in r.text and "text/html" in r.headers["content-type"]


def test_demo_search_needs_no_key_but_api_search_still_does(client):
    c, _ = client
    assert c.post("/demo/search", json={"query": "áo bà ba"}).status_code == 200
    assert c.post("/search", json={"query": "áo bà ba"}).status_code == 401


def test_demo_search_validates_query(client):
    assert client[0].post("/demo/search", json={"query": "  "}).status_code == 422
