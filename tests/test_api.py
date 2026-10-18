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


def test_mode_and_rerank_flags(client):
    c, svc = client
    calls = []
    orig = svc.reranker.score
    svc.reranker.score = lambda q, d: calls.append(len(d)) or orig(q, d)
    r = c.post("/search", json={"query": "áo bà ba", "mode": "image", "rerank": False}, headers=H)
    assert r.status_code == 200 and calls == [] and r.json()["timings_ms"]["rerank_ms"] == 0
    assert c.post("/search", json={"query": "áo bà ba", "mode": "bogus"}, headers=H).status_code == 422
    c.post("/search", json={"query": "áo bà ba"}, headers=H)
    assert calls  # mặc định có rerank


def test_budget_default_leaves_margin_below_3s():
    assert Services.__dataclass_fields__['budget_ms'].default <= 2700


def test_rerank_budget_is_16_candidates():
    """Ngân sách rerank theo ADR-002: INT8 + 16 cặp ~0,6 s trên CPU dev (24 cặp fp32 là ~1,8 s)."""
    from retrieval.api import CANDIDATES
    assert CANDIDATES == 16


def test_rerank_skipped_when_latency_budget_would_be_exceeded(client):
    c, svc = client
    calls = []
    orig = svc.reranker.score
    svc.reranker.score = lambda q, d: calls.append(1) or orig(q, d)
    svc.budget_ms = 0  # không còn dư địa
    r = c.post("/search", json={"query": "áo bà ba"}, headers=H).json()
    assert calls == [] and r["reranked"] is False and len(r["results"]) >= 1


def test_rerank_runs_when_budget_allows_and_updates_estimate(client):
    c, svc = client
    svc.rerank_ema_ms = 800.0
    r = c.post("/search", json={"query": "áo bà ba"}, headers=H).json()
    assert r["reranked"] is True
    assert svc.rerank_ema_ms < 800.0  # fake reranker nhanh nên ước lượng giảm


def test_explicit_rerank_false_reports_not_reranked(client):
    r = client[0].post("/search", json={"query": "áo bà ba", "rerank": False}, headers=H).json()
    assert r["reranked"] is False


def test_slow_rerank_is_cut_off_and_results_still_returned(client):
    import time

    c, svc = client

    class Slow:
        def score_pairs(self, pairs):
            time.sleep(2.0)
            return [0.0] * len(pairs)

    svc.reranker = Slow()
    svc.budget_ms = 700
    svc.rerank_ema_ms = 0.0  # guard ước lượng không chặn; chỉ giới hạn cứng mới cứu
    with c:  # giữ vòng lặp sống như uvicorn; nếu không, TestClient đợi luồng nền khi đóng vòng lặp
        t = time.perf_counter()
        r = c.post("/search", json={"query": "áo bà ba"}, headers=H).json()
        elapsed = time.perf_counter() - t
    assert elapsed < 1.6
    assert r["reranked"] is False and len(r["results"]) >= 1
    assert svc.rerank_ema_ms > 0  # lần sau guard biết rerank chậm
