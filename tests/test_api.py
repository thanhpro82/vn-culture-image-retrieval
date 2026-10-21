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


def test_search_response_splits_retrieve_into_embed_and_qdrant(client):
    t = client[0].post("/search", json={"query": "áo bà ba"}, headers=H).json()["timings_ms"]
    assert {"embed_ms", "qdrant_ms"} <= set(t)


async def test_cpu_bound_stages_do_not_run_concurrently(client):
    import asyncio
    import threading
    import time

    import httpx

    c, svc = client
    state = {"active": 0, "max": 0}
    lock = threading.Lock()
    inner = svc.embedder

    class Slow:
        def _wrap(self, fn, *a):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.15)
            try:
                return fn(*a)
            finally:
                with lock:
                    state["active"] -= 1

        def text_siglip(self, t):
            return self._wrap(inner.text_siglip, t)

        def text_bge(self, t):
            return self._wrap(inner.text_bge, t)

    svc.embedder = Slow()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as ac:
        rs = await asyncio.gather(*[ac.post("/search", json={"query": f"áo bà ba {i}", "rerank": False},
                                            headers=H) for i in range(3)])
    assert all(r.status_code == 200 for r in rs)
    assert state["max"] == 1


# ---------- sau review toàn nhánh ----------
class EntLLM:
    def __init__(self, ents):
        self.ents = ents

    async def generate(self, prompt, schema):
        import json
        return json.dumps({"qt": "single", "subs": [{"vi": "x", "en": "x", "ents": self.ents}]})


def test_low_confidence_flag_when_every_proposed_entity_is_unknown(client):
    c, svc = client
    svc.llm = EntLLM(["Sư tử", "Cá voi"])
    r = c.post("/search", json={"query": "con sư tử"}, headers=H).json()
    assert r["low_confidence"] is True
    svc.llm = EntLLM(["Áo bà ba"])
    assert c.post("/search", json={"query": "áo bà ba 2"}, headers=H).json()["low_confidence"] is False
    svc.llm = EntLLM([])  # LLM không đề xuất thực thể: tìm bằng văn bản bình thường, không cảnh báo
    assert c.post("/search", json={"query": "chợ nổi"}, headers=H).json()["low_confidence"] is False


def _tracking_pair(svc, embed_s=0.05, rerank_s=0.6):
    import threading
    import time

    state = {"active": 0, "max": 0}
    lock = threading.Lock()
    inner = svc.embedder

    def track(seconds, fn=None, *a):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        try:
            time.sleep(seconds)
            return fn(*a) if fn else None
        finally:
            with lock:
                state["active"] -= 1

    class Emb:
        def text_siglip(self, t):
            return track(embed_s, inner.text_siglip, t)

        def text_bge(self, t):
            return inner.text_bge(t)

    class Rer:
        def score_pairs(self, pairs):
            track(rerank_s)
            return [0.0] * len(pairs)

    svc.embedder, svc.reranker = Emb(), Rer()
    return state


async def test_timed_out_rerank_keeps_cpu_slot_until_its_thread_finishes(client):
    import httpx

    c, svc = client
    state = _tracking_pair(svc)
    svc.budget_ms, svc.rerank_ema_ms = 500, 0.0
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as ac:
        a = await ac.post("/search", json={"query": "áo bà ba"}, headers=H)  # rerank hết hạn giữa chừng
        assert a.json()["reranked"] is False
        b = await ac.post("/search", json={"query": "chợ nổi", "rerank": False}, headers=H)  # embed của B
        assert b.status_code == 200
    assert state["max"] == 1  # embed của B phải đợi luồng rerank bị bỏ rơi chạy xong


async def test_rerank_estimate_uses_measured_runtime_only(client):
    import asyncio

    import httpx

    c, svc = client
    _tracking_pair(svc, rerank_s=0.5)
    svc.budget_ms, svc.rerank_ema_ms = 400, 0.0
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as ac:
        r = await ac.post("/search", json={"query": "áo bà ba"}, headers=H)
        assert r.json()["reranked"] is False
        assert svc.rerank_ema_ms == 0.0  # timeout không tự bơm ước lượng
        await asyncio.sleep(0.8)  # luồng chạy xong
    assert 100 < svc.rerank_ema_ms < 500  # chỉ cập nhật theo thời gian chạy thật


async def test_overloaded_queue_returns_503_with_retry_after(client):
    import asyncio

    import httpx

    c, svc = client
    _tracking_pair(svc, embed_s=0.3)
    svc.max_inflight = 1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as ac:
        rs = await asyncio.gather(*[ac.post("/search", json={"query": f"áo {i}", "rerank": False}, headers=H)
                                    for i in range(3)])
    codes = sorted(r.status_code for r in rs)
    assert codes.count(200) >= 1 and 503 in codes
    assert all("retry-after" in r.headers for r in rs if r.status_code == 503)


def test_demo_endpoint_is_rate_limited_per_client_but_keyed_endpoint_is_not(client):
    c, svc = client
    svc.demo_rate_per_min = 2
    codes = [c.post("/demo/search", json={"query": "áo bà ba"}, headers={"x-forwarded-for": "1.2.3.4"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]
    other = c.post("/demo/search", json={"query": "áo bà ba"}, headers={"x-forwarded-for": "5.6.7.8"})
    assert other.status_code == 200  # người dùng khác không bị ảnh hưởng
    assert all(c.post("/search", json={"query": "áo bà ba"}, headers=H).status_code == 200 for _ in range(3))


def test_ui_handles_low_confidence_and_throttling(client):
    html = client[0].get("/").text
    assert "low_confidence" in html and "429" in html and "503" in html


# ---------- gia cố sau khi triển khai thử ----------
def test_non_ascii_api_key_is_401_not_500(client):
    c, _ = client
    r = c.post("/search", json={"query": "áo bà ba"}, headers={"X-API-Key": b"k" + bytes([0xF3])})  # byte 0xF3: không phải ASCII
    assert r.status_code == 401


def test_healthz_reports_unhealthy_when_qdrant_is_down(client):
    c, svc = client

    class Down:
        def get_collection(self, *a, **k):
            raise ConnectionError("qdrant tắt")

    svc.qdrant = Down()
    r = c.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


def test_healthz_ok_when_collection_exists(client):
    c, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_embed_and_rerank_endpoints_share_the_cpu_slot(client):
    import asyncio
    import threading
    import time

    import httpx

    c, svc = client
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def tracked(fn):
        def inner(*a, **k):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.15)
            try:
                return fn(*a, **k)
            finally:
                with lock:
                    state["active"] -= 1
        return inner

    svc.embedder.text_siglip = tracked(svc.embedder.text_siglip)
    svc.reranker.score = tracked(svc.reranker.score)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as ac:
        rs = await asyncio.gather(
            ac.post("/embed", json={"texts": ["a"], "space": "image"}, headers=H),
            ac.post("/rerank", json={"query": "q", "documents": ["x", "yy"]}, headers=H),
            ac.post("/embed", json={"texts": ["b"], "space": "image"}, headers=H))
    assert all(r.status_code == 200 for r in rs)
    assert state["max"] == 1


def test_low_confidence_cleared_when_results_visibly_contain_the_proposed_entity(client):
    """Gemini đề xuất tên không khớp danh mục theo cụm liền nhau, nhưng ảnh trả về đúng là thực thể đó: không cảnh báo."""
    import json

    class SwappedWordsLLM:
        async def generate(self, prompt, schema):
            return json.dumps({"qt": "single", "subs": [{"vi": "chợ nổi miền tây", "en": "chợ nổi miền tây", "ents": ["nổi chợ"]}]})

    c, svc = client
    svc.llm = SwappedWordsLLM()
    r = c.post("/search", json={"query": "chợ nổi miền tây", "rerank": False}, headers=H).json()
    assert any("Chợ nổi" in " ".join(x["entities"]) for x in r["results"][:5])
    assert r["subs"][0]["ents"] == []  # tên đề xuất vẫn không khớp danh mục (đúng như mô tả test)
    assert r["low_confidence"] is False


def test_low_confidence_cleared_when_entity_words_are_spread_over_several_top_results(client):
    """Thực thể đề xuất không nằm trọn trong một tên ("Văn Miếu Quốc Tử Giám" so với "Văn Miếu ..." và "Quốc Tử Giám ..."),
    nhưng các từ của nó xuất hiện đủ trong tên thực thể của 10 ảnh đầu: không cảnh báo."""
    import json

    class SpreadWordsLLM:
        async def generate(self, prompt, schema):
            return json.dumps({"qt": "single", "subs": [{"vi": "chợ nổi miền tây", "en": "chợ nổi miền tây", "ents": ["nổi hổ"]}]})

    c, svc = client
    svc.llm = SpreadWordsLLM()
    r = c.post("/search", json={"query": "chợ nổi miền tây", "rerank": False}, headers=H).json()
    names = [n for x in r["results"][:10] for n in x["entities"]]
    assert "Chợ nổi" in names and "Múa hổ" in names   # 'nổi' và 'hổ' nằm ở hai tên khác nhau
    assert r["subs"][0]["ents"] == [] and r["low_confidence"] is False
