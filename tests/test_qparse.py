import asyncio
import json
import time

from retrieval.qparse import (
    RESPONSE_SCHEMA,
    ParsedQuery,
    build_request,
    fallback_parse,
    parse_query,
)


class FakeClient:
    def __init__(self, reply=None, delay=0.0):
        self.reply, self.delay = reply, delay

    async def generate(self, prompt: str, schema: dict) -> str:
        await asyncio.sleep(self.delay)
        return self.reply


GOOD = json.dumps({"qt": "multi_intent", "subs": [
    {"vi": "hồ bơi ở Phú Quốc", "en": "swimming pool in Phu Quoc", "ents": [], "loc": "Phú Quốc", "cat": ""},
    {"vi": "công viên nước ở Phú Quốc", "en": "water park in Phu Quoc", "ents": [], "loc": "Phú Quốc", "cat": ""},
]})


async def test_valid_json_parsed():
    p = await parse_query("hồ bơi và công viên nước ở Phú Quốc", FakeClient(GOOD))
    assert isinstance(p, ParsedQuery) and p.qt == "multi_intent" and len(p.subs) == 2
    assert p.subs[0].en == "swimming pool in Phu Quoc"


async def test_timeout_uses_fallback():
    t = time.perf_counter()
    p = await parse_query("con hổ và con mèo", FakeClient(GOOD, delay=2.0), timeout_ms=1200)
    assert time.perf_counter() - t < 1.5
    assert p.qt == "multi_intent" and [s.vi for s in p.subs] == ["con hổ", "con mèo"]


async def test_invalid_json_uses_fallback():
    p = await parse_query("bãi biển Nha Trang", FakeClient("{không phải json"))
    assert p.qt == "single" and p.subs[0].vi == "bãi biển Nha Trang"


async def test_unknown_qt_uses_fallback():
    bad = json.dumps({"qt": "weird", "subs": [{"vi": "x", "en": "x", "ents": []}]})
    p = await parse_query("bãi biển Nha Trang", FakeClient(bad))
    assert p.qt == "single" and p.subs[0].vi == "bãi biển Nha Trang"


def test_fallback_splits_on_va_and_keeps_shared_location():
    p = fallback_parse("hồ bơi và công viên nước ở Phú Quốc", [])
    assert p.qt == "multi_intent" and len(p.subs) == 2
    assert all("Phú Quốc" in s.vi for s in p.subs)


def test_fallback_finds_known_entities_ignoring_accents():
    p = fallback_parse("ao ba ba cach tan", ["Áo bà ba cách tân", "Lễ hội Leo cột mỡ"])
    assert p.subs[0].ents == ["Áo bà ba cách tân"]


def test_schema_has_no_dollar_schema_key():
    assert "$schema" not in json.dumps(RESPONSE_SCHEMA)


def test_request_uses_minimal_thinking_level_not_budget():
    body = build_request("câu hỏi")
    cfg = body["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "thinkingBudget" not in json.dumps(body)
    assert cfg["responseSchema"] is RESPONSE_SCHEMA and cfg["temperature"] == 0


def test_schema_asks_only_for_vi_en_ents():
    props = RESPONSE_SCHEMA["properties"]["subs"]["items"]["properties"]
    assert set(props) == {"vi", "en", "ents"}


async def test_gemini_client_reuses_connection_between_calls():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from retrieval.qparse import GeminiClient

    ports = []

    async def handler(request):
        ports.append(request.transport.get_extra_info("peername")[1])
        return web.json_response({"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    app = web.Application()
    app.router.add_post("/v1beta/models/m:generateContent", handler)
    server = TestServer(app)
    await server.start_server()
    client = GeminiClient("k", "m", base_url=str(server.make_url("")).rstrip("/"))
    await client.generate("a", {})
    await client.generate("b", {})
    await client.close()
    await server.close()
    assert len(ports) == 2 and ports[0] == ports[1]


async def test_malformed_response_body_or_shape_never_raises():
    class Boom:
        def __init__(self, exc):
            self.exc = exc

        async def generate(self, prompt, schema):
            raise self.exc

    for exc in (ValueError("json hỏng"), TypeError("sai kiểu"), AttributeError("x")):
        p = await parse_query("bãi biển Nha Trang", Boom(exc))
        assert p.qt == "single" and p.subs[0].vi == "bãi biển Nha Trang"
