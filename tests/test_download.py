import io

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from PIL import Image

from retrieval.download import fetch_image, near_duplicates, to_webp


def _jpeg(size=(100, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


async def _serve(handlers):
    app = web.Application()
    for path, h in handlers.items():
        app.router.add_get(path, h)
    server = TestServer(app)
    await server.start_server()
    return server


async def test_404_returns_none():
    async def nf(_):
        return web.Response(status=404)

    server = await _serve({"/a.jpg": nf})
    async with ClientSession() as s:
        assert await fetch_image(s, str(server.make_url("/a.jpg"))) is None
    await server.close()


async def test_truncated_image_returns_none():
    async def cut(_):
        return web.Response(body=_jpeg()[:40], content_type="image/jpeg")

    server = await _serve({"/a.jpg": cut})
    async with ClientSession() as s:
        assert await fetch_image(s, str(server.make_url("/a.jpg"))) is None
    await server.close()


async def test_retries_on_503_then_ok():
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            return web.Response(status=503)
        return web.Response(body=_jpeg(), content_type="image/jpeg")

    server = await _serve({"/a.jpg": flaky})
    async with ClientSession() as s:
        data = await fetch_image(s, str(server.make_url("/a.jpg")))
    await server.close()
    assert data is not None and calls["n"] == 3


def test_to_webp_limits_side():
    out = Image.open(io.BytesIO(to_webp(_jpeg((2000, 1000)), max_side=640)))
    assert out.format == "WEBP" and max(out.size) == 640


def test_groups_images_within_threshold():
    ph = {"a": "0000000000000000", "b": "0000000000000003", "c": "ffffffffffffffff"}
    groups = near_duplicates(ph, max_hamming=2)
    assert groups == [{"a", "b"}]


def test_no_groups_when_all_far():
    ph = {"a": "0000000000000000", "b": "ffffffffffffffff"}
    assert near_duplicates(ph, max_hamming=10) == []
