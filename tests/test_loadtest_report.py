import json

from loadtest.report import render

SUMMARY = {"metrics": {
    "search_ms": {"values": {"p(50)": 900.5, "p(95)": 1800.2, "p(99)": 2100.0, "max": 2500.0, "avg": 1000.0}},
    "http_reqs": {"values": {"count": 300, "rate": 4.9}},
    "http_req_failed": {"values": {"rate": 0.0}},
    "search_under_3s": {"values": {"rate": 1.0}},
}}


def test_render_contains_throughput_and_percentiles():
    md = render(SUMMARY)
    assert "4.9" in md and "900" in md and "1800" in md and "2100" in md
    assert "100.0%" in md  # tỉ lệ dưới 3 giây


def test_render_is_json_roundtrip_safe():
    assert render(json.loads(json.dumps(SUMMARY)))
