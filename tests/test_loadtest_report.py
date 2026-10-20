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


def test_cli_writes_utf8_even_when_console_encoding_is_not(tmp_path):
    import os
    import subprocess
    import sys

    src = tmp_path / "s.json"
    src.write_text(json.dumps(SUMMARY), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONIOENCODING"] = "cp1252"  # mô phỏng console Windows
    out = subprocess.run([sys.executable, "-m", "loadtest.report", str(src)], capture_output=True, env=env, check=False)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    assert "Báo cáo tải".encode() in out.stdout


def test_p50_falls_back_to_k6_med_key():
    s = {"metrics": {**SUMMARY["metrics"], "search_ms": {"values": {"med": 777.0, "p(95)": 1.0, "p(99)": 2.0, "max": 3.0}}}}
    assert "777 ms" in render(s)
