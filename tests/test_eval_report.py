from eval.make_report import render

BASE = {"n": 300, "by_type": {
    "single": {"n": 100, "recall@10": 0.93, "mrr@10": 0.8, "latency_ms": {"p50": 2100, "p95": 2700, "max": 2900}},
    "multi_intent": {"n": 100, "recall@10": 0.99, "mrr@10": 0.9, "intent_coverage@10": 0.915,
                     "latency_ms": {"p50": 2300, "p95": 2800, "max": 2950}},
    "multi_hop": {"n": 100, "recall@10": 0.34, "mrr@10": 0.2, "latency_ms": {"p50": 2400, "p95": 2800, "max": 3000}},
}}


def test_render_has_a_row_per_run_and_type_with_metrics():
    md = render({"Chỉ ảnh": {**BASE, "mode": "image", "rerank": False},
                 "Hybrid": {**BASE, "mode": "hybrid", "rerank": True}})
    assert md.count("| Chỉ ảnh |") == 3 and md.count("| Hybrid |") == 3
    assert "0.930" in md and "0.915" in md and "2100" in md


def test_render_marks_missing_coverage_with_dash():
    md = render({"X": {**BASE, "mode": "hybrid", "rerank": True}})
    line = next(row for row in md.splitlines() if "| single |" in row)
    assert line.rstrip().endswith("— |") or "| — |" in line
