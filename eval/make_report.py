"""Sinh bảng Markdown cho docs/eval-report.md từ các file reports/eval-*.json."""
import json
import sys
from pathlib import Path

ORDER = ["single", "multi_intent", "multi_hop"]
LABEL = {"single": "một ý", "multi_intent": "nhiều ý", "multi_hop": "suy luận"}


def render(runs: dict[str, dict]) -> str:
    lines = ["| Cấu hình | Loại | n | Recall@10 | MRR@10 | Phủ ý@10 | p50 (ms) | p95 (ms) | max (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rep in runs.items():
        for kind in ORDER:
            r = rep["by_type"].get(kind)
            if not r:
                continue
            cov = f"{r['intent_coverage@10']:.3f}" if "intent_coverage@10" in r else "—"
            lat = r["latency_ms"]
            lines.append(f"| {name} | {kind} | {r['n']} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} | {cov} | "
                         f"{lat['p50']} | {lat['p95']} | {lat['max']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    spec = {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[1:]}
    runs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in spec.items()}
    print(render(runs))
