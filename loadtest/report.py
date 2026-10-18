"""Sinh báo cáo Markdown từ loadtest-summary.json của k6."""
import json
import sys
from datetime import date


def _v(m: dict, key: str, default=0.0) -> float:
    return float(m.get("values", {}).get(key, default))


def render(summary: dict) -> str:
    m = summary["metrics"]
    s = m["search_ms"]
    reqs = m["http_reqs"]
    lines = [
        f"# Báo cáo tải /search (k6, đo ngày {date.today().isoformat()})",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---|",
        f"| Tổng số request (cả /embed, /rerank) | {int(_v(reqs, 'count'))} |",
        f"| Thông lượng | {_v(reqs, 'rate'):.1f} req/s |",
        f"| Độ trễ /search p50 | {_v(s, 'p(50)', _v(s, 'med')):.0f} ms |",
        f"| Độ trễ /search p95 | {_v(s, 'p(95)'):.0f} ms |",
        f"| Độ trễ /search p99 | {_v(s, 'p(99)'):.0f} ms |",
        f"| Độ trễ /search lớn nhất | {_v(s, 'max'):.0f} ms |",
        f"| Tỉ lệ /search dưới 3 giây | {_v(m['search_under_3s'], 'rate') * 100:.1f}% |",
        f"| Tỉ lệ request lỗi | {_v(m['http_req_failed'], 'rate') * 100:.2f}% |",
        "",
        "Số liệu là ảnh chụp của một lần chạy; đo lại trước khi trích.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        md = render(json.load(f))
    sys.stdout.buffer.write(md.encode("utf-8"))  # không phụ thuộc mã hoá console của Windows
