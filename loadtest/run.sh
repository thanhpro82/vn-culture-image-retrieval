#!/usr/bin/env bash
# Chạy tải bằng k6 trong Docker, xuất báo cáo vào reports/. Yêu cầu: API đang chạy, biến API_KEY.
# Ví dụ: API_KEY=... VUS=2 DURATION=60s bash loadtest/run.sh   (đặt PYTHON=... nếu "python" không phải Python 3)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${API_KEY:?cần đặt API_KEY}"
PY=${PYTHON:-python}
TESTSET=${TESTSET:-eval/datasets/testset-v1-seed0.jsonl}
"$PY" loadtest/make_queries.py "$TESTSET"
mkdir -p reports
# Git Bash trên Windows làm hỏng đường dẫn có dấu cách và đường dẫn tuyệt đối trong container:
# dùng đường dẫn kiểu Windows cho thư mục gắn vào và tắt chuyển đổi đường dẫn.
ROOT=$(pwd -W 2>/dev/null || pwd)
export MSYS_NO_PATHCONV=1
set +e  # k6 trả mã lỗi khi vượt ngưỡng; vẫn phải sinh báo cáo
docker run --rm \
  -v "$ROOT/loadtest:/scripts" -v "$ROOT/reports:/reports" -w /scripts \
  -e API_KEY -e BASE_URL="${BASE_URL:-http://host.docker.internal:8000}" \
  -e VUS="${VUS:-2}" -e DURATION="${DURATION:-60s}" \
  -e K6_WEB_DASHBOARD=true -e K6_WEB_DASHBOARD_EXPORT=/reports/loadtest-report.html \
  grafana/k6@sha256:e66db15b860113878fa74670e31f5e274830b7b6e42c8bff28b2f2d86a257603 run script.js  # k6 v2.3.0 (ghim theo digest)
K6_RC=$?
set -e
"$PY" loadtest/report.py reports/loadtest-summary.json > reports/loadtest-report.md
echo "báo cáo: reports/loadtest-report.md, reports/loadtest-report.html (k6 exit code: $K6_RC)"
exit "$K6_RC"
