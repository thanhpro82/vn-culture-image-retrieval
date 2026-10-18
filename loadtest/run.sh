#!/usr/bin/env bash
# Chạy tải bằng k6 trong Docker, xuất báo cáo vào reports/. Yêu cầu: API đang chạy, biến API_KEY.
# Ví dụ: API_KEY=... VUS=5 DURATION=60s bash loadtest/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${API_KEY:?cần đặt API_KEY}"
TESTSET=${TESTSET:-eval/datasets/testset-v1-seed0.jsonl}
python loadtest/make_queries.py "$TESTSET"
mkdir -p reports
docker run --rm \
  -v "$PWD/loadtest:/scripts" -v "$PWD/reports:/reports" -w /scripts \
  -e API_KEY -e BASE_URL="${BASE_URL:-http://host.docker.internal:8000}" \
  -e VUS="${VUS:-5}" -e DURATION="${DURATION:-60s}" \
  -e K6_WEB_DASHBOARD=true -e K6_WEB_DASHBOARD_EXPORT=/reports/loadtest-report.html \
  grafana/k6:latest run script.js
python loadtest/report.py reports/loadtest-summary.json > reports/loadtest-report.md
echo "báo cáo: reports/loadtest-report.md, reports/loadtest-report.html"
