#!/usr/bin/env bash
# Chờ API khỏe (tối đa ~3 phút vì lần đầu phải nạp model).
set -euo pipefail
PORT=${PORT:-8000}
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then echo "healthy"; exit 0; fi
  sleep 3
done
echo "API không khỏe sau 3 phút" >&2
exit 1
