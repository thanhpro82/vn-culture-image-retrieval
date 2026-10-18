#!/usr/bin/env bash
# Chạy trên VPS: cập nhật image và khởi động lại. Cần .env và dữ liệu (Qdrant storage, ảnh) đã có sẵn.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose --profile deploy up -d --build
bash deploy/health.sh
