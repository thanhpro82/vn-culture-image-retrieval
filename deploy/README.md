# Triển khai lên VPS (đã chạy thật trên AWS EC2 `c6i.2xlarge`, Ubuntu)

Cần: máy x86-64 từ 8 vCPU, 16 GB RAM, ổ 40 GB; Security Group mở 22 (chỉ IP của bạn), 80, 443; Elastic IP.
Chưa có tên miền thì dùng `<IP-dấu-gạch>.sslip.io` (ví dụ `13-251-152-106.sslip.io`), Caddy vẫn cấp HTTPS.

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER"            # đăng nhập lại để có hiệu lực
git clone https://github.com/thanhpro82/vn-culture-image-retrieval.git && cd vn-culture-image-retrieval
```

1. Tạo `.env` (quyền 600): `GEMINI_API_KEY`, `AWS_*`, `S3_BUCKET`, `S3_PREFIX`, `API_KEY` (chuỗi ngẫu nhiên mới), `DOMAIN=<tên miền>`;
   tuỳ chọn rerank ngoài: `EXT_BASE_URL`, `EXT_API_KEY`, `EXT_RERANK_ENABLED=true`.
2. Khôi phục index (Qdrant tự tải snapshot từ link ký sẵn, khoảng 30 giây, không cần Python):
   ```bash
   mkdir -p Data && docker compose up -d qdrant
   curl -fsSL "<ENTITIES_URL>" -o Data/entities.parquet
   curl -sS -X PUT "http://127.0.0.1:6333/collections/images/snapshots/recover?wait=true" \
     -H "Content-Type: application/json" -d '{"location": "<SNAPSHOT_URL>"}'
   ```
3. Thư mục cache model phải ghi được bởi người dùng trong container (uid 1000); nếu không `model-init` lỗi `Permission denied: '/models/hub'`:
   ```bash
   mkdir -p Data/hf-cache && sudo chown -R 1000:1000 Data/hf-cache
   ```
4. Chạy (Caddy nhận `DOMAIN` từ `.env`):
   ```bash
   docker compose --profile deploy up -d --build
   bash deploy/health.sh
   ```
5. Tải k6 trên chính VPS: `API_KEY=... PYTHON=python3 BASE_URL=http://172.17.0.1:8000 VUS=3 bash loadtest/run.sh`.

Tắt máy khi không dùng (EC2 Stop) để không tốn tiền; sau khi bật lại chạy lại bước 4. Không commit file khoá `.pem` (`.gitignore` đã chặn).
