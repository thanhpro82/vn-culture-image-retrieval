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

## Deploy tự động (GitHub Actions, đã chạy thử 2026-10-21)

`.github/workflows/deploy.yml` chạy sau khi CI trên `main` xanh (hoặc chạy tay): SSH vào VPS, `git pull --ff-only`, `bash deploy/deploy.sh` (build, khởi động lại, chờ `/healthz`).
Cần đặt trên repo: biến `DEPLOY_ENABLED=true`, biến `VPS_FINGERPRINT` (vân tay **ECDSA** của máy chủ: `ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub`),
secrets `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` (khoá SSH riêng cho CD, thêm khoá công khai vào `~/.ssh/authorized_keys` của VPS).
Cổng 22 của VPS phải cho GitHub truy cập (IP của GitHub không cố định); chỉ mở khi cần và chỉ cho đăng nhập bằng khoá.
Khi VPS tắt thì đặt `DEPLOY_ENABLED=false` để mỗi lần push vào `main` không báo lỗi.

### Bằng chứng chạy thử (2026-10-21) và trạng thái hiện tại

- Chạy tay từ nhánh `ci/deploy-hardening`: [run 39557409198](https://github.com/thanhpro82/vn-culture-image-retrieval/actions/runs/39557409198), thành công (lần đầu thất bại vì vân tay ED25519 không khớp, xem ghi chú ECDSA ở trên).
- Chạy **hoàn toàn tự động** sau khi merge PR #8 vào `main` (push → CI xanh → Deploy tự chạy): [run 39558277555](https://github.com/thanhpro82/vn-culture-image-retrieval/actions/runs/39558277555), thành công; VPS cập nhật đúng commit `07719b8` và `/healthz` trả 200.
- **Hiện CD đã tắt** (`DEPLOY_ENABLED=false`, các secrets `VPS_*` đã xoá, khoá CD đã gỡ khỏi VPS, cổng 22 trả về trạng thái hạn chế): máy chủ chỉ bật khi cần nên mỗi lần push vào `main` lúc VPS tắt sẽ báo lỗi nếu để bật.
  Muốn bật lại: làm theo các bước ở mục trên (tạo khoá SSH CD mới, đặt lại secrets và biến, mở cổng 22 cho GitHub).
