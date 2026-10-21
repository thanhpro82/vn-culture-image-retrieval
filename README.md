# Tìm ảnh văn hoá Việt Nam (Capstone, Phần B: Retrieve ảnh)

Web demo tìm ảnh theo văn bản tiếng Việt trên khoảng 89.600 ảnh đã thẩm định. Hỗ trợ ba loại truy vấn:
một ý ("cầu Long Biên"), nhiều ý ("áo dài Huế và nhà thờ Phát Diệm") và truy vấn cần suy luận
("nghệ thuật sân khấu dân gian miền Bắc"). Bước tìm ảnh chỉ dùng embedding và rerank, không dùng agent.
Dữ liệu là ảnh di sản văn hoá, nên các truy vấn về du lịch biển hay động vật hoang dã không có ảnh đúng (hệ thống gắn cảnh báo thay vì giả vờ chắc chắn).

## Nộp bài

| Mục | Ở đâu |
|---|---|
| Repo nộp | repo tổ chức `VSF-AR-ES-V-Lookup/Retrieval_Image_NguyenTuanThanh` (riêng tư), nhánh `main` |
| Web demo | <https://13-251-152-106.sslip.io> (VPS bật trong thời gian demo và chấm; nếu không vào được thì chạy bằng `docker compose` theo hướng dẫn bên dưới) |
| Tài liệu phương pháp | [`docs/slides.pdf`](docs/slides.pdf) (xem ngay trên GitHub), [`docs/slides.html`](docs/slides.html) (bản có chuyển động: tải về, mở bằng trình duyệt, dùng phím ← → Home End), [`docs/architecture.md`](docs/architecture.md) (sơ đồ Mermaid), [`docs/adr/`](docs/adr/) |
| Slide trên web demo | <https://13-251-152-106.sslip.io/slides.html> (có chuyển động, phím ← →), PDF: <https://13-251-152-106.sslip.io/slides.pdf> |
| Báo cáo chất lượng, bộ test và cách tạo | [`docs/eval-report.md`](docs/eval-report.md), [`eval/datasets/`](eval/datasets/) |
| Báo cáo hiệu năng, k6 | [`docs/perf-report.md`](docs/perf-report.md), [`docs/results/`](docs/results/), [`loadtest/`](loadtest/) |
| Kịch bản demo | [`docs/demo-script.md`](docs/demo-script.md) |

Bản công khai `thanhpro82/vn-culture-image-retrieval` có cùng nhánh `main`; đó là nơi chạy CI/CD tự động lên VPS (CD đang bật), còn bản nộp là repo tổ chức ở trên.

- Kịch bản demo 10-15 phút: [`docs/demo-script.md`](docs/demo-script.md)
- Slide mô tả phương pháp: [`docs/slides.pdf`](docs/slides.pdf) (xem trên GitHub) và [`docs/slides.html`](docs/slides.html) (có chuyển động, mở bằng trình duyệt, phím ← →)
- Thiết kế: [`docs/architecture.md`](docs/architecture.md), [`docs/superpowers/specs/`](docs/superpowers/specs/)
- Kết quả đánh giá: [`docs/eval-report.md`](docs/eval-report.md). Hiệu năng: [`docs/perf-report.md`](docs/perf-report.md)
- Quyết định kỹ thuật kèm số đo: [`docs/adr/`](docs/adr/)

## Yêu cầu

Docker Desktop (hoặc Docker Engine), khoảng 8 GB RAM trống cho container, 10 GB ổ đĩa. Để dựng lại index từ đầu cần
thêm Python 3.12, `uv` và GPU NVIDIA (RTX 3050 4 GB là đủ; CPU cũng chạy được nhưng rất chậm).

## Chạy demo nhanh (đã có index)

1. Tạo `.env` từ mẫu và điền khoá: `cp .env.example .env` rồi điền `GEMINI_API_KEY`, `API_KEY`, và nếu dùng S3 thì
   `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
2. Cần có trong `Data/`: `qdrant_storage/` (index), `entities.parquet`, và ảnh (qua S3 hoặc `images/`).
   Nếu chưa có, khôi phục từ snapshot (không cần GPU, khoảng 1,6 GB):

   ```bash
   docker compose up -d qdrant
   PYTHONPATH=src python -m retrieval.snapshot restore "<SNAPSHOT_URL>" "<ENTITIES_URL>"
   ```

   `<SNAPSHOT_URL>` và `<ENTITIES_URL>` là link S3 ký sẵn (hết hạn sau 7 ngày) do chủ repo cấp; chủ repo tạo lại bằng
   `PYTHONPATH=src python -m retrieval.snapshot publish` (tạo snapshot từ Qdrant đang chạy, tải lên `s3://<bucket>/index/`).
   Ảnh: dùng bucket S3 của chủ repo (`S3_*` trong `.env`), hoặc tự tải lại từ URL gốc bằng `python -m retrieval.download all`
   (khoảng một giờ, một số link nguồn có thể đã chết), sau đó để `S3_BUCKET` trống thì API phục vụ ảnh từ `Data/images`.
3. Chạy:

```bash
docker compose up -d --build
```

Lần đầu `model-init` tải các model (khoảng 4 GB) vào `Data/hf-cache`; các lần sau dùng lại. Mở <http://localhost:8000>.

Kiểm tra sức khoẻ: `curl http://localhost:8000/healthz`. Thử API (cần `X-API-Key`):

```bash
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"query": "áo bà ba", "top_k": 10}'
```

Các endpoint: `POST /search`, `POST /embed` (`space`: `image` hoặc `text`), `POST /rerank`, `GET /healthz`.
Giao diện web gọi `POST /demo/search` (không cần khoá).

## Dữ liệu và index (dựng từ đầu)

Dữ liệu đầu vào là 21 file Excel của team 3 đặt ở `Data/incoming/final` và `Data/incoming/recheck`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,ml]"        # cài thêm torch bản CUDA nếu có GPU
PYTHONPATH=src python -m retrieval.manifest Data/incoming Data/manifest.parquet   # 92.339 URL được giữ
PYTHONPATH=src python -m retrieval.download all Data/manifest.parquet Data/images --concurrency 128
docker compose up -d qdrant      # Qdrant cho lúc dựng index (dữ liệu ở Data/qdrant_storage)
PYTHONPATH=src python -m retrieval.index build                         # embed ảnh + text, nạp Qdrant
PYTHONPATH=src python -m retrieval.index entities                      # danh mục thực thể cho cổng kiểm tra
```

Quy ước nhãn: ô `Giữ/Bỏ` trống là giữ, `x` là bỏ; nhãn lấy từ file recheck mới nhất, nếu chưa recheck thì từ file final.

## Kiểm thử

```bash
make check          # ruff + pytest (bỏ test cần model thật)
pytest -m slow      # test cần tải model (vài GB)
```

Đánh giá chất lượng (cần API đang chạy):

```bash
PYTHONPATH=src python -m eval.build_testset 100 0          # 300 query: 100 mỗi loại, seed 0
python -m eval.evaluate eval/datasets/testset-v1-seed0.jsonl --key "$API_KEY" --out reports/eval-hybrid.json
python -m eval.evaluate eval/datasets/testset-v1-seed0.jsonl --key "$API_KEY" --mode image --no-rerank
```

Load test k6 (chạy trong Docker, sinh báo cáo tự động):

```bash
API_KEY=... VUS=5 DURATION=60s bash loadtest/run.sh        # xuất reports/loadtest-report.md và .html
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`): lint, test nhanh, kiểm tra compose và build image. Workflow `deploy.yml` chỉ chạy
khi đặt biến repo `DEPLOY_ENABLED=true` và các secrets `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`; nó cập nhật mã trên VPS và gọi
`deploy/deploy.sh`. Tên miền và HTTPS do Caddy (`docker compose --profile deploy up -d`, đặt biến `DOMAIN`).

## Cấu trúc

```
src/retrieval/   config, manifest, download, storage, embed, index, qparse, grounding, search, rerank, api, main
web/             giao diện một trang
eval/            bộ sinh test và đánh giá (Recall@10, MRR@10, độ phủ ý)
loadtest/        k6 + báo cáo
tests/           pytest
docs/            spec, kế hoạch, ADR, báo cáo
```
