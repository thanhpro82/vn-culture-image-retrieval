# Báo cáo hiệu năng

Đo ngày 2026-10-18/19. Mọi số liệu là ảnh chụp của một lần chạy trên một máy; đo lại trước khi trích.

## Môi trường đo

| Mục | Giá trị |
|---|---|
| Máy | Laptop Ryzen 7 6800H (8 nhân, 16 luồng), RAM 16 GB (15,2 GB dùng được), Windows 11 |
| Cách chạy | `docker compose` trên Docker Desktop (WSL2, 16 CPU và ~7,9 GB RAM cấp cho Docker): container `api` (FastAPI, model trên CPU) và `qdrant` |
| Dữ liệu | 89.596 ảnh, collection `images` với 3 vector mỗi ảnh |
| Gemini | `gemini-3.5-flash-lite` qua mạng cá nhân; một lần gọi mỗi truy vấn, timeout 1.600 ms |
| Ảnh | S3 `ap-southeast-1`, link ký sẵn; không đi qua API |

## Độ trễ một yêu cầu (yêu cầu "dưới 3 giây")

Cách đo: `loadtest/latency_probe.py` gọi `/search` lần lượt, một yêu cầu mỗi lần, 20 truy vấn mới (Gemini chưa cache), client dùng `127.0.0.1`.
Không dùng `localhost` vì trên Windows nó thử IPv6 trước và cộng thêm ~2 giây mỗi yêu cầu (đã xác nhận bằng đo).

| Cấu hình | p50 | p95 | max | Dưới 3 s |
|---|---|---|---|---|
| Chạy trực tiếp (không Docker), 20 truy vấn | 2.340 ms | 2.619 ms | 2.787 ms | 20/20 |
| `docker compose` (cấu hình giao nộp), 20 truy vấn | 2.318 ms | 2.718 ms | 2.724 ms | 20/20 |
| `docker compose` sau khi khôi phục index từ snapshot S3 (mã cuối cùng, gồm giới hạn tần suất), 10 truy vấn mới | (xem `docs/results/probe-compose-final.txt`) | | | 10/10 |
| Bộ 300 query đánh giá chất lượng, chạy trực tiếp, tuần tự | 2.298-2.391 ms | 2.663-2.701 ms | 2.712-2.837 ms | (theo từng loại, xem `docs/eval-report.md`) |

Thành phần điển hình của một truy vấn (container): Gemini 0,8-1,4 s; embed text trên CPU 0,5-1,0 s; Qdrant 0,03-0,07 s; rerank 0,8-1,1 s khi chạy.
Rerank chỉ chạy khi còn dư ngân sách thời gian (xem ADR-002), ngược lại trả thứ tự hybrid. Trong phép đo `docker compose` ở trên, rerank bị bỏ qua ở **15/20** truy vấn (`docs/results/probe-compose.txt`), nghĩa là chất lượng của cấu hình giao nộp trên máy này gần với dòng "Hybrid, không rerank" trong `docs/eval-report.md` hơn là dòng "Cấu hình cuối" (đo chạy trực tiếp, nơi embed nhanh hơn và rerank chạy nhiều hơn; tỉ lệ rerank thực tế của lần đó không được ghi lại).

## Thông lượng và tải đồng thời (k6, tự động, chạy lại được)

Chạy bằng `loadtest/run.sh` (k6 trong Docker, 60 giây mỗi mức, truy vấn xoay vòng trong bộ 300 query, cứ 10 vòng thêm một lần gọi `/embed` và `/rerank`).
Kết quả thô nằm ở `docs/results/k6-vus*-summary.json`, báo cáo HTML ở `docs/results/k6-vus*.html`.

| Người dùng đồng thời | Thông lượng (tất cả request) | /search p50 | /search p95 | /search p99 | max | /search dưới 3 s | Lỗi |
|---|---|---|---|---|---|---|---|
| 1 | 0,6 req/s | 1.958 ms | 2.668 ms | 2.795 ms | 2.834 ms | 100,0% | 0% |
| 2 | 0,9 req/s | 2.337 ms | 2.926 ms | 3.903 ms | 4.641 ms | 95,7% | 0% |
| 3 | 1,0 req/s | 3.374 ms | 4.136 ms | 4.844 ms | 5.466 ms | 21,6% | 0% |

Cách đọc:

- **Dung lượng phục vụ của cấu hình này là khoảng 1 yêu cầu/giây.** Hệ thống bị giới hạn bởi CPU (embed và rerank), không phải bởi Qdrant hay mạng.
  Với một đến hai người dùng đồng thời vẫn đạt mục tiêu dưới 3 giây ở p95 hoặc sát; từ ba người trở lên, độ trễ tăng gần tuyến tính theo số người dùng.
- Không có lỗi HTTP ở cả ba mức.
- Số liệu k6 ở mức 1 người cao hơn chút so với đo tuần tự vì có cả các lần gọi `/embed` và `/rerank` xen kẽ dùng chung CPU.

## Nút cổ chai và cách đã xử lý

| Nút cổ chai | Số đo | Xử lý |
|---|---|---|
| Rerank cross-encoder trên CPU | fp32 24 cặp: 1.770 ms | INT8 động, 16 cặp, nhiều luồng: ~600-900 ms (ADR-002); bảo vệ ngân sách và hạn chót cứng |
| Gemini | ~1,0 s, không rút ngắn được bằng prompt ngắn hơn | Timeout 1.600 ms và fallback quy tắc (ADR-003) |
| Embed text trong container | 1,0-1,8 s với 16 luồng; 0,5-1,0 s với 8 luồng | `TORCH_THREADS=8` mặc định trong compose |
| Nhiều yêu cầu tranh CPU | Một lần đo sơ bộ với 2 người dùng (không lưu lại) cho kết quả kém hơn rõ rệt | Chạy tuần tự phần nặng CPU: 95,7% dưới 3 s ở 2 người dùng (`docs/results/k6-vus2.md`) |
| Nạp model lười lần đầu | 39 s ở yêu cầu đầu | Làm nóng khi khởi động (API sẵn sàng sau ~50 s) |

## Dựng index (ngoại tuyến)

Xem `docs/data-notes.md`: nhúng 89.596 ảnh bằng SigLIP 2 mất khoảng 85 phút và nhúng text bằng bge-m3 khoảng 25 phút trên RTX 3050 4 GB (máy dùng chung với việc khác);
ước lượng 95-130 ảnh/giây trong tài liệu tham khảo không đạt, thực tế 15-27 ảnh/giây.

## Đóng gói Docker

| Chỉ số | Giá trị |
|---|---|
| Kích thước image `api` (chưa gồm model) | 1,52 GB khi build sạch trên GitHub Actions (Linux); 2,07 GB bản build cục bộ trên Docker Desktop. torch CPU, transformers và FlagEmbedding là phần lớn |
| Thời gian build sạch (không cache) | 4 phút 4 giây trên GitHub Actions (`docker` job, run 39134558606) |
| Khởi động đến khi healthy | ~50 giây (nạp ba model và làm nóng); Qdrant healthy sau ~5 giây |
| Model | tải vào volume `Data/hf-cache` bằng dịch vụ `model-init`, không đóng gói trong image |
| Multi-stage, người dùng không phải root, `HEALTHCHECK` | có |

## Hạn chế

- Máy đo là một laptop cũng đang chạy các ứng dụng khác; số liệu thay đổi theo tải nền.
- Chưa đo trên VPS thật; chưa đo với GPU cho phía phục vụ.
- Mức tải 60 giây và 300 truy vấn xoay vòng; chưa có thử nghiệm kéo dài.
- Image 2 GB còn lớn: FlagEmbedding kéo theo nhiều thư viện (scikit-learn, datasets, peft...) mà chỉ cần một phần nhỏ; có thể thay bằng cài đặt bge-m3 tối giản nếu cần giảm.
- Số đo "docker compose" ở trên chạy trên image cục bộ đã chồng mã nguồn mới lên bản build đầy đủ trước đó (build lại từ đầu trên mạng của máy dev quá chậm); Dockerfile build sạch đã được kiểm chứng riêng trên GitHub Actions (xanh).
