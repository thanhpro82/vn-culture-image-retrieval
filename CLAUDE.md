# CLAUDE.md

Hệ thống tìm ảnh theo văn bản tiếng Việt (Phần B của capstone Image Processing & Retrieval). Hạn nộp 2026-10-22.

- Spec: `docs/superpowers/specs/2026-10-18-image-retrieval-design.md`. Kế hoạch: `docs/superpowers/plans/`.
- Mã nguồn ở `src/retrieval/`, test ở `tests/`, đánh giá ở `eval/`, load test ở `loadtest/`.
- Dữ liệu của team nằm ở `Data/` (không commit). Bí mật nằm ở `.env` (không commit, không in ra).
- Lệnh: `make test` (test nhanh), `make lint`, `make check`. Python 3.12 trong `.venv`.
- Marker pytest: `slow` và `integration` cho test cần model thật, Qdrant thật hoặc mạng; CI bỏ qua chúng.

## Quy tắc

- Mọi con số trong báo cáo là ảnh chụp có ngày. Đo lại trước khi trích, ghi rõ lần đo.
- Số liệu ước lượng phải ghi là ước lượng. Số liệu từ nguồn bên ngoài phải ghi nguồn.
- Nhánh `feature/<giai-đoạn>` rồi PR vào `main`. Commit dạng `type(scope): mô tả`.
- Bước tìm ảnh chỉ dùng embedding và rerank, không dùng agent.
