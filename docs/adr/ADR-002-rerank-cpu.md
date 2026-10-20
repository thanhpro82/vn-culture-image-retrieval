# ADR-002: Rerank bằng cross-encoder trên CPU

- Trạng thái: Chấp nhận có điều kiện (đo 2026-10-18; điều kiện: kết quả đánh giá chất lượng, xem `docs/eval-report.md`)
- Máy đo: Ryzen 7 6800H (8 nhân, 16 luồng), PyTorch 2.14 CPU, `BAAI/bge-reranker-base`, 128 token, các tài liệu dài ~60-110 token.

| Cấu hình (chấm 24 cặp, p50 của 5 lần) | Thời gian |
|---|---|
| fp32, 8 luồng (mặc định) | 1.770 ms |
| fp32, 12 luồng | 1.348 ms |
| INT8 động (`quantize_dynamic`), 8 luồng | 1.143 ms |
| INT8 động, 12 luồng | 853 ms |
| INT8 động, 16 luồng | 811 ms |
| INT8 động, 16 luồng, 16 cặp | 604 ms |
| INT8 động, 16 luồng, 12 cặp | 481 ms |
| INT8 động, 12 luồng, 24 cặp, max_len 96 | 885 ms (rút token không giúp nhiều) |

Số đo trong API thật (query mới, Gemini chưa cache): rerank 0,7-1,0 s với 16 cặp.

## Quyết định

- Dùng INT8 động, số luồng = số CPU logic, chấm tối đa **16** ứng viên (thay cho 24 trong kế hoạch ban đầu).
- Báo cáo research ước lượng 180-320 ms cho 24 cặp (ONNX INT8); thực đo trên máy này cao hơn 3 lần với PyTorch INT8.
- Ngân sách tổng < 3 giây buộc phải tiết kiệm: Gemini ~1,0-1,5 s chiếm gần nửa.

## Chưa làm

- Xuất ONNX Runtime INT8 (có thể nhanh hơn thêm 1,5-2 lần), reranker nhỏ hơn đa ngôn ngữ, và chỉ rerank khi truy vấn mơ hồ. Chỉ làm khi đánh giá chất lượng cho thấy rerank đáng giá hoặc cần thêm dư địa độ trễ.
