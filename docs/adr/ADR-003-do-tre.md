# ADR-003: Độ trễ end-to-end (đang điền dần)

## Gemini 3.5 Flash Lite xử lý query (đo 2026-10-18, từ máy dev, mạng cá nhân)

| Chiều | Ngưỡng đề xuất | Số đo | Căn cứ |
|---|---|---|---|
| Độ trễ một lần gọi | p95 < 1.600 ms | p50 = 991 ms, p95 = 1.214 ms, max = 1.272 ms, 0/15 timeout | 5 query x 3 lần, `thinkingLevel=minimal`, schema rút gọn |
| So với research | ước lượng 450 ms | thực đo ~900-1.150 ms (mức sàn mạng + token đầu) | rút gọn prompt hoặc tắt schema không giảm đáng kể (p50 ~907 ms) |
| `thinkingBudget: 0` | báo cáo research dùng | HTTP 400 với model này | chuyển sang `thinkingLevel: minimal` |

Quyết định: timeout Gemini = 1.600 ms (spec ghi 1.200 ms, mâu thuẫn số đo; 1.200 ms sẽ đẩy nhiều query sang fallback). Phần còn lại (embed + tìm + rerank + trả JSON) phải nằm trong ~1,4 giây.

## Độ trễ end-to-end /search (đo 2026-10-19, API thật, CPU dev, 89.596 điểm, tuần tự một yêu cầu/lần)

Bộ query: 20 truy vấn tiếng Việt mới (Gemini chưa cache), trộn một ý và nhiều ý; client gọi `http://127.0.0.1:8000` (không dùng `localhost`: trên Windows thêm ~2 s do thử IPv6).

| Chỉ số | Giá trị |
|---|---|
| p50 / p95 / max | 2.340 / 2.619 / 2.787 ms |
| Dưới 3 giây | 20/20 |
| Thành phần điển hình | parse (Gemini) 0,9-1,5 s; truy xuất (embed + Qdrant) 0,3-0,5 s (0,7-1,2 s với ca lạnh); rerank 0,7-1,0 s |
| Rerank bị bỏ qua do cơ chế ngân sách | 5/20 (khi parse + truy xuất đã dùng quá nhiều thời gian) |

Quá trình đo, theo thứ tự: bản đầu (fp32, 24 cặp, không guard) p50 client ~2,7 s với một số ca > 3 s; sau INT8 + 16 cặp + làm nóng khi khởi động: p50 2,6 s, 19/20 < 3 s (max 3,18 s, ca nhiều ý chấm tách từng ý); sau gộp chấm một lần và trần 2,7 s: bảng trên.

Hạn chế: một yêu cầu mỗi lần. Khi chạy đồng thời CPU bị chia nên độ trễ tăng (xem `docs/perf-report.md`).
