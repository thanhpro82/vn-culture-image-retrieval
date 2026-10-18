# ADR-003: Độ trễ end-to-end (đang điền dần)

## Gemini 3.5 Flash Lite xử lý query (đo 2026-10-18, từ máy dev, mạng cá nhân)

| Chiều | Ngưỡng đề xuất | Số đo | Căn cứ |
|---|---|---|---|
| Độ trễ một lần gọi | p95 < 1.600 ms | p50 = 991 ms, p95 = 1.214 ms, max = 1.272 ms, 0/15 timeout | 5 query x 3 lần, `thinkingLevel=minimal`, schema rút gọn |
| So với research | ước lượng 450 ms | thực đo ~900-1.150 ms (mức sàn mạng + token đầu) | rút gọn prompt hoặc tắt schema không giảm đáng kể (p50 ~907 ms) |
| `thinkingBudget: 0` | báo cáo research dùng | HTTP 400 với model này | chuyển sang `thinkingLevel: minimal` |

Quyết định: timeout Gemini = 1.600 ms (spec ghi 1.200 ms, mâu thuẫn số đo; 1.200 ms sẽ đẩy nhiều query sang fallback). Phần còn lại (embed + tìm + rerank + trả JSON) phải nằm trong ~1,4 giây.
