# ADR-001: Tỉ lệ link ảnh còn sống đủ để đạt 80.000 ảnh

- Trạng thái: Chấp nhận (đo 2026-10-18)
- Quyết định: dùng 92.339 URL ảnh đã thẩm định làm tập index, không bổ sung ảnh ứng viên chưa duyệt.

| Chiều | Ngưỡng đề xuất | Số đo (2026-10-18) | Căn cứ |
|---|---|---|---|
| Ảnh tải được trên mẫu | alive_rate x 92.339 >= 80.000, tức alive_rate >= 0,866 | 199/200 = 0,995 | `python -m retrieval.download probe Data/manifest.parquet --n 200 --seed 1`; lỗi duy nhất: http_403 |
| Dung lượng WebP (cạnh dài <= 640) | trung bình < 150 KB | 52 KB | cùng lần đo |
| Ước tính ảnh dùng được | >= 80.000 | ~91.900 | 0,995 x 92.339 (ước lượng từ mẫu 200; sai số lớn, đo lại khi tải đủ) |

Hạn chế: mẫu 200 là nhỏ, khoảng tin cậy rộng. Số thật sẽ có sau khi tải toàn bộ (`status.parquet`). Không giả `Referer`, không tắt kiểm tra SSL.

## Cập nhật sau khi tải đủ (2026-10-18)

| Chiều | Số đo |
|---|---|
| Tải được | 89.596 / 92.339 = 97,03% (mẫu 200 trước đó: 99,5%, hơi lạc quan) |
| Lỗi | timeout 1.433; HTTP 429: 578; HTTP 403: 216; lỗi chứng chỉ SSL: 194; HTTP 502: 64; HTTP 404: 61 |
| Gần trùng (pHash 64-bit, 89.377 ảnh có pHash) | Hamming <=0: còn 84.635; <=4: 83.023; <=8: 81.359; <=10: 79.048 |

Quyết định: giữ toàn bộ ảnh tải được trong index; gắn `dup_group` (Hamming <=4) và gộp ảnh trùng khi trả kết quả, để vẫn phục vụ > 80.000 ảnh.
