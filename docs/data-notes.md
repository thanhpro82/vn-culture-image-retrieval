# Ghi chú đo đạc dữ liệu và index (ảnh chụp có ngày, đo lại trước khi trích)

## Index (dựng 2026-10-18, máy dev: RTX 3050 Laptop 4 GB, Ryzen 7 6800H, máy đồng thời chạy việc khác)

| Bước | Kết quả |
|---|---|
| Manifest | 92.339 URL ảnh được giữ từ 21 file Excel |
| Tải ảnh | 89.596 ảnh (97,03%), WebP cạnh dài <= 640, trung bình ~52 KB (mẫu 200); lưu `Data/images` và S3 (0 lỗi tải lên) |
| Nhúng ảnh SigLIP 2 B/16 (FP16, batch 64) | 89.596 ảnh trong khoảng 85 phút (15-27 ảnh/giây khi máy chạy đồng thời việc khác; mức cao nhất đo được ~27 ảnh/giây) |
| Nhúng text bge-m3 (FP16, max_length 256) | 89.596 tài liệu trong khoảng 25 phút (50-100 tài liệu/giây) |
| Nạp Qdrant | 89.596 điểm, 3 vector mỗi điểm (img 768, txt 1024, sparse), scalar INT8 |
| Kích thước | embeddings `.npy` + `.pkl` ~0,75 GB; Qdrant storage xem `Data/qdrant_storage` |

Ước lượng 95-130 ảnh/giây trong báo cáo research cho RTX 3050 không đạt: thực tế thấp hơn 4-5 lần (máy dùng chung với việc khác, nạp ảnh bằng một luồng CPU).

## Gần trùng (pHash 64-bit, 89.377 ảnh có pHash)

Hamming <=0: còn 84.635; <=4: 83.023; <=8: 81.359; <=10: 79.048. Chọn <=4 làm `dup_group` (gộp lúc trả kết quả, không loại khỏi index).
