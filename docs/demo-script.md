# Kịch bản demo (10-15 phút)

Các truy vấn và kết quả dưới đây là ảnh chụp chạy trên hệ thống thật ngày 2026-10-20 (rerank qua endpoint ngoài bật).
Kết quả có thể lệch đôi chút giữa các lần chạy; **hãy chạy thử từng truy vấn một lần trước buổi demo** (xem mục "Trước khi demo").

## Trước khi demo (15 phút trước)

1. Mở Docker Desktop, chờ engine sẵn sàng. Nếu báo lỗi `sailor-ingest.sock` hoặc `docker-secrets-engine`: tắt Docker Desktop, đổi tên hai thư mục
   `%LOCALAPPDATA%\Docker\run` và `%LOCALAPPDATA%\docker-secrets-engine` (thêm đuôi `.bak`), rồi mở lại.
2. `docker compose up -d` (biến `HF_CACHE` đã nằm trong `.env`, không cần đặt tay). Đợi `curl http://localhost:8000/healthz` trả `ok`.
3. **Làm nóng** (lần đầu một truy vấn có thể mất 3-17 giây vì Gemini và model chưa nóng, các lần sau nhanh và được cache): mở <http://localhost:8000> và
   chạy từng truy vấn trong danh sách bên dưới một lần.
4. Đóng bớt ứng dụng nặng (trình duyệt nhiều tab, ChatGPT, Notion); hệ thống chạy trên CPU nên tranh tài nguyên sẽ làm chậm.
5. Mở sẵn `docs/slides.html` ở một tab khác. Chỉ **một** người thao tác lúc demo: từ 3 người dùng đồng thời độ trễ vượt 3 giây.
6. Nếu endpoint rerank của mentor chết hoặc chậm: không sao, hệ thống tự quay về model local (chỉ chậm hơn khoảng 0,5 giây).

## Dòng thời gian

| Phút | Nội dung |
|---|---|
| 0-2 | Slide 1-2: bài toán, ràng buộc, 89.596 ảnh |
| 2-4 | Slide 3-5: kiến trúc, pipeline dữ liệu, cách xử lý ba loại truy vấn |
| 4-9 | **Demo trực tiếp** (các truy vấn bên dưới) |
| 9-11 | Slide 8-9: Recall@10, độ trễ, k6, rerank của mentor |
| 11-13 | Slide 10: hạn chế và chưa làm |
| 13-15 | Hỏi đáp |

## Demo trực tiếp (mở <http://localhost:8000>)

### A. Một ý: hệ thống tìm đúng thực thể

1. **`cầu Long Biên`**: top đầu là Cầu Long Biên (cầu sắt thời Pháp, trận địa). Nói: thực thể được nhận ra nhờ vector văn bản và sparse, không chỉ nhìn pixel.
2. **`nhà thờ Phát Diệm`**: toàn ảnh Nhà thờ chính toà / Nhà thờ Đá Phát Diệm.
3. **`đồ chơi dân gian làm bằng bột gạo`**: top đầu là **Tò he** (Xuân La, Bắc Ninh). Nói: câu hỏi không nhắc tên "tò he" mà vẫn ra đúng; Gemini nhận ra khái niệm. Giao diện không hiện
   phần này; nếu muốn cho xem, gọi `/search` bằng curl và chỉ vào trường `subs` (cụm tiếng Anh dùng cho SigLIP và danh sách thực thể).

### B. Nhiều ý: top 10 phủ cả hai ý

4. **`áo dài Huế và nhà thờ Phát Diệm`**: kết quả có cả ảnh áo dài/ca Huế và Nhà thờ Phát Diệm xen kẽ. Nói: Gemini tách hai ý, mỗi ý tìm riêng, rồi xen kẽ theo hạn ngạch để một ý không chiếm hết top 10.
5. **`chợ nổi và múa lân`**: có cả Chợ Cái Răng / chợ nổi Cà Mau và múa lân sư rồng.
6. (Dự phòng) **`rối nước và hát chèo`**: có cả hai ý.

### C. Khái niệm rộng và suy luận: thẳng thắn về điểm yếu

7. **`nghệ thuật sân khấu dân gian miền Bắc`**: ra rối nước, chèo, tuồng. Đây là ví dụ khái niệm rộng được mở rộng thành thực thể cụ thể.
8. **`con hổ`**: kết quả là Tranh Ngũ hổ, Tranh Bạch Hổ, Miếu Ông Hổ; giao diện hiện cảnh báo độ tin cậy thấp. Nói: dữ liệu là di sản văn hoá, không có ảnh con hổ thật,
   nên hệ thống trả các hình ảnh văn hoá liên quan thay vì giả vờ chắc chắn (cờ `low_confidence`).
9. **`động vật sống trong rừng`** hoặc **`ảnh con trai chủ tịch Phạm Nhật Vượng`**: ra kết quả không liên quan và có cờ cảnh báo. Nói rõ đây là điểm yếu:
   nhóm truy vấn suy luận chỉ đạt Recall@10 khoảng 0,50 (sau cải tiến ghép truy vấn ngày 21/10; trước đó 0,37) vì thực thể suy ra không có trong dữ liệu; hệ thống lọc thực thể bịa (grounding) và không bịa ảnh.

**Không nên demo:** `hồ bơi và công viên nước ở Phú Quốc` (ra nhà tù Phú Quốc, vì dữ liệu không có ảnh hồ bơi) và `bãi biển Nha Trang` (ra Tháp Bà Po Nagar). Nếu bị hỏi
đúng các câu này, trả lời: dữ liệu tập trung di sản văn hoá, nên các truy vấn mẫu về du lịch biển không có ảnh đúng trong index (đã ghi trong báo cáo).

## Câu hỏi có thể gặp

| Câu hỏi | Trả lời ngắn (kèm số) |
|---|---|
| Vì sao không chỉ dùng vector ảnh? | Vector ảnh không biết tên riêng. Recall@10: chỉ ảnh 0,32 / 0,35 / 0,11; hybrid 0,93 / 0,99 / 0,34; sau cải tiến ghép truy vấn: 0,99 / 1,00 / 0,50 (300 query, rerank bật) |
| Có dùng agent không? | Không. Bước tìm ảnh chỉ dùng embedding và rerank; Gemini chỉ hiểu truy vấn |
| Độ trễ bao nhiêu, đo thế nào? | 10/10 truy vấn dưới 3 giây (p95 1,78 s) với 1 người dùng, `docker compose`, đo bằng `loadtest/latency_probe.py`; k6 cho 1-3 người dùng |
| Vì sao 3 người dùng thì chậm? | CPU bão hoà (embed và rerank trên CPU): thông lượng khoảng 1 yêu cầu/giây. Hướng cải thiện: thêm nhân CPU hoặc rerank ngoài |
| Test case làm thế nào? | 300 query sinh từ tên thực thể và quan hệ, không sinh từ caption (tránh rò rỉ). Chưa có người duyệt: đã ghi là hạn chế |
| Đã dùng endpoint của mentor chưa? | Có, rerank `jina-reranker-v2` (tuỳ chọn, tự quay về local). Recall@10 ngang bộ local, nhanh hơn: 140-323 ms so với 640-940 ms. Embedding của mentor chưa dùng (2560 chiều không khớp 1024; hạn mức 50 yêu cầu/phút) |
| Gemini lỗi thì sao? | Timeout 1,6 s rồi dùng tách ý bằng quy tắc, không trả lỗi 500 |
| Dữ liệu lấy về từ đâu? | S3: ảnh WebP và snapshot Qdrant; README hướng dẫn khôi phục vào máy trống (đã thử) |
| Deploy tên miền? | Chưa: có workflow và Caddyfile nhưng chưa chạy vì chưa có VPS (điểm cộng, không bắt buộc) |
