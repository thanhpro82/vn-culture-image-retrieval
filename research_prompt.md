# Prompt research: hệ thống Image Retrieval (Phần B)

Copy toàn bộ phần dưới vào Claude (Deep Research) hoặc ChatGPT Deep Research.

---

Bạn là kỹ sư ML/search giàu kinh nghiệm. Hãy research và đề xuất phương pháp cụ thể, có dẫn nguồn (paper, benchmark, repo, tài liệu chính thức, ưu tiên 2024-2026), cho bài toán sau. Ưu tiên số liệu đo được hơn mô tả chung chung. Nếu thiếu số liệu, nói rõ là ước lượng.

## 1. Bài toán
Xây hệ thống truy xuất ảnh theo văn bản (text-to-image retrieval) cho tiếng Việt, phục vụ ít nhất 80.000 ảnh, chạy được bằng docker compose.

- Chất lượng: đo bằng recall@10. Mỗi query trả 10 ảnh, chỉ cần 1 ảnh trùng ground truth là tính đúng.
- Latency: dưới 3 giây cho mỗi query, tính trên toàn bộ 80k ảnh, gồm cả thời gian gọi LLM.
- Bắt buộc: bước tìm ảnh chỉ dùng embedding và/hoặc rerank, KHÔNG dùng agent để tìm ảnh. LLM/VLM chỉ được dùng để xử lý query (và có thể để tạo caption lúc index).
- LLM cho phép: Gemini 3.5 Flash Lite qua API.
- Phải có endpoint /embed và /rerank, có API key cho auto test.
- Phải có load test tái lập được (k6 hoặc Locust) kèm báo cáo sinh tự động.

## 2. Ba loại query
1. Single intent: "bãi biển Nha Trang", "con hổ".
2. Multi-intent: "hồ bơi và công viên nước ở Phú Quốc", "con hổ và con mèo". Kết quả top 10 phải có ảnh của từng ý.
3. Multi-hop / khái niệm: "ảnh con trai chủ tịch Phạm Nhật Vượng", "động vật sống trong rừng". Cần suy luận qua nhiều bước hoặc mở rộng thành thực thể cụ thể.

## 3. Dữ liệu
- Chủ đề: văn hoá, di sản, lễ hội, trang phục, địa danh Việt Nam (ví dụ "Áo bà ba cách tân", "Lễ hội Leo cột mỡ", "Chiến khu D").
- Dạng: file Excel, KHÔNG phải file ảnh. Mỗi ảnh là một URL cần tải về. Dữ liệu chia theo thực thể (entity), mỗi thực thể có nhiều ảnh đã được người verify.
- Metadata mỗi ảnh: tên thực thể, mô tả ảnh do Gemini Vision sinh ra (tiếng Việt hoặc Anh), tiêu đề trang nguồn, tên miền, độ phân giải, pHash, dHash, cờ giữ/bỏ.
- Metadata mỗi thực thể: nhánh phân loại (ví dụ "Trang phục & diện mạo"), loại thực thể, địa bàn, thể loại văn hoá, mô tả.
- Ảnh được đẩy lên S3 và phải tải về được.
- Máy dev: RTX 3050 4GB, RAM 15GB, Ryzen 7 6800H. Deploy mục tiêu: docker compose trên CPU (VPS thường), GPU chỉ dùng lúc index.

## 4. Câu hỏi cần research (trả lời từng mục)

### 4.1 Embedding model
So sánh các model text-image đa ngôn ngữ chạy được trên GPU 4GB và CPU: SigLIP 2, Jina CLIP v2, Multilingual CLIP, các model Việt hoá (nếu có), và các lựa chọn mới hơn. Cho mỗi model: kích thước, chất lượng với tiếng Việt, tốc độ index (ảnh/giây) trên GPU 4GB, tốc độ embed text trên CPU, kết quả benchmark liên quan.

### 4.2 Hybrid retrieval
Kết hợp embedding ảnh với index text trên caption và metadata (BM25 + text embedding như bge-m3). Có đáng làm không so với chỉ embedding ảnh? Có số liệu recall nào so sánh không? Cách fusion nào tốt (RRF, trọng số, learned)?

### 4.3 Rerank
Rerank bằng gì trong 3 giây, trên CPU? Cross-encoder (bge-reranker, jina-reranker), late-interaction (ColPali/ColQwen có hợp không?), VLM nhỏ. Chi phí latency cho top 30-50.

### 4.4 Xử lý query bằng LLM
Prompt thiết kế thế nào để một lần gọi Gemini Flash Lite làm đủ: tách multi-intent, mở rộng multi-hop thành thực thể cụ thể, dịch Việt → Anh nếu cần, trả JSON cố định. Latency thực tế của Flash Lite, cách giảm (streaming, cache, schema ngắn). Cách xử lý khi LLM lỗi hoặc chậm (fallback).

### 4.5 Multi-intent
Cách gộp kết quả để top 10 phủ đủ các thực thể: interleave, RRF theo sub-query, quota theo ý, MMR/diversification. Cách đo "phủ đủ ý" ngoài recall@10.

### 4.6 Multi-hop
Với dữ liệu thiên về thực thể văn hoá Việt Nam, multi-hop nên làm bằng: (a) LLM mở rộng query thành danh sách thực thể rồi search từng cái, (b) knowledge graph dựng từ metadata thực thể, hay (c) cách khác? So sánh độ chính xác, latency, rủi ro hallucination. Cách kiểm tra LLM không bịa thực thể không có trong dữ liệu.

### 4.7 Vector index
FAISS (HNSW, IVF-PQ), hnswlib, Qdrant, pgvector, Milvus Lite cho 80k đến 1M vector. Chọn gì để image nhỏ, khởi động nhanh, latency dưới 50ms trên CPU. Cần quantization không ở quy mô này?

### 4.8 Pipeline dữ liệu
Tải 80k ảnh từ URL (tỉ lệ lỗi, retry, rate limit), dedupe bằng pHash, lọc chất lượng, lưu S3. Cách tổ chức schema (thực thể, ảnh, caption, embedding). Có nên tạo caption mới bằng VLM không khi đã có mô tả Gemini Vision sẵn? Chi phí Flash Lite cho 80k ảnh.

### 4.9 Đánh giá
Cách làm test set cho 3 loại query khi không có ground truth sẵn: sinh query từ tên thực thể, từ caption, từ quan hệ giữa thực thể; kiểm tra bằng người; tránh leakage (query trùng caption dùng để index). Cỡ test set hợp lý, cách báo cáo recall@10, MRR, latency p50/p95/p99.

### 4.10 Vận hành
Docker multi-stage, image nhỏ, healthcheck, nạp index nhanh lúc khởi động, FastAPI (hay Go) cho endpoint embed/rerank, load test bằng k6 hoặc Locust và cách sinh báo cáo tự động, CI/CD GitHub Actions (build, test, deploy), deploy có tên miền.

## 5. Định dạng đầu ra
1. Bảng so sánh cho mỗi mục 4.1, 4.2, 4.3, 4.7 (ưu, nhược, số liệu, nguồn).
2. Đề xuất cuối: kiến trúc khuyến nghị kèm ngân sách latency theo từng bước (LLM, embed, search, rerank), có tổng dưới 3 giây.
3. Danh sách rủi ro và cách giảm.
4. Kế hoạch 4 ngày (hạn nộp 22/10/2026) ưu tiên theo giá trị: cái gì làm trước để có demo sớm, cái gì làm sau nếu còn thời gian.
5. Liệt kê rõ chỗ nào là số liệu từ nguồn, chỗ nào là suy đoán của bạn.
