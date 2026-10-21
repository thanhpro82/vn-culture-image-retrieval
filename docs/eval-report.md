# Báo cáo đánh giá chất lượng truy xuất

Đo ngày 2026-10-18/19 trên index 89.596 ảnh. Mọi số liệu là ảnh chụp của một lần chạy (seed 0); đo lại trước khi trích.

## Cách làm test case

Bộ test sinh tự động bằng `eval/build_testset.py` từ manifest (seed 0), gồm **300 query: 100 mỗi loại**.

| Loại | Cách sinh query | Đáp án (ảnh đúng) |
|---|---|---|
| Một ý | Tên thực thể đã bỏ phần trong ngoặc, ví dụ "Chè bột lọc" (thực thể có ít nhất 3 ảnh) | Mọi ảnh của thực thể (gộp các thực thể trùng tên) |
| Nhiều ý | Hai thực thể khác nhau cùng địa bàn ghép bằng "và", ví dụ "A và B" | Hợp ảnh của hai thực thể; mỗi ý có tập đáp án riêng để đo độ phủ ý |
| Suy luận / khái niệm | "<thể loại văn hoá> <địa bàn>" (nhóm có ít nhất 5 thực thể), ví dụ "trang phục nam bộ" | Mọi ảnh thuộc nhóm thể loại và địa bàn đó |

Nguyên tắc chống rò rỉ: không query nào trùng nguyên văn một caption đã index; test kiểm tra điều này (`tests/test_evaluate.py`). Cách
chấm: một query **trúng** nếu ít nhất 1 trong 10 ảnh trả về thuộc đáp án (đúng định nghĩa Recall@10 của đề). MRR@10 là nghịch đảo hạng
của ảnh đúng đầu tiên. Độ phủ ý@10 (chỉ nhiều ý) là tỉ lệ ý có ít nhất một ảnh đúng trong top 10.

Tái lập:

```bash
PYTHONPATH=src python -m eval.build_testset 100 0
python -m eval.evaluate eval/datasets/testset-v1-seed0.jsonl --key "$API_KEY" --out reports/eval-final.json
python -m eval.evaluate eval/datasets/testset-v1-seed0.jsonl --key "$API_KEY" --mode image --no-rerank --out reports/eval-image-only.json
python -m eval.make_report "Tên=reports/eval-final.json" ...
```

## Kết quả

| Cấu hình | Loại | n | Recall@10 | MRR@10 | Phủ ý@10 | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|---|---|---|---|
| Chỉ vector ảnh | single | 100 | 0.320 | 0.183 | — | 387 | 493 | 806 |
| Chỉ vector ảnh | multi_intent | 100 | 0.350 | 0.207 | 0.230 | 526 | 614 | 667 |
| Chỉ vector ảnh | multi_hop | 100 | 0.110 | 0.034 | — | 422 | 577 | 646 |
| Hybrid, không rerank | single | 100 | 0.930 | 0.800 | — | 437 | 558 | 945 |
| Hybrid, không rerank | multi_intent | 100 | 0.990 | 0.906 | 0.915 | 535 | 638 | 713 |
| Hybrid, không rerank | multi_hop | 100 | 0.340 | 0.201 | — | 453 | 647 | 782 |
| Hybrid + rerank (luôn chạy, 2 luồng) | single | 100 | 0.930 | 0.836 | — | 2889 | 3729 | 4889 |
| Hybrid + rerank (luôn chạy, 2 luồng) | multi_intent | 100 | 1.000 | 0.898 | 0.945 | 3550 | 3908 | 4471 |
| Hybrid + rerank (luôn chạy, 2 luồng) | multi_hop | 100 | 0.380 | 0.195 | — | 3090 | 3664 | 4077 |
| **Cấu hình cuối**: hybrid + rerank có bảo vệ ngân sách (1 luồng) | single | 100 | 0.960 | 0.820 | — | 2298 | 2701 | 2837 |
| **Cấu hình cuối** | multi_intent | 100 | 1.000 | 0.909 | 0.935 | 2391 | 2693 | 2736 |
| **Cấu hình cuối** | multi_hop | 100 | 0.370 | 0.200 | — | 2360 | 2663 | 2712 |

Độ trễ của ba cấu hình đầu bị lệch: hai cấu hình không rerank chạy lại khi Gemini đã được cache nên chỉ phản ánh truy xuất (~0,4-0,5 s);
cấu hình "luôn chạy rerank" chạy 2 luồng nên CPU bị chia, vượt 3 giây. Chỉ dòng "Cấu hình cuối" là phép đo độ trễ chuẩn
(một yêu cầu mỗi lần, Gemini chưa cache): **p95 tối đa 2,70 giây, lớn nhất 2,84 giây trên 300 query**.

## Nhận xét

1. **Hybrid là yếu tố quyết định.** Chuyển từ chỉ vector ảnh sang hybrid (thêm vector text và sparse trên tên thực thể, mô tả, caption)
   làm Recall@10 một ý tăng từ 0,32 lên 0,93, nhiều ý từ 0,35 lên 0,99, suy luận từ 0,11 lên 0,34. Phù hợp giả thuyết: embedding ảnh không
   biết tên riêng bản địa.
2. **Rerank cải thiện nhỏ.** Thêm khoảng +0,03 Recall@10 (một ý, suy luận) và +0,02 độ phủ ý, nhưng tốn 0,7-1,0 giây. Vì vậy chỉ chạy khi còn dư
   ngân sách thời gian; khi không còn thì trả thứ tự hybrid. Sự chênh lệch giữa "cấu hình cuối" và "luôn rerank" ở Recall@10 (0,96 và 0,93) nằm
   trong sai số của 100 query nên không nên đọc là cải thiện thật.
3. **Nhóm suy luận khó nhất** (0,37). Đáp án là cả nhóm thể loại và địa bàn nên tập đáp án rộng, nhưng truy vấn khái niệm ("lễ hội ở Tây Nguyên")
   phải khớp đúng thể loại và vùng mà chỉ metadata phản ánh; LLM mở rộng thành thực thể cụ thể nhưng không phải thực thể nào cũng nằm trong nhóm
   đáp án.

## Hạn chế của phép đo (đọc kỹ trước khi trích số)

- **Query "một ý" dùng tên thực thể, và tên thực thể nằm trong văn bản đã index.** Đây là truy vấn dễ nhất có thể; 0,93-0,96 cho thấy
  hệ thống tìm đúng khi người dùng biết tên, không phải khi người dùng mô tả bằng lời của mình. Chưa có bộ query diễn đạt tự do.
- **Đáp án của truy vấn nhóm là xấp xỉ.** "trang phục nam bộ" coi mọi ảnh cùng thể loại và địa bàn là đúng, kể cả ảnh người dùng có thể không thấy phù hợp.
- **Chưa có người duyệt.** `eval/datasets/review-seed0.csv` liệt kê query kèm số ảnh đáp án để duyệt tay; cột `verdict` còn trống.
- **Không ghi lại tỉ lệ truy vấn thực sự được rerank** trong dòng "Cấu hình cuối" (cơ chế bảo vệ ngân sách bỏ qua rerank khi không còn thời gian); vì vậy không thể tách riêng đóng góp của rerank trong dòng đó.
- **Một lần chạy, một seed, n=100 mỗi loại.** Khoảng sai số của một tỉ lệ ~0,93 với n=100 vào khoảng ±0,05.
- **Truy vấn mẫu trong ghi chú của đề** ("con hổ", "bãi biển Nha Trang", "hồ bơi và công viên nước ở Phú Quốc") có thể không có ảnh đúng trong dữ liệu
  di sản này (kết quả thực tế: tranh, đền, tháp Po Nagar, Nhà tù Phú Quốc). Đó là giới hạn của dữ liệu chứ không phải của hệ thống.
- Truy vấn kiểu "con trai của X" chỉ đúng khi LLM biết quan hệ và thực thể có trong dữ liệu; thử với "con trai chủ tịch Phạm Nhật Vượng" không ra
  ảnh liên quan vì dữ liệu không có.

## So sánh bộ rerank (đo ngày 2026-10-20)

Cùng danh sách 16 ứng viên hybrid cho mỗi truy vấn, 300 query, chạy trong cùng môi trường với API (`eval/compare_rerank.py`, kết quả `docs/results/rerank-ab.json`).
Khác với bảng trên, phép đo này không có ngân sách thời gian: rerank luôn chạy, nên số tuyệt đối khác bảng trên (ví dụ single 0,97 thay vì 0,96).

| Loại query | Không rerank | bge-reranker-base (local, INT8) | jina-reranker-v2 (endpoint của mentor) |
|---|---|---|---|
| single (100) | 0,92 | 0,97 | 0,97 |
| multi_intent (100) | 1,00 | 1,00 | 1,00 |
| multi_hop (100) | 0,30 | 0,34 | 0,34 |

- **Hai bộ rerank cho Recall@10 như nhau** trên bộ test này (mỗi nhóm 100 query, chênh lệch dưới ngưỡng phân biệt được).
- Độ trễ gọi endpoint ngoài p50 152 ms cho 16 văn bản (434 yêu cầu, 0 lỗi); local khoảng 0,7-1,0 s trên CPU (xem bảng trên).
- Hạn chế: endpoint ngoài có hạn mức chung 50 yêu cầu/phút và là tunnel tạm, nên chỉ dùng như tuỳ chọn có quay về local; chưa tích hợp vào API.

## Duyệt tay một mẫu bộ test (2026-10-21)

Bảng duyệt `eval/datasets/review-sheet-seed0.csv`: 45 truy vấn (15 mỗi loại, lấy mẫu cố định theo seed 0) từ bộ test 300 query. Người duyệt xem truy vấn, các thực thể đáp án và ảnh mẫu của đáp án
(`python -m eval.make_review_html`), rồi chọn `dung` / `sai` / `mo_ho`. Kết quả thô: `eval/datasets/review-done-seed0.csv`. Người duyệt: Nguyễn Tuấn Thành (duyệt ngày 2026-10-21; cùng người làm dự án, không phải người duyệt độc lập).

| Loại | Đúng | Mơ hồ | Sai | Tổng |
|---|---|---|---|---|
| single | 15 | 0 | 0 | 15 |
| multi_intent | 14 | 1 (`mu-0142`: "Chùa Non và Chùa Kim Phong") | 0 | 15 |
| multi_hop | 15 | 0 | 0 | 15 |
| **Cộng** | **44** | **1** | **0** | **45** |

Người duyệt không ghi chú lý do cho truy vấn mơ hồ. Điều đã duyệt là **đáp án có hợp lý với truy vấn không**, không phải kết quả của hệ thống.

Recall@10 của hệ thống trên 44 truy vấn được duyệt "đúng" (`docs/results/eval-reviewed-44.json`, đo 2026-10-21, hybrid + rerank, rerank qua endpoint ngoài có quay về local):

| Loại | n | Recall@10 | MRR@10 | Phủ ý@10 |
|---|---|---|---|---|
| single | 15 | 0,933 (14/15) | 0,876 | — |
| multi_intent | 14 | 1,000 | 0,964 | 0,893 |
| multi_hop | 15 | 0,333 (5/15) | 0,240 | — |

Hạn chế khi đọc kết quả này:
- Mẫu nhỏ (14-15 truy vấn mỗi loại): chênh lệch vài điểm so với 300 query (0,96 / 1,00 / 0,37) nằm trong dao động của mẫu nhỏ, không kết luận được hệ thống tốt hơn hay kém hơn trên phần đã duyệt.
- Kết quả duyệt (44/45 đúng) cho thấy đáp án sinh tự động hợp lý với truy vấn trên mẫu này; không chứng minh 255 truy vấn còn lại cũng vậy.
- Truy vấn một ý vẫn dùng tên thực thể nên dễ hơn truy vấn của người dùng thật (đã nêu ở phần hạn chế phía trên).

## Cải tiến cách ghép truy vấn (đo 2026-10-21)

**Chẩn đoán** (100 truy vấn suy luận, chỉ truy xuất): chỉ 32/100 trúng. Truy vấn suy luận của bộ test là "thể loại + vùng" (ví dụ "di tích lịch sử - văn hóa hải dương"), mà văn bản đã index
có sẵn "Địa bàn" và "Thể loại" của từng ảnh. Nguyên nhân sai: (1) mã nối tên các **thực thể do Gemini đoán** vào văn bản tìm, nên các thực thể đoán sai
(ví dụ "Bánh đậu xanh Hải Dương" cho truy vấn di tích) kéo kết quả sang thực thể khác; (2) với truy vấn Gemini phân loại là suy luận, mã bỏ truy vấn gốc và chỉ tìm riêng từng thực thể đoán.

**So sánh có kiểm soát** (cùng kết quả phân tích Gemini cho 300 query, chỉ truy xuất, chưa rerank; `eval/ablate_units.py`, kết quả `docs/results/units-ablation.json`; Recall@10, phủ ý của nhiều ý trong ngoặc):

| Cách ghép (`retrieval.search.POLICIES`) | Suy luận | Nhiều ý | Một ý |
|---|---|---|---|
| `current` (trước đó) | 0,30 | 0,99 (0,92) | 0,92 |
| `vi_only`: không nối thực thể vào văn bản | 0,43 | 1,00 (0,965) | 1,00 |
| `mh_keep_query`: suy luận tìm thêm truy vấn gốc | 0,32 | 0,99 (0,92) | 0,92 |
| **`both`** (chọn) | **0,45** | 1,00 (0,965) | 1,00 |
| `ignore_ents`: bỏ hẳn thực thể | 0,44 | 1,00 (0,965) | 1,00 |

Chọn `both` thay vì `ignore_ents` vì vẫn giữ các thực thể Gemini suy ra như những ý tìm thêm, cần cho truy vấn quan hệ thật (ví dụ "con trai chủ tịch ...").
Bộ test không có truy vấn quan hệ thật, nên khả năng này **chưa được đo**; chỉ được giữ lại theo thiết kế.

**Đo toàn bộ qua API** (hybrid + rerank, 300 query, `docs/results/eval-units-both.json`, 2026-10-21) so với lần đo 2026-10-19 (`eval-final.json`):

| Loại | Recall@10 trước | Recall@10 sau | MRR@10 trước | MRR@10 sau |
|---|---|---|---|---|
| single | 0,96 | 0,99 | 0,820 | 0,944 |
| multi_intent (phủ ý) | 1,00 (0,935) | 1,00 (0,975) | 0,909 | 0,965 |
| multi_hop | 0,37 | 0,50 | 0,200 | 0,373 |

Hạn chế khi đọc bảng này:
- Hai lần đo cách nhau hai ngày và khác điều kiện (lần sau rerank qua endpoint của mentor có quay về local, và trạng thái cache Gemini khác), nên phần cải thiện **không hoàn toàn do cách ghép truy vấn**;
  phép so sánh có kiểm soát ở bảng trên (cùng phân tích, không rerank) là bằng chứng chính.
- Một truy vấn suy luận chạm 3.198 ms (p95 2.676 ms; max trước đó 2.712 ms): cách ghép mới thêm một lần nhúng cho truy vấn suy luận.
- Bộ test sinh từ chính các trường "thể loại" và "địa bàn" của dữ liệu nên có lợi cho cách ghép dựa vào văn bản; truy vấn của người dùng thật có thể kém hơn.
- Suy luận vẫn là điểm yếu: 0,50.
