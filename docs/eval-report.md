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
- **Một lần chạy, một seed, n=100 mỗi loại.** Khoảng sai số của một tỉ lệ ~0,93 với n=100 vào khoảng ±0,05.
- **Truy vấn mẫu trong ghi chú của đề** ("con hổ", "bãi biển Nha Trang", "hồ bơi và công viên nước ở Phú Quốc") có thể không có ảnh đúng trong dữ liệu
  di sản này (kết quả thực tế: tranh, đền, tháp Po Nagar, Nhà tù Phú Quốc). Đó là giới hạn của dữ liệu chứ không phải của hệ thống.
- Truy vấn kiểu "con trai của X" chỉ đúng khi LLM biết quan hệ và thực thể có trong dữ liệu; thử với "con trai chủ tịch Phạm Nhật Vượng" không ra
  ảnh liên quan vì dữ liệu không có.
