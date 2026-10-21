# Bộ test (phiên bản hoá)

Sinh bởi `eval/build_testset.py` (Task 13). Mỗi lần sinh ghi seed và ngày.

## Duyệt tay một mẫu (review-sheet)

`python -m eval.make_review_sheet` tạo `review-sheet-seed0.csv`: 45 truy vấn (15 mỗi loại, lấy mẫu cố định theo seed), mỗi dòng kèm các thực thể đáp án
(tối đa 5 thực thể nhiều ảnh nhất, số trong ngoặc là số ảnh đáp án thuộc thực thể đó). Mở bằng Excel, điền cột `verdict`:

| `verdict` | Khi nào |
|---|---|
| `dung` | Một người dùng hỏi câu này sẽ chấp nhận các thực thể đáp án là kết quả hợp lệ |
| `sai` | Đáp án lạc đề, hoặc có phần đáng kể không liên quan tới truy vấn |
| `mo_ho` | Truy vấn quá rộng hoặc có thể hiểu theo nhiều cách, đáp án chỉ là một cách hiểu |

Gợi ý theo loại: `single`: truy vấn là tên thực thể, đáp án có đúng là ảnh của thực thể đó không. `multi_intent`: mỗi ý (`Ý 1`, `Ý 2`) có hợp lý và đáp án từng ý có đúng không.
`multi_hop`: truy vấn là một nhóm khái niệm (văn hoá + vùng); các thực thể đáp án có phải thứ người dùng mong đợi không. Cột `note` để ghi lý do khi chọn `sai` hoặc `mo_ho`.
Ghi tên người duyệt và ngày ở cuối file hoặc trong báo cáo. Khi có kết quả, báo cáo ghi tỉ lệ `dung`/`sai`/`mo_ho` theo từng loại và đo lại Recall@10 trên phần `dung`.

### Duyệt nhanh bằng trang HTML

`python -m eval.make_review_html` sinh `review-form-seed0.html` từ bảng CSV ở trên, kèm ảnh thu nhỏ của đáp án (cần `Data/images` và `Data/manifest.parquet`). Mở thẳng bằng trình duyệt, không cần mạng. **Không commit** file này (repo công khai, trang nhúng ảnh crawl từ nguồn khác): `.gitignore` đã chặn; gửi riêng cho người duyệt.
Mỗi lần hiện một truy vấn kèm ảnh mẫu của từng ý (trải đều qua các thực thể đáp án, bấm ảnh để phóng to): phím **1** = đúng, **2** = sai, **3** = mơ hồ (tự chuyển sang truy vấn tiếp theo), **←/→** đi lại, **N** gõ ghi chú, **Esc** thoát ô ghi chú.
Tiến độ tự lưu trong trình duyệt (đóng tab rồi mở lại vẫn còn, cùng máy và cùng trình duyệt). Bấm **Xuất CSV** để tải file `review-done-<ngày>.csv`
(thêm cột `reviewer`); gửi file đó để đưa kết quả vào báo cáo.
