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
