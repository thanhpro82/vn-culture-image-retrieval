from eval.make_review_html import render


def test_embeds_rows_as_json_and_escapes_script_end():
    html = render([{"id": "a", "type": "single", "query": "x</script><b>", "gold": "g", "verdict": "", "note": ""}])
    assert "</script><b>" not in html           # dữ liệu không được đóng thẻ <script> của trang
    assert '"id": "a"' in html
    assert "/*__DATA__*/[]" not in html


def test_keeps_vietnamese_text_unescaped():
    html = render([{"id": "a", "type": "single", "query": "hát chèo", "gold": "Đáp án", "verdict": "", "note": ""}])
    assert "hát chèo" in html and "Đáp án" in html
