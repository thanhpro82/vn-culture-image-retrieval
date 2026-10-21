from eval.make_review_html import render


def test_embeds_rows_as_json_and_escapes_script_end():
    html = render([{"id": "a", "type": "single", "query": "x</script><b>", "gold": "g", "verdict": "", "note": ""}])
    assert "</script><b>" not in html           # dữ liệu không được đóng thẻ <script> của trang
    assert '"id": "a"' in html
    assert "/*__DATA__*/[]" not in html


def test_keeps_vietnamese_text_unescaped():
    html = render([{"id": "a", "type": "single", "query": "hát chèo", "gold": "Đáp án", "verdict": "", "note": ""}])
    assert "hát chèo" in html and "Đáp án" in html


def test_pick_images_spreads_across_entities_before_repeating():
    from eval.make_review_html import pick_images
    names = {"a1": ["A"], "a2": ["A"], "a3": ["A"], "b1": ["B"], "b2": ["B"], "c1": ["C"]}
    picked = pick_images(["a1", "a2", "a3", "b1", "b2", "c1"], names, 3)
    assert {names[i][0] for i in picked} == {"A", "B", "C"}  # mỗi thực thể một ảnh trước khi lặp
    assert len(picked) == 3


def test_pick_images_handles_fewer_images_than_k_and_unknown_ids():
    from eval.make_review_html import pick_images
    assert pick_images(["x", "y"], {}, 6) == ["x", "y"]


def test_thumb_returns_small_jpeg_data_uri(tmp_path):
    from eval.make_review_html import thumb
    from PIL import Image
    p = tmp_path / "i.webp"
    Image.new("RGB", (800, 600), (200, 30, 30)).save(p, "WEBP")
    uri = thumb(p, side=100)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) < 6000


def test_thumb_missing_file_is_empty_string(tmp_path):
    from eval.make_review_html import thumb
    assert thumb(tmp_path / "none.webp") == ""


def test_build_rows_attaches_images_per_intent(tmp_path):
    from eval.make_review_html import build_rows
    from PIL import Image
    for i in ("a", "b", "c"):
        Image.new("RGB", (50, 50), (1, 2, 3)).save(tmp_path / f"{i}.webp", "WEBP")
    sheet = [{"id": "t1", "type": "multi_intent", "query": "q", "gold": "g", "verdict": "", "note": ""}]
    tests = {"t1": {"id": "t1", "intents": [["a"], ["b", "c"]]}}
    rows = build_rows(sheet, tests, {"a": ["A"], "b": ["B"], "c": ["B"]}, tmp_path, per_intent=2)
    assert len(rows[0]["imgs"]) == 2 and len(rows[0]["imgs"][0]) == 1 and len(rows[0]["imgs"][1]) == 2
