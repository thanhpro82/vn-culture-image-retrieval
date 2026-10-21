"""Kiểm tra bất biến của trang web demo (web/index.html), không chạy trình duyệt."""
import re
from pathlib import Path

import pytest
from eval.build_testset_v2 import OUT_OF_SCOPE

HTML = Path("web/index.html").read_text(encoding="utf-8")
GOOD_EXAMPLES = ["cầu Long Biên", "nhà thờ Phát Diệm", "áo dài Huế và nhà thờ Phát Diệm", "chợ nổi và múa lân"]


def test_examples_are_verified_queries_not_out_of_data_ones():
    for q in GOOD_EXAMPLES:
        assert q in HTML, q
    for q in OUT_OF_SCOPE + ["động vật sống trong rừng", "con hổ và con mèo", "hồ bơi và công viên nước ở Phú Quốc"]:
        assert q not in HTML, f"gợi ý mẫu không được chứa truy vấn ngoài dữ liệu: {q}"


def test_page_never_builds_markup_from_data():
    assert "innerHTML" not in HTML and "insertAdjacentHTML" not in HTML and "document.write" not in HTML


@pytest.mark.parametrize("needle", ["prefers-reduced-motion", "<dialog", "skeleton", "aria-live", "focus-visible", "low_confidence", "429", "503"])
def test_page_has_accessibility_and_state_features(needle):
    assert needle in HTML


def test_page_shows_how_the_query_was_understood():
    assert "d.subs" in HTML or ".subs" in HTML


def test_broken_images_keep_their_card():
    assert "card.remove()" not in HTML


def test_no_external_resources():
    assert not re.search(r'(src|href)="https?://', HTML)
    assert "@import" not in HTML


def test_motion_only_animates_transform_and_opacity():
    """Chuyển động chạy trên GPU: keyframes chỉ được đổi transform và opacity (không đổi bố cục)."""
    blocks = re.findall(r"@keyframes\s+[\w-]+\s*\{(.*?)\}\s*(?=@|\n\s*\.|\n\s*/\*|\n\s*[a-z#.\[]|\n\s*\})", HTML, re.DOTALL)
    assert blocks, "không tìm thấy keyframes"
    for b in blocks:
        props = set(re.findall(r"([a-z-]+)\s*:", b))
        assert props <= {"opacity", "transform"}, props


def test_transitions_avoid_layout_properties():
    for m in re.findall(r"transition:\s*([^;}]+)", HTML):
        for layout in ("width", "height", "top", "left", "margin", "padding"):
            assert not re.search(rf"\b{layout}\b", m), m


def test_reduced_motion_switches_everything_off():
    block = HTML[HTML.index("@media (prefers-reduced-motion: reduce)"):]
    assert "animation: none !important" in block and "transition: none !important" in block
