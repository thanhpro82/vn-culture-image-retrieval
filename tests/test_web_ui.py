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


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block))


def _lum(h: str) -> float:
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _themes() -> dict[str, dict[str, str]]:
    light = _tokens(HTML[HTML.index(":root {"):HTML.index("@media (prefers-color-scheme: dark)")])
    dark_start = HTML.index("@media (prefers-color-scheme: dark)")
    dark = {**light, **_tokens(HTML[dark_start:HTML.index("}", HTML.index(":root {", dark_start))])}
    return {"sáng": light, "tối": dark}


@pytest.mark.parametrize("theme", ["sáng", "tối"])
def test_text_colours_meet_wcag_aa_in_both_themes(theme):
    t = _themes()[theme]
    for fg, bg in [("ink", "ground"), ("ink", "surface"), ("muted", "ground"), ("muted", "surface"), ("on-accent", "accent"),
                   ("warn-ink", "warn-bg"), ("err-ink", "err-bg"), ("ground", "ink")]:
        assert _contrast(t[fg], t[bg]) >= 4.5, (theme, fg, bg, round(_contrast(t[fg], t[bg]), 2))
    assert _contrast(t["accent"], t["ground"]) >= 3.0, theme  # thanh nhấn và viền focus


def test_no_raw_hex_colours_outside_the_palette_tokens():
    """Mọi màu đi qua token: mã màu rời trong một thành phần là bảng màu thứ hai bắt đầu."""
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    css_wo_tokens = re.sub(r":root\s*\{[^}]*\}", "", css)
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b", css_wo_tokens)


def test_accent_is_not_used_for_large_fills_or_headings():
    """Màu nhấn chiếm ít chỗ: chỉ nút chính, thanh nhấn của ô tìm và viền focus."""
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    uses = [u for u in re.findall(r"([^{}]+)\{[^}]*var\(--accent\)", css) if "focus" not in u]  # viền focus dùng màu nhấn là đúng
    selectors = " ".join(u.strip() for u in uses)
    for forbidden in (".chip", ".card", "h1", ".understood", ".summary", "dialog"):
        assert forbidden not in selectors.replace(".go:disabled", ""), forbidden


def test_page_declares_color_scheme_for_native_controls():
    assert "color-scheme: light dark" in HTML
