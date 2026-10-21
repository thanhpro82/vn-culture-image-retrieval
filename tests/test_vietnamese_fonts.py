"""Georgia không vẽ đúng các dấu chồng của tiếng Việt (ế, ầ, ấ, ề...): dấu bị lệch khỏi chữ. Cấm dùng cho trang web và slide."""
from pathlib import Path

import pytest

PAGES = ["web/index.html", "docs/slides.html"]


@pytest.mark.parametrize("page", PAGES)
def test_display_font_is_not_georgia(page):
    html = Path(page).read_text(encoding="utf-8")
    assert "Georgia" not in html, f"{page}: dùng phông có hỗ trợ đủ dấu tiếng Việt (Palatino Linotype, Cambria...)"


@pytest.mark.parametrize("page", PAGES)
def test_display_font_stack_starts_with_a_vietnamese_capable_serif(page):
    html = Path(page).read_text(encoding="utf-8")
    line = next(ln for ln in html.splitlines() if "--display:" in ln)
    assert "Palatino Linotype" in line and "Cambria" in line, line


def test_reduced_motion_forces_headings_visible():
    """Người bật 'giảm chuyển động' vẫn phải thấy tiêu đề (chữ được che bằng overflow khi chưa hiện)."""
    html = Path("docs/slides.html").read_text(encoding="utf-8")
    block = html[html.index("@media (prefers-reduced-motion: reduce)"):]
    assert ".wi" in block and "transform: none !important" in block and ".w { overflow: visible" in block
