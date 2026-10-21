"""Bất biến của bộ slide (docs/slides.html): khung cố định, tự chứa, in được, đủ số liệu chính."""
import re
from pathlib import Path

HTML = Path("docs/slides.html").read_text(encoding="utf-8")


def test_has_ten_slides_in_a_fixed_16_9_stage():
    assert len(re.findall(r'<section class="slide', HTML)) == 10
    assert "width: 1920px; height: 1080px" in HTML


def test_is_self_contained_no_external_resources():
    assert not re.search(r'(src|href)="https?://', HTML)
    assert "@import" not in HTML and "<link" not in HTML


def test_supports_print_and_reduced_motion():
    assert "@media print" in HTML and "@page { size: 1920px 1080px" in HTML
    assert "@media (prefers-reduced-motion: reduce)" in HTML


def test_motion_only_uses_transform_and_opacity():
    for m in re.findall(r"transition:\s*([^;}]+)", HTML):
        for part in m.split(","):
            prop = part.split()[0]
            assert prop in {"transform", "opacity", "visibility", "none"}, m


def test_contains_the_key_measured_numbers():
    for needle in ("89.596", "0,99", "0,70", "0,83", "0,38", "92,2%", "21,6%", "95,7%", "140-323 ms", "VSF-AR-ES-V-Lookup"):
        assert needle in HTML, needle


def test_navigation_keys_and_deep_link_are_wired():
    for needle in ("ArrowRight", "ArrowLeft", '"Home"', '"End"', "location.hash"):
        assert needle in HTML, needle


def test_pdf_export_exists_and_has_ten_pages():
    pdf = Path("docs/slides.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page[^s]", pdf)) == 10
