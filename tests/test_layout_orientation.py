from __future__ import annotations

from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import blue, red
from reportlab.pdfgen import canvas

from rich_renderer import (
    create_side_by_side_pdf,
    create_vertical_dual_pdf,
    page_display_size,
)


def _solid_page(path: Path, width: float, height: float, color) -> None:
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.setFillColor(color)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.showPage()
    pdf.save()


def _rotate_page(source: Path, target: Path, degrees: int) -> None:
    page = PdfReader(str(source)).pages[0]
    page.rotate(degrees)
    writer = PdfWriter()
    writer.add_page(page)
    with target.open("wb") as stream:
        writer.write(stream)


def test_side_by_side_uses_visible_orientation(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    source = tmp_path / "rotated.pdf"
    rebuilt = tmp_path / "rebuilt.pdf"
    output = tmp_path / "side.pdf"
    _solid_page(raw, 300, 500, red)
    _rotate_page(raw, source, 90)
    _solid_page(rebuilt, 500, 300, blue)

    assert page_display_size(PdfReader(str(source)).pages[0]) == (500, 300)
    create_side_by_side_pdf(source, [1], [rebuilt], output)

    page = PdfReader(str(output)).pages[0]
    assert float(page.mediabox.width) == 1000
    assert float(page.mediabox.height) == 300
    assert int(page.get("/Rotate", 0) or 0) % 360 == 0

    with fitz.open(output) as rendered:
        pixmap = rendered[0].get_pixmap(colorspace=fitz.csRGB, alpha=False)
        left = pixmap.pixel(pixmap.width // 4, pixmap.height // 2)
        right = pixmap.pixel(pixmap.width * 3 // 4, pixmap.height // 2)
    assert left[0] > 200 and left[2] < 80
    assert right[2] > 200 and right[0] < 80


def test_vertical_uses_visible_orientation(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    source = tmp_path / "rotated.pdf"
    rebuilt = tmp_path / "rebuilt.pdf"
    output = tmp_path / "vertical.pdf"
    _solid_page(raw, 300, 500, red)
    _rotate_page(raw, source, 90)
    _solid_page(rebuilt, 500, 300, blue)

    create_vertical_dual_pdf(source, 1, rebuilt, output)

    page = PdfReader(str(output)).pages[0]
    assert float(page.mediabox.width) == 500
    assert float(page.mediabox.height) == 600
    assert int(page.get("/Rotate", 0) or 0) % 360 == 0
