from __future__ import annotations

import tempfile
from pathlib import Path

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from app import normalize_output_selection
from rich_renderer import create_chinese_only_pdf, create_side_by_side_pdf


def _page(path: Path, width: float = 300, height: float = 500) -> None:
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.drawString(30, height - 40, path.stem)
    pdf.showPage()
    pdf.save()


def run() -> None:
    assert normalize_output_selection("bilingual_vertical") == ["bilingual_vertical"]
    assert normalize_output_selection("bilingual_side_by_side") == ["bilingual_side_by_side"]
    assert normalize_output_selection("chinese_only") == ["chinese_only"]
    assert normalize_output_selection(["双语左右并排"]) == ["bilingual_side_by_side"]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.pdf"
        rebuilt = root / "rebuilt.pdf"
        side = root / "side.pdf"
        chinese = root / "chinese.pdf"
        _page(source)
        _page(rebuilt)

        second = normalize_output_selection("bilingual_side_by_side")
        third = normalize_output_selection("chinese_only")
        if second == ["bilingual_side_by_side"]:
            create_side_by_side_pdf(source, [1], [rebuilt], side)
        if third == ["chinese_only"]:
            create_chinese_only_pdf([rebuilt], chinese, [1])

        side_page = PdfReader(str(side)).pages[0]
        chinese_page = PdfReader(str(chinese)).pages[0]
        assert (float(side_page.mediabox.width), float(side_page.mediabox.height)) == (600, 500)
        assert (float(chinese_page.mediabox.width), float(chinese_page.mediabox.height)) == (300, 500)
        assert side.name != chinese.name

    print("output selection regression: PASS")


if __name__ == "__main__":
    run()
