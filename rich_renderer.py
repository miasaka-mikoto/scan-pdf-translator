from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.pdfbase.ttfonts import TTFont

from mathjax_renderer import render_formula_svg
from figure_vectorizer import vectorize_figure_svg
from translator import _message_text, _post_json, display_text


CJK_FONT = "EmbeddedCJK"


def _ensure_cjk_font() -> str:
    """Register an embedded TrueType CJK font for browser PDF viewers."""
    try:
        pdfmetrics.getFont(CJK_FONT)
        return CJK_FONT
    except KeyError:
        pass
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(CJK_FONT, str(font_path)))
            return CJK_FONT
    raise FileNotFoundError("找不到可嵌入的中文 TrueType 字体（simhei.ttf/Deng.ttf）")


@dataclass
class Region:
    id: str
    kind: str
    bbox: tuple[int, int, int, int]
    source: str
    translated: str = ""
    confidence: float | None = None
    fallback_reason: str = ""
    status: str = "pending"


MARKER = re.compile(
    r"<\|ref\|>(?P<kind>[^<]+)<\|/ref\|>"
    r"<\|det\|>\[\[(?P<bbox>\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\]\]<\|/det\|>"
)


def parse_grounded_markdown(markdown: str) -> list[Region]:
    matches = list(MARKER.finditer(markdown))
    regions: list[Region] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        source = markdown[match.end() : end].strip()
        bbox = tuple(int(value.strip()) for value in match.group("bbox").split(","))
        regions.append(
            Region(
                id=f"r{index + 1:02d}",
                kind=match.group("kind").strip(),
                bbox=bbox,  # type: ignore[arg-type]
                source=source,
            )
        )
    return regions


def translate_regions(api_key: str, regions: list[Region], model: str, glossary: str = "") -> tuple[list[Region], dict]:
    translatable = [
        region
        for region in regions
        if region.kind in {"text", "sub_title", "title", "image_caption", "equation"} and region.source
    ]
    request_data = [
        {"id": region.id, "type": region.kind, "text": region.source}
        for region in translatable
    ]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是射频电路教材翻译引擎。将输入 JSON 中每个块独立翻译成简体中文。"
                    "公式块必须保持 LaTeX 结构、变量、数值、单位和式号，只翻译公式中的英文说明。"
                    "图号、变量和元件符号保持不变。禁止分析、解释、合并或遗漏。"
                    + ("\n必须遵循下列术语表（英文=中文）：\n" + glossary if glossary.strip() else "")
                    + "只返回 JSON 对象，格式为 {\"translations\":[{\"id\":\"r01\",\"text\":\"译文\"}]}。"
                ),
            },
            {"role": "user", "content": json.dumps(request_data, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "stream": True,
    }
    response = _post_json(api_key, payload)
    content = _message_text(response)
    try:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        data = json.loads(content[start : end + 1])
        mapping = {
            item["id"]: item["text"]
            for item in data.get("translations", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("text"), str)
        }
        missing = [region.id for region in translatable if region.id not in mapping]
        if missing:
            raise ValueError(f"missing IDs: {missing}")
        usage: dict = response.get("usage", {})
    except (json.JSONDecodeError, ValueError, TypeError):
        # Small/free models occasionally emit an unescaped quotation mark in a
        # JSON string. Do not lose the whole page: retry each stable region as
        # plain text, which removes the JSON-serialization failure mode.
        mapping = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "translation_fallback": "per_block_plain_text"}
        system = (
            "你是射频电路教材翻译引擎。忠实翻译为简体中文。保留 LaTeX、变量、数值、单位、图号和式号。"
            "禁止解释、分析、Markdown 代码围栏或任何前缀；只输出该区块的译文。"
            + ("\n必须遵循术语表：\n" + glossary if glossary.strip() else "")
        )
        for region in translatable:
            single = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": region.source}],
                "temperature": 0.0,
                "max_tokens": 2048,
                "enable_thinking": False,
                "stream": True,
            }
            single_response = _post_json(api_key, single)
            mapping[region.id] = _message_text(single_response).strip()
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = single_response.get("usage", {}).get(key, 0)
                if isinstance(value, (int, float)):
                    usage[key] += int(value)
    for region in regions:
        region.translated = mapping.get(region.id, "")
        if region.id in mapping:
            region.status = "translated"
    return regions, usage


def _wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        if character == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        candidate = current + character
        if current and pdfmetrics.stringWidth(candidate, font, size) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _draw_text_box(c: canvas.Canvas, text: str, bbox: tuple[float, float, float, float], kind: str) -> None:
    x, y, width, height = bbox
    text = display_text(text)
    if not text:
        return
    padding = 1.8
    available_width = max(5, width - 2 * padding)
    available_height = max(5, height - 2 * padding)
    maximum = 11.5 if kind in {"title", "sub_title"} else 9.5
    minimum = 4.6
    chosen_size = minimum
    chosen_lines: list[str] = []
    size = maximum
    while size >= minimum:
        lines = _wrap(text, CJK_FONT, size, available_width)
        leading = size * 1.28
        if len(lines) * leading <= available_height:
            chosen_size, chosen_lines = size, lines
            break
        size -= 0.35
    if not chosen_lines:
        chosen_lines = _wrap(text, CJK_FONT, chosen_size, available_width)
    c.setFillColor(HexColor("#111827"))
    c.setFont(CJK_FONT, chosen_size)
    leading = chosen_size * 1.28
    cursor_y = y + height - padding - chosen_size
    for line in chosen_lines:
        if cursor_y < y + padding - 0.1:
            break
        if kind == "image_caption":
            line_width = pdfmetrics.stringWidth(line, CJK_FONT, chosen_size)
            c.drawString(x + max(padding, (width - line_width) / 2), cursor_y, line)
        else:
            c.drawString(x + padding, cursor_y, line)
        cursor_y -= leading


def _pdf_bbox(region: Region, width: float, height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = region.bbox
    x = x0 / 1000 * width
    top = y0 / 1000 * height
    region_width = (x1 - x0) / 1000 * width
    region_height = (y1 - y0) / 1000 * height
    y = height - top - region_height
    return x, y, region_width, region_height


def _crop_region(image: Image.Image, region: Region) -> Image.Image:
    x0, y0, x1, y1 = region.bbox
    left = max(0, round(x0 / 1000 * image.width))
    top = max(0, round(y0 / 1000 * image.height))
    right = min(image.width, round(x1 / 1000 * image.width))
    bottom = min(image.height, round(y1 / 1000 * image.height))
    return image.crop((left, top, right, bottom))


def _deterministic_figure_cleanup(crop: Image.Image) -> Image.Image:
    """Improve scanned line art without inventing or moving any geometry."""
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(1.12)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray.convert("RGB")


def create_reconstructed_page(
    source_image: Path,
    regions: list[Region],
    page_width: float,
    page_height: float,
    output_pdf: Path,
    label: str,
    formula_strategy: str = "mathjax",
    figure_strategy: str = "preserve",
    assets_dir: Path | None = None,
) -> None:
    _ensure_cjk_font()
    image = Image.open(source_image).convert("RGB")
    c = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))
    c.setFillColor(HexColor("#FFFFFF"))
    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    c.setFillColor(HexColor("#4B5563"))
    c.setFont(CJK_FONT, 7.2)
    c.drawRightString(page_width - 10, page_height - 12, "下半页 · 中文图文重构")
    for region in regions:
        box = _pdf_bbox(region, page_width, page_height)
        if region.kind in {"image", "table", "chart", "circuit"}:
            crop = _crop_region(image, region)
            vector_drawn = False
            if figure_strategy == "vector_trace" and assets_dir:
                svg_path = vectorize_figure_svg(crop, assets_dir / f"{region.id}-figure.svg")
                if svg_path:
                    try:
                        from svglib.svglib import svg2rlg
                        drawing = svg2rlg(str(svg_path))
                        scale = min(box[2] / drawing.width, box[3] / drawing.height)
                        drawing.scale(scale, scale)
                        renderPDF.draw(drawing, c, box[0] + (box[2] - drawing.width * scale) / 2, box[1] + (box[3] - drawing.height * scale) / 2)
                        region.status = "figure_vector_traced"
                        region.fallback_reason = ""
                        vector_drawn = True
                    except Exception:
                        region.fallback_reason = "figure_svg_embed_failed"
            if vector_drawn:
                continue
            if figure_strategy in {"preserve", "vector_trace"}:
                crop = _deterministic_figure_cleanup(crop)
            c.drawImage(ImageReader(crop), *box, preserveAspectRatio=False, mask="auto")
            region.status = "source_preserved"
            region.fallback_reason = region.fallback_reason or "deterministic_source_crop"
        elif region.kind == "equation":
            rendered = None
            if formula_strategy == "mathjax" and assets_dir:
                rendered = render_formula_svg(region.translated or region.source, assets_dir / f"{region.id}.svg", assets_dir)
            if rendered:
                try:
                    from svglib.svglib import svg2rlg
                    drawing = svg2rlg(str(rendered))
                    scale = min(box[2] / drawing.width, box[3] / drawing.height)
                    drawing.scale(scale, scale)
                    renderPDF.draw(drawing, c, box[0] + (box[2] - drawing.width * scale) / 2, box[1] + (box[3] - drawing.height * scale) / 2)
                    region.status = "formula_svg"
                except Exception:
                    crop = _crop_region(image, region)
                    c.drawImage(ImageReader(crop), *box, preserveAspectRatio=False, mask="auto")
                    region.status = "source_preserved"
                    region.fallback_reason = "mathjax_svg_embed_failed"
            else:
                crop = _crop_region(image, region)
                c.drawImage(ImageReader(crop), *box, preserveAspectRatio=False, mask="auto")
                region.status = "source_preserved"
                region.fallback_reason = "formula_source_crop_strategy_or_mathjax_failed"
        elif region.translated:
            _draw_text_box(c, region.translated, box, region.kind)
            region.status = "composed"
    c.showPage()
    c.save()


def create_vertical_dual_pdf(
    source_pdf: Path,
    page_number: int,
    reconstructed_pdf: Path,
    output_pdf: Path,
) -> None:
    source_page = PdfReader(str(source_pdf)).pages[page_number - 1]
    rebuilt_page = PdfReader(str(reconstructed_pdf)).pages[0]
    width = float(source_page.mediabox.width)
    height = float(source_page.mediabox.height)
    writer = PdfWriter()
    combined = writer.add_blank_page(width=width, height=height * 2)
    combined.merge_transformed_page(source_page, Transformation().translate(tx=0, ty=height))
    combined.merge_page(rebuilt_page)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def create_chinese_only_pdf(rebuilt_pages: list[Path], output_pdf: Path, page_numbers: list[int] | None = None) -> None:
    writer = PdfWriter()
    for index, rebuilt in enumerate(rebuilt_pages):
        writer.add_page(PdfReader(str(rebuilt)).pages[0])
        if page_numbers:
            writer.add_outline_item(f"原 PDF 第 {page_numbers[index]} 页", index)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def create_side_by_side_pdf(source_pdf: Path, pages: list[int], rebuilt_pages: list[Path], output_pdf: Path) -> None:
    source = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page_number, rebuilt in zip(pages, rebuilt_pages, strict=True):
        original = source.pages[page_number - 1]
        translated = PdfReader(str(rebuilt)).pages[0]
        width, height = float(original.mediabox.width), float(original.mediabox.height)
        page = writer.add_blank_page(width=width * 2, height=height)
        page.merge_page(original)
        page.merge_transformed_page(translated, Transformation().translate(tx=width, ty=0))
        writer.add_outline_item(f"原 PDF 第 {page_number} 页", len(writer.pages) - 1)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)
