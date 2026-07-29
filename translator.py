from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


API_URL = "https://api.siliconflow.cn/v1/chat/completions"


@dataclass(frozen=True)
class Profile:
    slug: str
    label: str
    ocr_model: str
    translation_model: str
    reviewer_model: str | None = None


FAST_PROFILE = Profile(
    slug="fast",
    label="快速版",
    ocr_model="deepseek-ai/DeepSeek-OCR",
    translation_model="deepseek-ai/DeepSeek-V4-Flash",
)

QUALITY_PROFILE = Profile(
    slug="quality",
    label="最高质量版",
    ocr_model="deepseek-ai/DeepSeek-OCR",
    reviewer_model="Qwen/Qwen3-VL-32B-Thinking",
    translation_model="deepseek-ai/DeepSeek-V4-Pro",
)

PROFILES = {p.slug: p for p in (FAST_PROFILE, QUALITY_PROFILE)}


OCR_PROMPT = """Perform exact OCR on this scanned technical-book page.
Return clean Markdown in natural reading order.
Preserve headings, paragraphs, lists, figure captions, equations, variables, component labels, numbers, units, figure numbers and equation numbers exactly.
Do not translate, summarize, describe illustrations, or invent missing content.
Use LaTeX only when an equation is clearly visible. Output only Markdown."""

DEEPSEEK_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


REVIEW_PROMPT = """You are the second-pass OCR verifier for a scanned RF circuit-design textbook.
Compare the page image against the draft OCR below. Correct recognition errors, lost text, wrong reading order, broken equations, symbols, units, figure numbers and captions.
Do not translate, summarize, explain, or invent text. The draft may contain
`<|ref|>...<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>` grounding tags. Preserve
every grounding tag and its coordinates exactly, correcting only the text that
follows each tag. Output only corrected grounded Markdown.

DRAFT OCR:
"""


TRANSLATION_SYSTEM = """你是射频、模拟电路和电子工程教材的专业中英翻译引擎。
把输入的英文 Markdown 完整、忠实地翻译为简体中文。
必须保持标题、段落、列表、图题和原有顺序。
公式、LaTeX、变量、元件符号、数值、单位、图号、式号和引用不得改变。
输入可能包含 OCR 造成的连续机械重复；只保留第一份完整内容，删除明显重复的标签序列，不得把重复噪声继续复制到译文。
同一术语保持一致；不得总结、解释、扩写或遗漏。
只输出译文 Markdown。"""


def _post_json(api_key: str, payload: dict, retries: int = 5) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                if not payload.get("stream"):
                    return json.loads(response.read().decode("utf-8"))
                content_parts: list[str] = []
                usage: dict = {}
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError(f"SiliconFlow stream error: {chunk['error']}")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                if not content_parts:
                    raise RuntimeError("SiliconFlow stream returned no content")
                return {
                    "choices": [{"message": {"content": "".join(content_parts)}}],
                    "usage": usage,
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"SiliconFlow HTTP {exc.code}: {detail}")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"SiliconFlow request failed after {retries} attempts: {last_error}")


def _image_part(image_path: Path) -> dict:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded}",
            "detail": "high",
        },
    }


def _message_text(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected SiliconFlow response: {response}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"SiliconFlow returned empty content: {response}")
    return strip_code_fence(content)


def strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:markdown|md|text)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()


def ocr_page(api_key: str, image_path: Path, model: str) -> tuple[str, dict]:
    prompt = DEEPSEEK_OCR_PROMPT if model == "deepseek-ai/DeepSeek-OCR" else OCR_PROMPT
    max_tokens = 4096 if model == "deepseek-ai/DeepSeek-OCR" else 3072
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    _image_part(image_path),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    response = _post_json(api_key, payload)
    return _message_text(response), response.get("usage", {})


def review_ocr(api_key: str, image_path: Path, draft: str, model: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    _image_part(image_path),
                    {"type": "text", "text": REVIEW_PROMPT + draft},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "thinking_budget": 4096,
        "stream": True,
    }
    response = _post_json(api_key, payload)
    return _message_text(response), response.get("usage", {})


def translate_page(api_key: str, source_markdown: str, model: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM},
            {"role": "user", "content": source_markdown},
        ],
        "temperature": 0.05,
        "max_tokens": 4096,
        "enable_thinking": False,
        "stream": True,
    }
    response = _post_json(api_key, payload)
    return strip_translation_preamble(_message_text(response)), response.get("usage", {})


def strip_translation_preamble(text: str) -> str:
    text = text.strip()
    if text.lower().startswith(("here is ", "below is ", "the translation")):
        divider = re.search(r"\n\s*---+\s*\n", text)
        if divider:
            text = text[divider.end() :]
        else:
            lines = text.splitlines()
            if len(lines) > 1:
                text = "\n".join(lines[1:]).lstrip()
    return text.strip()


def parse_pages(spec: str, page_count: int) -> list[int]:
    values: set[int] = set()
    for item in re.split(r"[,，\s]+", spec.strip()):
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    pages = sorted(values)
    if not pages:
        raise ValueError("至少选择一页。")
    invalid = [p for p in pages if p < 1 or p > page_count]
    if invalid:
        raise ValueError(f"页码超出范围 1-{page_count}: {invalid}")
    return pages


def _poppler_dir() -> Path:
    override = os.environ.get("POPPLER_BIN")
    if override:
        return Path(override)
    return Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin"


def render_page(input_pdf: Path, page_number: int, output_jpg: Path) -> None:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    poppler = _poppler_dir()
    executable = poppler / "pdftoppm.exe"
    if not executable.exists():
        executable = Path(shutil.which("pdftoppm") or "")
    if not executable.exists():
        raise FileNotFoundError("找不到 pdftoppm；请设置 POPPLER_BIN。")
    prefix = output_jpg.with_suffix("")
    env = os.environ.copy()
    env["PATH"] = str(poppler) + os.pathsep + env.get("PATH", "")
    subprocess.run(
        [
            str(executable),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-jpeg",
            "-jpegopt",
            "quality=90",
            "-scale-to",
            "2000",
            str(input_pdf),
            str(prefix),
        ],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def markdown_blocks(markdown: str) -> list[tuple[str, str]]:
    markdown = strip_code_fence(markdown)
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = " ".join(part.strip() for part in buffer if part.strip())
            text = display_text(text)
            if text:
                blocks.append(("body", text))
            buffer.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("#"):
            flush()
            level = min(3, len(line) - len(line.lstrip("#")))
            blocks.append((f"h{level}", display_text(line.lstrip("#").strip())))
        elif re.match(r"^(?:[-*+] |\d+[.)] )", line):
            flush()
            blocks.append(("bullet", display_text(line)))
        else:
            buffer.append(line)
    flush()
    return blocks


def _braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _replace_fractions(text: str) -> str:
    marker = "\\frac"
    while marker in text:
        start = text.find(marker)
        first = _braced(text, start + len(marker))
        if not first:
            text = text[:start] + "/" + text[start + len(marker) :]
            continue
        numerator, after_first = first
        second = _braced(text, after_first)
        if not second:
            text = text[:start] + numerator + "/" + text[after_first:]
            continue
        denominator, after_second = second
        replacement = f"({_replace_fractions(numerator)})/({_replace_fractions(denominator)})"
        text = text[:start] + replacement + text[after_second:]
    return text


def display_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = _replace_fractions(text)
    for command in ("mathrm", "text", "operatorname"):
        pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")
        previous = None
        while previous != text:
            previous = text
            text = pattern.sub(r"\1", text)
    replacements = {
        r"\[": "",
        r"\]": "",
        r"\(": "",
        r"\)": "",
        r"\,": "",
        r"\;": " ",
        r"\quad": "  ",
        r"\log": "log",
        r"\times": "×",
        r"\cdot": "·",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\approx": "≈",
        r"\Omega": "Ω",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("$", "")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return re.sub(r"[ \t]{3,}", "  ", text).strip()


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and pdfmetrics.stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _style(kind: str, base_size: float) -> tuple[float, float, float]:
    if kind == "h1":
        size = base_size * 1.55
        return size, size * 1.35, base_size * 0.8
    if kind == "h2":
        size = base_size * 1.3
        return size, size * 1.35, base_size * 0.6
    if kind == "h3":
        size = base_size * 1.15
        return size, size * 1.35, base_size * 0.45
    return base_size, base_size * 1.45, base_size * 0.55


def _layout_columns(blocks: list[tuple[str, str]], width: float, height: float, base_size: float) -> tuple[bool, list[tuple[int, float, float, float, str]]]:
    margin_x = 34.0
    top = height - 50.0
    bottom = 34.0
    gap = 22.0
    column_width = (width - 2 * margin_x - gap) / 2
    column = 0
    y = top
    placements: list[tuple[int, float, float, float, str]] = []
    for kind, text in blocks:
        size, leading, after = _style(kind, base_size)
        lines = _wrap(text, "STSong-Light", size, column_width)
        needed = len(lines) * leading + after
        if y - needed < bottom:
            column += 1
            y = top
        if column > 1 or y - needed < bottom:
            return False, []
        x = margin_x + column * (column_width + gap)
        for line in lines:
            placements.append((column, x, y, size, line))
            y -= leading
        y -= after
    return True, placements


def create_translation_page(markdown: str, width: float, height: float, output_pdf: Path, title: str) -> None:
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    blocks = markdown_blocks(markdown)
    placements: list[tuple[int, float, float, float, str]] = []
    chosen_size = 6.0
    for size in (10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0):
        fits, candidate = _layout_columns(blocks, width, height, size)
        if fits:
            chosen_size = size
            placements = candidate
            break
    if not placements and blocks:
        raise RuntimeError("译文无法在两栏页面内排下；请缩小所选内容或调整渲染器。")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=(width, height))
    c.setFillColor(HexColor("#172033"))
    c.setFont("STSong-Light", 7.5)
    c.drawString(34, height - 24, title)
    c.drawRightString(width - 34, height - 24, f"自动排版 · 基准字号 {chosen_size:g} pt")
    c.setStrokeColor(HexColor("#D6DBE5"))
    c.line(width / 2, 34, width / 2, height - 44)
    c.setFillColor(HexColor("#111827"))
    for _column, x, y, size, line in placements:
        c.setFont("STSong-Light", size)
        c.drawString(x, y, line)
    c.showPage()
    c.save()


def create_comparison_pdf(
    input_pdf: Path,
    pages: list[int],
    translations: list[str],
    profile: Profile,
    job_dir: Path,
    output_pdf: Path,
) -> None:
    source = PdfReader(str(input_pdf))
    writer = PdfWriter()
    for page_number, markdown in zip(pages, translations, strict=True):
        original = source.pages[page_number - 1]
        width = float(original.mediabox.width)
        height = float(original.mediabox.height)
        translation_pdf = job_dir / f"page-{page_number:03d}-{profile.slug}-typeset.pdf"
        create_translation_page(
            markdown,
            width,
            height,
            translation_pdf,
            f"{profile.label} · 原 PDF 第 {page_number} 页中文译文",
        )
        translated = PdfReader(str(translation_pdf)).pages[0]
        comparison = writer.add_blank_page(width=width * 2, height=height)
        comparison.merge_page(original)
        comparison.merge_transformed_page(translated, Transformation().translate(tx=width, ty=0))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def run_pipeline(
    input_pdf: Path,
    page_spec: str,
    api_key: str,
    profile_slugs: Iterable[str],
    output_dir: Path,
    work_root: Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Path], Path]:
    if not api_key.strip():
        raise ValueError("请输入硅基流动 API Key。")
    input_pdf = Path(input_pdf)
    page_count = len(PdfReader(str(input_pdf)).pages)
    pages = parse_pages(page_spec, page_count)
    slugs = list(dict.fromkeys(profile_slugs))
    if not slugs:
        raise ValueError("至少选择一个输出版本。")
    unknown = [slug for slug in slugs if slug not in PROFILES]
    if unknown:
        raise ValueError(f"未知版本: {unknown}")

    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = work_root / job_id
    image_dir = job_dir / "images"
    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        if progress:
            progress(message)

    images: dict[int, Path] = {}
    usage: list[dict] = []
    for index, page_number in enumerate(pages, start=1):
        report(f"渲染第 {page_number} 页（{index}/{len(pages)}）")
        image_path = image_dir / f"page-{page_number:03d}.jpg"
        render_page(input_pdf, page_number, image_path)
        images[page_number] = image_path

    outputs: list[Path] = []
    ocr_cache: dict[tuple[str, int], str] = {}
    for slug in slugs:
        profile = PROFILES[slug]
        translations: list[str] = []
        for index, page_number in enumerate(pages, start=1):
            cache_key = (profile.ocr_model, page_number)
            if cache_key in ocr_cache:
                report(f"复用 {profile.ocr_model} 第 {page_number} 页 OCR")
                source_text = ocr_cache[cache_key]
            else:
                report(f"{profile.ocr_model} 识别第 {page_number} 页（{index}/{len(pages)}）")
                source_text, token_usage = ocr_page(api_key, images[page_number], profile.ocr_model)
                ocr_cache[cache_key] = source_text
                usage.append({"page": page_number, "stage": f"ocr_{slug}", "model": profile.ocr_model, "usage": token_usage})
            (job_dir / f"page-{page_number:03d}-{slug}-ocr.md").write_text(source_text, encoding="utf-8")
            if profile.reviewer_model:
                report(f"{profile.reviewer_model} 复核第 {page_number} 页 OCR（{index}/{len(pages)}）")
                source_text, token_usage = review_ocr(
                    api_key,
                    images[page_number],
                    source_text,
                    profile.reviewer_model,
                )
                (job_dir / f"page-{page_number:03d}-ocr-reviewed.md").write_text(source_text, encoding="utf-8")
                usage.append({"page": page_number, "stage": "ocr_review", "model": profile.reviewer_model, "usage": token_usage})
            report(f"{profile.translation_model} 翻译第 {page_number} 页（{index}/{len(pages)}）")
            translated, token_usage = translate_page(api_key, source_text, profile.translation_model)
            translations.append(translated)
            (job_dir / f"page-{page_number:03d}-{slug}-zh.md").write_text(translated, encoding="utf-8")
            usage.append({"page": page_number, "stage": f"translate_{slug}", "model": profile.translation_model, "usage": token_usage})

        page_label = "_".join(str(p) for p in pages)
        output_pdf = output_dir / f"RF_Circuit_Design_pages_{page_label}_{slug}_zh.pdf"
        report(f"生成 {profile.label} PDF")
        create_comparison_pdf(input_pdf, pages, translations, profile, job_dir, output_pdf)
        outputs.append(output_pdf)

    manifest = {
        "job_id": job_id,
        "input_pdf": str(input_pdf),
        "pages": pages,
        "profiles": [asdict(PROFILES[slug]) for slug in slugs],
        "outputs": [str(path) for path in outputs],
        "usage": usage,
    }
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report("完成")
    return outputs, manifest_path
