from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from job_store import JobStore
from rich_renderer import (
    Region,
    create_chinese_only_pdf,
    create_reconstructed_page,
    create_side_by_side_pdf,
    create_vertical_dual_pdf,
    parse_grounded_markdown,
    translate_regions,
)
from translator import QUALITY_PROFILE, _poppler_dir, parse_pages, render_page, review_ocr


@dataclass
class JobSpec:
    input_pdf: Path
    page_spec: str
    api_key: str
    source_mode: str = "siliconflow"
    quality: str = "balanced"
    outputs: list[str] = field(default_factory=lambda: ["bilingual_vertical"])
    formula_strategy: str = "mathjax"
    figure_strategy: str = "preserve"
    glossary: str = ""
    auto_repair_missing: bool = True
    repair_pages: str = ""


def _json_regions(regions: list[Region]) -> list[dict[str, Any]]:
    return [asdict(region) for region in regions]


def _regions_from_json(items: list[dict[str, Any]]) -> list[Region]:
    return [Region(id=x["id"], kind=x["kind"], bbox=tuple(x["bbox"]), source=x["source"], translated=x.get("translated", ""), confidence=x.get("confidence"), fallback_reason=x.get("fallback_reason", ""), status=x.get("status", "pending")) for x in items]


def _model_for_quality(quality: str, source_mode: str = "siliconflow") -> str:
    if source_mode == "siliconflow_free":
        return "Qwen/Qwen3-8B"
    return "deepseek-ai/DeepSeek-V4-Pro" if quality == "high" else "deepseek-ai/DeepSeek-V4-Flash"


def _classify_pdf(reader: PdfReader) -> str:
    sample = min(5, len(reader.pages))
    chars = sum(len((reader.pages[index].extract_text() or "").strip()) for index in range(sample))
    return "scanned" if chars < 100 * sample else "digital_or_hybrid"


def _all_pages(spec: str, count: int) -> list[int]:
    if spec.strip() in {"全部", "all", "ALL", "*"}:
        return list(range(1, count + 1))
    return parse_pages(spec, count)


def _page_label(pages: list[int]) -> str:
    """Compact, Windows-safe label for output paths."""
    if not pages:
        return "pages_none"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages_{pages[0]}-{pages[-1]}"
    preview = "_".join(str(page) for page in pages[:8])
    suffix = f"_plus_{len(pages) - 8}" if len(pages) > 8 else ""
    return f"pages_{preview}{suffix}"


def estimate_translation_cost_yuan(page_count: int, quality: str, source_mode: str = "siliconflow") -> float:
    """Conservative translation-only estimate; OCR billing is reported after use.

    Price basis: SiliconFlow public pricing checked 2026-07-27.  A complex
    technical page is budgeted as 1,200 input + 800 output tokens.
    """
    if source_mode == "siliconflow_free":
        return 0.0
    input_price, output_price = (12.0, 24.0) if quality == "high" else (1.0, 2.0)
    return round(page_count * (1200 * input_price + 800 * output_price) / 1_000_000, 4)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _usage_totals(page_dirs: list[Path]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for page_dir in page_dirs:
        for name in ("ocr-usage.json", "ocr-review-usage.json", "translation-usage.json"):
            path = page_dir / name
            if not path.exists():
                continue
            try:
                usage = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in totals:
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    return totals


def _render_output_for_qa(pdf_path: Path, qa_dir: Path) -> bool:
    poppler = _poppler_dir()
    executable = poppler / "pdftoppm.exe"
    if not executable.exists():
        return False
    qa_dir.mkdir(parents=True, exist_ok=True)
    prefix = qa_dir / pdf_path.stem
    env = os.environ.copy()
    env["PATH"] = str(poppler) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(executable), "-png", "-scale-to", "1400", str(pdf_path), str(prefix)], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return any(qa_dir.glob(f"{pdf_path.stem}-*.png"))


def _qa_page(regions: list[Region]) -> dict[str, Any]:
    source = "\n".join(region.source for region in regions if region.source)
    translated = "\n".join(region.translated for region in regions if region.translated)
    unit_pattern = r"\b(?:\d+(?:\.\d+)?\s*)?(?:dB|MHz|GHz|kHz|pF|nF|uF|V|mA|Ω)\b"
    source_units = re.findall(unit_pattern, source)
    translated_units = re.findall(unit_pattern, translated)
    number_pattern = r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?[+-]?\d+)?"
    source_numbers = re.findall(number_pattern, source)
    translated_numbers = re.findall(number_pattern, translated)
    equation_source = "\n".join(region.source for region in regions if region.kind == "equation")
    equation_translated = "\n".join(region.translated for region in regions if region.kind == "equation")
    variable_pattern = r"\b(?:[A-Za-z]+_[A-Za-z0-9]+|[VIZRCLXYG][A-Za-z0-9]+|[A-Z]{1,3}\d*)\b"
    ignored_variable_words = {"Eq", "Fig", "The", "This", "That", "Loss", "Keep", "Let", "Thus", "For", "And", "With", "From"}
    source_variables = sorted(set(item for item in re.findall(variable_pattern, equation_source) if item not in ignored_variable_words))
    translated_variables = set(re.findall(variable_pattern, equation_translated))
    equations = [region for region in regions if region.kind == "equation"]
    images = [region for region in regions if region.kind in {"image", "table", "chart", "circuit"}]
    figure_caption_only = [region.id for region in images if (region.bbox[3] - region.bbox[1]) < 35]
    missing = [region.id for region in regions if region.kind in {"text", "sub_title", "title", "image_caption", "equation"} and region.source and not region.translated]
    unit_count_match = len(source_units) == len(translated_units)
    missing_variables = [item for item in source_variables if item not in translated_variables]
    suspicious_markdown = [region.id for region in regions if "```" in region.translated or "<think" in region.translated.lower()]
    # This is intentionally conservative: the renderer reduces font size before
    # drawing, so flag only text far beyond a realistic minimum-font capacity.
    overflow_risk = [region.id for region in regions if region.kind in {"text", "sub_title", "title", "image_caption"} and region.translated and len(region.translated) > max(160, int((region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) / 40))]
    review_reasons: list[str] = []
    if missing: review_reasons.append("missing_translations")
    if not unit_count_match: review_reasons.append("unit_count_mismatch")
    if suspicious_markdown: review_reasons.append("model_artifact")
    if overflow_risk: review_reasons.append("text_overflow_risk")
    if figure_caption_only: review_reasons.append("figure_detection_incomplete")
    return {
        "regions": len(regions),
        "equations": len(equations),
        "figures": len(images),
        "missing_translations": missing,
        "source_units": len(source_units),
        "translated_units": len(translated_units),
        "unit_count_match": unit_count_match,
        "source_numbers": len(source_numbers),
        "translated_numbers": len(translated_numbers),
        "number_count_match": len(source_numbers) == len(translated_numbers),
        "unpreserved_variables": missing_variables,
        "model_artifact_regions": suspicious_markdown,
        "text_overflow_risk": overflow_risk,
        "figure_caption_only_regions": figure_caption_only,
        "review_reasons": review_reasons,
        "formula_svg": sum(region.status == "formula_svg" for region in equations),
        "formula_fallbacks": sum(bool(region.fallback_reason) for region in equations),
        "figure_source_preserved": sum(region.status == "source_preserved" for region in images),
        "figure_vector_traced": sum(region.status == "figure_vector_traced" for region in images),
        "fallback_regions": [
            {"id": region.id, "type": region.kind, "reason": region.fallback_reason}
            for region in regions if region.fallback_reason
        ],
        "status": "needs_review" if review_reasons else "passed",
    }


def post_generation_translation_audit(qa_pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Final deterministic audit for omissions before an output is delivered.

    This does not invent a replacement translation. It makes every source
    block with an empty target visible in the report, together with pages that
    need review for unit/number/overflow checks.
    """
    missing_blocks = [
        {"page": item.get("page"), "regions": item.get("missing_translations", [])}
        for item in qa_pages
        if item.get("missing_translations")
    ]
    review_pages = [
        {"page": item.get("page"), "reasons": item.get("review_reasons", [])}
        for item in qa_pages
        if item.get("review_reasons")
    ]
    source_regions = sum(int(item.get("regions", 0)) for item in qa_pages)
    return {
        "method": "source_region_to_target_region_completeness",
        "checked_pages": len(qa_pages),
        "checked_regions": source_regions,
        "missing_translation_pages": missing_blocks,
        "missing_translation_count": sum(len(item["regions"]) for item in missing_blocks),
        "review_pages": review_pages,
        "delivery_gate": "passed" if not missing_blocks else "blocked_missing_translations",
    }


def repair_missing_translations(api_key: str, regions: list[Region], model: str, glossary: str = "") -> tuple[list[Region], dict[str, int]]:
    """Retry only genuinely empty source blocks before a page is rendered."""
    eligible = {"text", "sub_title", "title", "image_caption", "equation"}
    missing = [region for region in regions if region.kind in eligible and region.source.strip() and not region.translated.strip()]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "repaired_missing_regions": 0}
    if not missing:
        return regions, usage
    repaired, retry_usage = translate_regions(api_key, missing, model, glossary)
    by_id = {region.id: region for region in repaired}
    for region in regions:
        replacement = by_id.get(region.id)
        if replacement and replacement.translated.strip():
            region.translated = replacement.translated
            region.status = "translated_repair"
            usage["repaired_missing_regions"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = retry_usage.get(key, 0)
        if isinstance(value, (int, float)):
            usage[key] = int(value)
    return regions, usage


def run_cloud_job(spec: JobSpec, store: JobStore, output_dir: Path) -> tuple[list[Path], Path, Path]:
    """Cloud processing. All semantic work is delegated to SiliconFlow; this function only orchestrates."""
    if spec.source_mode not in {"siliconflow", "siliconflow_free", "auto"}:
        raise RuntimeError("当前云端任务只接受 siliconflow、siliconflow_free 或 auto 模式。")
    if not spec.api_key.strip():
        raise ValueError("请输入硅基流动 API Key。")
    reader = PdfReader(str(spec.input_pdf))
    total_pdf_pages = len(reader.pages)
    pages = _all_pages(spec.page_spec, total_pdf_pages)
    repair_targets = set(_all_pages(spec.repair_pages, total_pdf_pages)) if spec.repair_pages.strip() else set()
    pages = sorted(set(pages) | repair_targets)
    preflight = {
        "pages": total_pdf_pages,
        "selected_pages": pages,
        "encrypted": reader.is_encrypted,
        "kind": _classify_pdf(reader),
        "page_size": [float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)],
        "estimated_cloud_pages": len(pages),
        "estimated_seconds": len(pages) * (45 if spec.quality == "high" else 20),
        "estimated_translation_yuan": estimate_translation_cost_yuan(len(pages), spec.quality, spec.source_mode),
        "cost_note": "免费模式当前使用官方标价为免费的 DeepSeek-OCR 与 Qwen3-8B；价格和限流以硅基流动实时页面为准。" if spec.source_mode == "siliconflow_free" else "仅为翻译 token 估算；视觉 OCR 按实际账单计费。",
    }
    _write_json(store.root / "preflight.json", preflight)
    store.save_status(state="running", preflight=preflight, started_at=time.time())
    store.emit("preflight", "PDF 预检完成", 0.03, total_pages=len(pages), **preflight)

    rebuilt_pages: list[Path] = []
    qa_pages: list[dict[str, Any]] = []
    total = len(pages)
    for index, page_number in enumerate(pages, start=1):
        store.control.checkpoint()
        page_dir = store.page_dir(page_number)
        progress_base = 0.05 + (index - 1) / total * 0.8
        image_path = page_dir / "source.jpg"
        grounded_path = page_dir / "grounded.md"
        regions_path = page_dir / "regions.json"
        rebuilt_path = page_dir / "rebuilt.pdf"

        # Explicit repair reuses source rendering and OCR cache, but replaces
        # the translated layout and page PDF for exactly the requested pages.
        if page_number in repair_targets:
            for stale in (regions_path, rebuilt_path, page_dir / "qa.json", page_dir / "translation-usage.json"):
                stale.unlink(missing_ok=True)
            store.emit("repair_page", f"指定修复第 {page_number} 页（复用 OCR）", progress_base, page_number, total)

        if not image_path.exists():
            store.emit("render_page", f"渲染第 {page_number} 页", progress_base, page_number, total)
            render_page(spec.input_pdf, page_number, image_path)

        if not grounded_path.exists():
            store.emit("ocr_page", f"硅基流动 OCR 第 {page_number} 页", progress_base + 0.08 / total, page_number, total)
            from translator import ocr_page
            grounded, usage = ocr_page(spec.api_key, image_path, "deepseek-ai/DeepSeek-OCR")
            grounded_path.write_text(grounded, encoding="utf-8")
            _write_json(page_dir / "ocr-usage.json", usage)
        grounded = grounded_path.read_text(encoding="utf-8")

        if spec.quality == "high" and spec.source_mode != "siliconflow_free" and not (page_dir / "ocr-review.md").exists():
            store.emit("ocr_review", f"高质量 OCR 复核第 {page_number} 页", progress_base + 0.12 / total, page_number, total)
            reviewed, usage = review_ocr(spec.api_key, image_path, grounded, QUALITY_PROFILE.reviewer_model or "Qwen/Qwen3-VL-32B-Thinking")
            (page_dir / "ocr-review.md").write_text(reviewed, encoding="utf-8")
            _write_json(page_dir / "ocr-review-usage.json", usage)
        # The high-quality reviewer is authoritative for downstream layout and
        # translation, while the first-pass grounded output remains cached for
        # audit and safe retry.
        review_path = page_dir / "ocr-review.md"
        if spec.quality == "high" and spec.source_mode != "siliconflow_free" and review_path.exists():
            reviewed_grounded = review_path.read_text(encoding="utf-8")
            if "<|det|>" in reviewed_grounded:
                grounded = reviewed_grounded
            else:
                store.emit("ocr_review", f"第 {page_number} 页复核未保留坐标，安全回退首轮定位", progress_base + 0.13 / total, page_number, total, reviewer_grounding_preserved=False)

        if regions_path.exists():
            regions = _regions_from_json(json.loads(regions_path.read_text(encoding="utf-8")))
        else:
            store.emit("layout_page", f"解析第 {page_number} 页版面区域", progress_base + 0.15 / total, page_number, total)
            regions = parse_grounded_markdown(grounded)
            if not regions:
                raise RuntimeError(f"第 {page_number} 页未获得带坐标的 OCR 区域")
            store.emit("translate_block", f"翻译第 {page_number} 页 {len(regions)} 个区域", progress_base + 0.22 / total, page_number, total, regions=len(regions))
            regions, usage = translate_regions(spec.api_key, regions, _model_for_quality(spec.quality, spec.source_mode), spec.glossary)
            if spec.auto_repair_missing:
                regions, repair_usage = repair_missing_translations(spec.api_key, regions, _model_for_quality(spec.quality, spec.source_mode), spec.glossary)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[key] = int(usage.get(key, 0)) + int(repair_usage.get(key, 0))
                usage["repaired_missing_regions"] = repair_usage["repaired_missing_regions"]
            _write_json(page_dir / "translation-usage.json", usage)
            _write_json(regions_path, _json_regions(regions))

        if not rebuilt_path.exists():
            store.emit("redraw_formula", f"公式 SVG 重绘/回退第 {page_number} 页", progress_base + 0.35 / total, page_number, total)
            store.emit("redraw_figure", f"图表确定性保留第 {page_number} 页", progress_base + 0.42 / total, page_number, total)
            original = reader.pages[page_number - 1]
            create_reconstructed_page(
                image_path,
                regions,
                float(original.mediabox.width),
                float(original.mediabox.height),
                rebuilt_path,
                "中文图文重构页",
                formula_strategy=spec.formula_strategy,
                figure_strategy=spec.figure_strategy,
                assets_dir=page_dir / "assets",
            )
            _write_json(regions_path, _json_regions(regions))
        qa = _qa_page(regions)
        qa["page"] = page_number
        qa_pages.append(qa)
        _write_json(page_dir / "qa.json", qa)
        rebuilt_pages.append(rebuilt_path)
        store.save_page_state(page_number, stage="completed", rebuilt=str(rebuilt_path), qa=qa)
        event_qa = {key: value for key, value in qa.items() if key != "page"}
        store.emit("qa_page", f"第 {page_number} 页 QA 完成", progress_base + 0.7 / total, page_number, total, **event_qa)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    prefix = f"{spec.input_pdf.stem}_{_page_label(pages)}"
    if "bilingual_vertical" in spec.outputs:
        target = output_dir / f"{prefix}_bilingual_rich_zh.pdf"
        # Multi-page vertical output requires a single writer, so build each pair and merge.
        pair_paths: list[Path] = []
        for page_number, rebuilt in zip(pages, rebuilt_pages, strict=True):
            pair = store.root / f"pair-{page_number:04d}.pdf"
            create_vertical_dual_pdf(spec.input_pdf, page_number, rebuilt, pair)
            pair_paths.append(pair)
        from pypdf import PdfWriter
        writer = PdfWriter()
        for page_number, pair in zip(pages, pair_paths, strict=True):
            writer.add_page(PdfReader(str(pair)).pages[0])
            writer.add_outline_item(f"原 PDF 第 {page_number} 页", len(writer.pages) - 1)
        with target.open("wb") as stream:
            writer.write(stream)
        outputs.append(target)
    if "bilingual_side_by_side" in spec.outputs:
        target = output_dir / f"{prefix}_bilingual_side_by_side_zh.pdf"
        create_side_by_side_pdf(spec.input_pdf, pages, rebuilt_pages, target)
        outputs.append(target)
    if "chinese_only" in spec.outputs:
        target = output_dir / f"{prefix}_chinese_only_zh.pdf"
        create_chinese_only_pdf(rebuilt_pages, target, pages)
        outputs.append(target)

    store.emit("merge_pdf", "合并输出 PDF", 0.94, total_pages=total)
    render_results = {str(path): _render_output_for_qa(path, store.root / "qa-renders") for path in outputs}
    translation_audit = post_generation_translation_audit(qa_pages)
    report = {
        "job_id": store.root.name,
        "preflight": preflight,
        "pages": qa_pages,
        "outputs": [str(path) for path in outputs],
        "pdf_render_checks": render_results,
        "post_generation_translation_audit": translation_audit,
        "summary": {
            "completed_pages": len(qa_pages),
            "failed_pages": len([page for page in qa_pages if page["status"] != "passed"]),
            "formula_regions": sum(page["equations"] for page in qa_pages),
            "figure_regions": sum(page["figures"] for page in qa_pages),
            "formula_fallbacks": sum(page["formula_fallbacks"] for page in qa_pages),
            "preserved_figures": sum(page["figure_source_preserved"] for page in qa_pages),
            "vector_traced_figures": sum(page["figure_vector_traced"] for page in qa_pages),
            "missing_translation_count": translation_audit["missing_translation_count"],
            "post_generation_delivery_gate": translation_audit["delivery_gate"],
            "token_usage": _usage_totals([store.page_dir(number) for number in pages]),
        },
    }
    report_json = store.root / "QA-report.json"
    _write_json(report_json, report)
    report_html = store.root / "QA-report.html"
    report_html.write_text("<html><meta charset='utf-8'><body><h1>PDF 翻译 QA 报告</h1><pre>" + json.dumps(report, ensure_ascii=False, indent=2) + "</pre></body></html>", encoding="utf-8")
    store.save_status(state="completed", outputs=[str(path) for path in outputs], report=str(report_json), completed_at=time.time())
    store.emit("completed", "任务完成", 1.0, total_pages=total)
    return outputs, report_json, report_html


def run_local_job(spec: JobSpec, store: JobStore, output_dir: Path) -> tuple[list[Path], Path, Path]:
    """Offline counterpart using installed PaddleOCR-VL and Ollama only."""
    from PIL import Image
    from local_backend import local_health, ollama_translate, paddle_result_regions, paddleocr_markdown

    health = local_health()
    if not health["ollama"]:
        raise RuntimeError("本地 Ollama 服务未运行。请先执行 local-bootstrap\\start-local.ps1")
    reader = PdfReader(str(spec.input_pdf))
    pages = _all_pages(spec.page_spec, len(reader.pages))
    store.save_status(state="running", started_at=time.time(), backend="local")
    store.emit("preflight", "本地模式预检完成", 0.03, total_pages=len(pages), local_models=health["models"])
    rebuilt: list[Path] = []
    qa_pages: list[dict[str, Any]] = []
    for index, page_number in enumerate(pages, 1):
        store.control.checkpoint()
        page_dir = store.page_dir(page_number)
        image_path = page_dir / "source.jpg"
        if not image_path.exists():
            store.emit("render_page", f"渲染第 {page_number} 页", 0.05 + index / len(pages) * .1, page_number, len(pages))
            render_page(spec.input_pdf, page_number, image_path)
        raw_path = page_dir / "local-ocr.json"
        region_path = page_dir / "regions.json"
        if region_path.exists():
            regions = _regions_from_json(json.loads(region_path.read_text(encoding="utf-8")))
            # Re-normalize cached Paddle output when available. Older adapters
            # discarded image/chart blocks whose content was empty, leaving
            # only their captions and producing blank reconstructed figures.
            if raw_path.exists():
                im = Image.open(image_path)
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                blocks = paddle_result_regions(raw, im.width, im.height)
                old_by_source = {region.source: region.translated for region in regions if region.source}
                normalized: list[Region] = []
                for n, block in enumerate(blocks, 1):
                    source = block["text"]
                    translated = old_by_source.get(source, "")
                    normalized.append(Region(f"r{n:03d}", block.get("kind", "text"), tuple(block["bbox"]), source, translated, confidence=None, status="translated" if translated else "ocr"))
                if normalized:
                    regions = normalized
                    _write_json(region_path, _json_regions(regions))
        else:
            store.emit("ocr_page", f"本地 PaddleOCR-VL 识别第 {page_number} 页", .15 + index / len(pages) * .25, page_number, len(pages))
            markdown, raw = paddleocr_markdown(image_path, page_dir / "local-ocr")
            _write_json(raw_path, raw)
            im = Image.open(image_path)
            blocks = paddle_result_regions(raw, im.width, im.height)
            regions = [Region(f"r{n:03d}", block.get("kind", "text"), tuple(block["bbox"]), block["text"], confidence=None, status="ocr") for n, block in enumerate(blocks, 1)]
            if not regions:
                # A real, visible error is safer than silently producing an empty
                # translated page if an incompatible PaddleOCR result schema is used.
                raise RuntimeError("本地 PaddleOCR-VL 未提供带坐标版面块；请更新扩展包后重试。")
            store.emit("translate_block", f"本地 Ollama 翻译第 {page_number} 页", .4 + index / len(pages) * .25, page_number, len(pages), regions=len(regions))
            for region in regions:
                if region.source:
                    region.translated = ollama_translate(region.source)
                    region.status = "translated"
            _write_json(region_path, _json_regions(regions))
        rebuilt_path = page_dir / "rebuilt.pdf"
        if not rebuilt_path.exists():
            original = reader.pages[page_number - 1]
            create_reconstructed_page(image_path, regions, float(original.mediabox.width), float(original.mediabox.height), rebuilt_path, "中文图文重构页", spec.formula_strategy, spec.figure_strategy, page_dir / "assets")
            _write_json(region_path, _json_regions(regions))
        qa = _qa_page(regions); qa["page"] = page_number; qa_pages.append(qa); rebuilt.append(rebuilt_path)
        _write_json(page_dir / "qa.json", qa); store.save_page_state(page_number, stage="completed", qa=qa, rebuilt=str(rebuilt_path))
        store.emit("qa_page", f"第 {page_number} 页本地 QA 完成", .75 + index / len(pages) * .15, page_number, len(pages), **{k:v for k,v in qa.items() if k != "page"})
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{spec.input_pdf.stem}_{_page_label(pages)}_local"
    outputs: list[Path] = []
    if "bilingual_vertical" in spec.outputs:
        from pypdf import PdfWriter
        target = output_dir / f"{prefix}_bilingual_zh.pdf"; writer = PdfWriter()
        for number, item in zip(pages, rebuilt, strict=True):
            pair = store.root / f"pair-{number:04d}.pdf"; create_vertical_dual_pdf(spec.input_pdf, number, item, pair); writer.add_page(PdfReader(str(pair)).pages[0])
        for number in pages:
            writer.add_outline_item(f"原 PDF 第 {number} 页", len(writer.pages) - len(pages) + pages.index(number))
        with target.open("wb") as handle: writer.write(handle)
        outputs.append(target)
    if "bilingual_side_by_side" in spec.outputs:
        target = output_dir / f"{prefix}_side_by_side_zh.pdf"; create_side_by_side_pdf(spec.input_pdf, pages, rebuilt, target); outputs.append(target)
    if "chinese_only" in spec.outputs:
        target = output_dir / f"{prefix}_chinese_only_zh.pdf"; create_chinese_only_pdf(rebuilt, target, pages); outputs.append(target)
    translation_audit = post_generation_translation_audit(qa_pages)
    report = {"job_id": store.root.name, "backend": "local", "pages": qa_pages, "outputs": [str(item) for item in outputs], "pdf_render_checks": {str(item): _render_output_for_qa(item, store.root / "qa-renders") for item in outputs}, "post_generation_translation_audit": translation_audit, "summary": {"missing_translation_count": translation_audit["missing_translation_count"], "post_generation_delivery_gate": translation_audit["delivery_gate"]}}
    report_json = store.root / "QA-report.json"; _write_json(report_json, report)
    report_html = store.root / "QA-report.html"; report_html.write_text("<meta charset='utf-8'><pre>" + json.dumps(report, ensure_ascii=False, indent=2) + "</pre>", encoding="utf-8")
    store.save_status(state="completed", outputs=[str(item) for item in outputs], report=str(report_json), completed_at=time.time())
    store.emit("completed", "本地任务完成", 1, total_pages=len(pages))
    return outputs, report_json, report_html
