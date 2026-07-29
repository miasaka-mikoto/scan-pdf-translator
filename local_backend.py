from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def ollama_translate(markdown: str, model: str = "qwen3:8b") -> str:
    """Local translation through the Ollama daemon; no document text leaves the machine."""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": "将英文工程教材Markdown忠实翻译为简体中文。保留公式、变量、单位、数值、图号和式号。只输出译文。",
            },
            {"role": "user", "content": markdown},
        ],
    }
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["message"]["content"].strip()


def paddleocr_markdown(image_path: Path, output_dir: Path) -> tuple[str, dict]:
    """Invoke the official local PaddleOCR-VL API. Models download on first use."""
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError("未安装 PaddleOCR-VL。请先执行 local-bootstrap\\install-and-enable-local.ps1") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PaddleOCRVL(pipeline_version=os.environ.get("LOCAL_PADDLEOCR_VL_VERSION", "v1.5"))
    results = list(pipeline.predict(str(image_path)))
    if not results:
        raise RuntimeError("PaddleOCR-VL 未返回结果")
    result = results[0]
    result.save_to_json(save_path=output_dir)
    result.save_to_markdown(save_path=output_dir)
    markdown_files = sorted(output_dir.glob("*.md"))
    if not markdown_files:
        raise RuntimeError("PaddleOCR-VL 未生成 Markdown")
    raw_json = result.json if hasattr(result, "json") else {}
    if callable(raw_json):
        raw_json = raw_json()
    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except json.JSONDecodeError:
            raw_json = {"raw_result": raw_json}
    return markdown_files[0].read_text(encoding="utf-8"), raw_json


def local_health() -> dict:
    request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        return {"ollama": True, "models": [model.get("name") for model in models]}
    except OSError:
        return {"ollama": False, "models": []}


def paddle_result_regions(raw: Any, image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Best-effort adapter for PaddleOCR-VL JSON variants.

    Paddle has changed the enclosing keys between releases, but layout blocks
    consistently expose a four-number bbox plus text/content.  This keeps the
    local backend on the same normalized coordinate contract as cloud OCR.
    """
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            bbox = next((value.get(key) for key in ("bbox", "block_bbox", "coordinate", "rec_box") if value.get(key) is not None), None)
            text = next((value.get(key) for key in ("content", "text", "block_content", "rec_text", "markdown") if isinstance(value.get(key), str) and value.get(key).strip()), "")
            label = str(next((value.get(key) for key in ("label", "block_label", "type", "block_type") if value.get(key)), "text")).lower()
            visual_without_text = any(word in label for word in ("image", "chart", "figure", "picture", "circuit", "schematic", "table")) and not any(word in label for word in ("title", "caption"))
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and (text or visual_without_text):
                numbers = list(bbox[:4])
                if all(isinstance(item, (int, float)) for item in numbers):
                    x0, y0, x1, y1 = map(float, numbers)
                    if x1 > x0 and y1 > y0:
                        if any(word in label for word in ("formula", "equation", "math")):
                            kind = "equation"
                        elif "table" in label:
                            kind = "table"
                        elif any(word in label for word in ("circuit", "schematic")):
                            kind = "circuit"
                        elif "chart" in label:
                            kind = "chart"
                        elif any(word in label for word in ("figure_title", "caption")):
                            kind = "image_caption"
                        elif any(word in label for word in ("figure", "image", "picture")):
                            kind = "image"
                        elif any(word in label for word in ("title", "heading")):
                            kind = "title"
                        else:
                            kind = "text"
                        found.append({"bbox": [round(x0 / image_width * 1000), round(y0 / image_height * 1000), round(x1 / image_width * 1000), round(y1 / image_height * 1000)], "text": text, "type": label, "kind": kind})
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(raw)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in found:
        key = tuple(item["bbox"]) + (item["text"],)
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["bbox"][1], item["bbox"][0]))
