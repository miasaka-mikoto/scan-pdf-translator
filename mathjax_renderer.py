from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def render_formula_svg(latex: str, output_svg: Path, work_dir: Path) -> Path | None:
    """Render locally with MathJax SVG; return None for a safe source-crop fallback."""
    # MathJax's bundled SVG fonts do not reliably cover Chinese text inside
    # \mathrm/\text. Preserve the source formula in that case rather than
    # silently dropping explanatory glyphs.
    if any("\u4e00" <= char <= "\u9fff" for char in latex):
        return None
    node = shutil.which("node")
    script = Path(__file__).with_name("mathjax-render.mjs")
    module = Path(__file__).with_name("node_modules") / "mathjax-full"
    if not node or not script.exists() or not module.exists():
        return None
    output_svg = output_svg.resolve()
    work_dir.resolve().mkdir(parents=True, exist_ok=True)
    clean = latex.strip().replace("\\[", "").replace("\\]", "").replace("$$", "")
    payload = json.dumps({"latex": clean, "output": str(output_svg)})
    try:
        subprocess.run([node, str(script)], input=payload, text=True, cwd=Path(__file__).parent, check=True, timeout=20, capture_output=True)
        return output_svg if output_svg.exists() else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
