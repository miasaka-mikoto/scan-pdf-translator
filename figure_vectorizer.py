from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def vectorize_figure_svg(crop: Image.Image, output_svg: Path) -> Path | None:
    """Trace scanned black line art into deterministic SVG paths.

    This preserves the observed geometry. It does not infer, reconnect, or
    generate circuit elements and curves.
    """
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(1.12)
    gray = gray.filter(ImageFilter.SHARPEN)
    pixels = np.asarray(gray)
    _threshold, ink = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    subpaths: list[str] = []
    kept = 0
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if area < 0.8 and perimeter < 4:
            continue
        approximation = cv2.approxPolyDP(contour, max(0.35, perimeter * 0.0012), True)
        points = approximation.reshape(-1, 2)
        if len(points) < 2:
            continue
        commands = [f"M {int(points[0][0])} {int(points[0][1])}"]
        commands.extend(f"L {int(x)} {int(y)}" for x, y in points[1:])
        commands.append("Z")
        subpaths.append(" ".join(commands))
        kept += 1
    if kept == 0:
        return None

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    width, height = crop.size
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<path d="{" ".join(subpaths)}" fill="#111111" fill-rule="evenodd"/>'
        '</svg>'
    )
    output_svg.write_text(svg, encoding="utf-8")
    return output_svg
