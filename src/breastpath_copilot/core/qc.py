from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from breastpath_copilot.schemas import QCResult, SlideInput


def _laplacian_like_blur_score(image: Image.Image) -> float:
    gray = image.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edge, dtype=np.float32)
    return float(arr.var())


def run_qc(slide: SlideInput, preview_image: Image.Image) -> QCResult:
    arr = np.asarray(preview_image.convert("L"), dtype=np.float32)
    blur_score = _laplacian_like_blur_score(preview_image)
    mean_brightness = float(arr.mean())
    notes: list[str] = []
    passed = True

    if blur_score < 40:
        passed = False
        notes.append("Image may be too blurry for reliable patch extraction.")
    if mean_brightness < 30:
        passed = False
        notes.append("Image appears too dark.")
    if mean_brightness > 225:
        passed = False
        notes.append("Image appears too bright or overexposed.")

    return QCResult(
        slide_id=slide.slide_id,
        passed=passed,
        blur_score=blur_score,
        mean_brightness=mean_brightness,
        notes=notes,
    )
