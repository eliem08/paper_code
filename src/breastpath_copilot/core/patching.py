from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from breastpath_copilot.schemas import PatchRecord, SlideInput
from breastpath_copilot.utils.io import ensure_dir


def estimate_tissue_ratio(image: Image.Image, white_threshold: int = 220) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    non_white = np.any(arr < white_threshold, axis=-1)
    return float(non_white.mean())


def extract_grid_patches(
    slide: SlideInput,
    image: Image.Image,
    out_dir: str | Path,
    patch_size: int = 224,
    stride: int = 224,
    min_tissue_ratio: float = 0.35,
    max_patches: int = 500,
) -> list[PatchRecord]:
    out_dir = ensure_dir(out_dir)
    width, height = image.size
    records: list[PatchRecord] = []

    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            tissue_ratio = estimate_tissue_ratio(patch)
            if tissue_ratio < min_tissue_ratio:
                continue

            patch_id = f"{slide.slide_id}_x{x}_y{y}"
            patch_path = out_dir / f"{patch_id}.png"
            patch.save(patch_path)
            records.append(
                PatchRecord(
                    slide_id=slide.slide_id,
                    patch_id=patch_id,
                    x=x,
                    y=y,
                    width=patch_size,
                    height=patch_size,
                    tissue_ratio=tissue_ratio,
                    patch_path=str(patch_path),
                )
            )
            if len(records) >= max_patches:
                return records
    return records
