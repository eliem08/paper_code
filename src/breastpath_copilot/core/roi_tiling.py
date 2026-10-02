"""Native-resolution tiling of pathology RoIs into patches.

BRACS RoIs range from small crops to >100-megapixel regions. Resizing a whole
RoI to 224x224 destroys the cellular detail that fine-grained subtyping
(UDH / ADH / FEA / DCIS) depends on. This module tiles an RoI into
native-resolution ``patch_size`` patches, keeps the tissue-bearing ones, and
caps the count -- the standard multi-instance recipe for pathology.

A small RoI (smaller than ``patch_size``) falls back to a single resized
whole-image patch so every RoI yields at least one patch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # trusted local pathology files can be huge


def _is_tissue(patch: np.ndarray, max_mean_intensity: float = 220.0, min_std: float = 8.0) -> bool:
    """Heuristic: a patch is tissue if it isn't mostly white background.

    H&E background is bright/uniform; stained tissue is darker and textured.
    """
    gray = patch.mean(axis=2)
    return bool(gray.mean() < max_mean_intensity and gray.std() > min_std)


def tile_roi_image(
    image: Image.Image,
    *,
    patch_size: int = 224,
    max_patches: int = 16,
    stride: int | None = None,
    tissue_only: bool = True,
    seed: int = 0,
) -> list[Image.Image]:
    """Tile a PIL RoI into up to ``max_patches`` native-resolution patches.

    Returns RGB PIL patches of ``patch_size``. Grid tiling with optional
    tissue filtering; if more tissue patches than ``max_patches``, a
    deterministic spread-out subset is taken. Small RoIs -> one resized patch.
    """
    image = image.convert("RGB")
    W, H = image.size
    stride = stride or patch_size

    if W < patch_size or H < patch_size:
        return [image.resize((patch_size, patch_size), Image.BICUBIC)]

    arr = np.asarray(image, dtype=np.uint8)
    coords: list[tuple[int, int]] = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            coords.append((x, y))
    if not coords:
        return [image.resize((patch_size, patch_size), Image.BICUBIC)]

    kept: list[tuple[int, int]] = []
    for (x, y) in coords:
        patch = arr[y : y + patch_size, x : x + patch_size]
        if (not tissue_only) or _is_tissue(patch):
            kept.append((x, y))
    if not kept:
        # No tissue found by the heuristic -> use the densest (darkest) patch
        # plus a resized whole-image fallback so we never return nothing.
        darkest = min(coords, key=lambda c: arr[c[1]:c[1]+patch_size, c[0]:c[0]+patch_size].mean())
        kept = [darkest]

    if len(kept) > max_patches:
        # Deterministic, spread-out subsample (evenly spaced indices).
        idx = np.linspace(0, len(kept) - 1, max_patches).round().astype(int)
        kept = [kept[i] for i in idx]

    return [image.crop((x, y, x + patch_size, y + patch_size)) for (x, y) in kept]


def tile_roi_path(
    path: str | Path,
    *,
    patch_size: int = 224,
    max_patches: int = 16,
    stride: int | None = None,
    tissue_only: bool = True,
) -> list[Image.Image]:
    return tile_roi_image(
        Image.open(path),
        patch_size=patch_size,
        max_patches=max_patches,
        stride=stride,
        tissue_only=tissue_only,
    )


def tile_roi_with_coords(
    image: Image.Image,
    *,
    patch_size: int = 224,
    max_patches: int = 36,
    stride: int | None = None,
    tissue_only: bool = True,
) -> list[tuple[Image.Image, tuple[int, int]]]:
    """Like tile_roi_image but returns (patch, (x_topleft, y_topleft)) pairs.

    Coordinates are the patch top-left in the RoI's native pixel space — used
    to place patch/nucleus nodes in a spatial graph.
    """
    image = image.convert("RGB")
    W, H = image.size
    stride = stride or patch_size
    if W < patch_size or H < patch_size:
        return [(image.resize((patch_size, patch_size), Image.BICUBIC), (0, 0))]
    arr = np.asarray(image, dtype=np.uint8)
    coords: list[tuple[int, int]] = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            coords.append((x, y))
    if not coords:
        return [(image.resize((patch_size, patch_size), Image.BICUBIC), (0, 0))]
    kept = [(x, y) for (x, y) in coords if (not tissue_only) or _is_tissue(arr[y:y + patch_size, x:x + patch_size])]
    if not kept:
        kept = [min(coords, key=lambda c: arr[c[1]:c[1] + patch_size, c[0]:c[0] + patch_size].mean())]
    if len(kept) > max_patches:
        idx = np.linspace(0, len(kept) - 1, max_patches).round().astype(int)
        kept = [kept[i] for i in idx]
    return [(image.crop((x, y, x + patch_size, y + patch_size)), (x, y)) for (x, y) in kept]
