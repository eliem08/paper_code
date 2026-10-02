"""Tests for native-resolution RoI tiling + multi-patch embedding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from breastpath_copilot.core.roi_tiling import _is_tissue, tile_roi_image, tile_roi_path


def _tissue_image(w: int, h: int) -> Image.Image:
    # Textured, colored (non-white) -> looks like tissue.
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 180, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_small_roi_falls_back_to_single_resized_patch():
    img = _tissue_image(100, 80)  # smaller than 224
    patches = tile_roi_image(img, patch_size=224, max_patches=16)
    assert len(patches) == 1
    assert patches[0].size == (224, 224)


def test_large_roi_tiles_into_multiple_patches():
    img = _tissue_image(700, 500)  # ~ 3x2 grid of 224 patches
    patches = tile_roi_image(img, patch_size=224, max_patches=16, tissue_only=True)
    assert len(patches) >= 2
    assert all(p.size == (224, 224) for p in patches)


def test_max_patches_cap_is_respected():
    img = _tissue_image(2000, 2000)  # 8x8 = 64 candidate patches
    patches = tile_roi_image(img, patch_size=224, max_patches=9, tissue_only=True)
    assert len(patches) == 9


def test_white_background_is_filtered_but_never_empty():
    white = Image.fromarray(np.full((700, 700, 3), 250, dtype=np.uint8), "RGB")
    patches = tile_roi_image(white, patch_size=224, max_patches=16, tissue_only=True)
    # All background -> heuristic keeps a single fallback patch, never zero.
    assert len(patches) == 1


def test_is_tissue_heuristic():
    rng = np.random.default_rng(1)
    tissue = rng.integers(40, 180, size=(224, 224, 3), dtype=np.uint8)
    white = np.full((224, 224, 3), 252, dtype=np.uint8)
    assert _is_tissue(tissue) is True
    assert _is_tissue(white) is False


def test_tile_roi_path_round_trip(tmp_path: Path):
    p = tmp_path / "roi.png"
    _tissue_image(500, 500).save(p)
    patches = tile_roi_path(p, patch_size=224, max_patches=4)
    assert 1 <= len(patches) <= 4


# --------------------------------------------------------------------------- #
# Multi-patch embedding via a fake backend (no weights)
# --------------------------------------------------------------------------- #

def test_encode_roi_multipatch_mean_pools(tmp_path: Path):
    pytest.importorskip("torch")
    import numpy as np

    from breastpath_copilot.core.fm_embeddings import FMBackend

    class _FakeBackend(FMBackend):
        name = "fake"; model_id = "d"; embedding_dim = 8; input_size = 224
        requires_hf_token = False

        def _load_model(self):  # pragma: no cover
            raise NotImplementedError

        def _forward(self, batch):  # pragma: no cover
            raise NotImplementedError

        def encode_images(self, images):
            images = list(images)
            # Return one row per patch so we can verify pooling over K patches.
            return np.ones((len(images), self.embedding_dim), dtype=np.float32) * len(images)

    p = tmp_path / "roi.png"
    _tissue_image(700, 500).save(p)
    be = _FakeBackend(device="cpu", fp16=False)
    pooled = be.encode_roi_multipatch(p, max_patches=16)
    assert pooled.shape == (8,)
    bag = be.encode_roi_multipatch(p, max_patches=16, return_bag=True)
    assert bag.ndim == 2 and bag.shape[1] == 8
    # mean-pool of constant rows == that constant
    np.testing.assert_allclose(pooled, bag.mean(axis=0))
