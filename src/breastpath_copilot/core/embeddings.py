from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from breastpath_copilot.schemas import PatchRecord


def _lbp_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    gray = gray.astype(np.float32)
    c = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, :-2], gray[1:-1, :-2],
    ]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        code |= ((n >= c).astype(np.uint8) << i)
    hist, _ = np.histogram(code, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-8)


def _channel_hist(channel: np.ndarray, bins: int = 16, value_range: tuple[int, int] = (0, 255)) -> np.ndarray:
    hist, _ = np.histogram(channel, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-8)


class EmbeddingBackend:
    """Lightweight handcrafted embedding backend.

    This is still not a pathology foundation model, but it is much richer than a
    single flattened intensity histogram and works offline on a CPU-only setup.
    """

    def __init__(self, embedding_dim: int = 256, image_size: int = 96) -> None:
        self.embedding_dim = embedding_dim
        self.image_size = image_size

    def _load_image(self, patch_path: str | Path, image_size: int | None = None) -> np.ndarray:
        size = image_size or self.image_size
        image = Image.open(patch_path).convert('RGB').resize((size, size))
        return np.asarray(image, dtype=np.uint8)

    def _compose_features(self, rgb: np.ndarray) -> np.ndarray:
        arr = rgb.astype(np.float32)
        gray = arr.mean(axis=2)

        # RGB + simple stain/color distributions
        feats: list[np.ndarray] = []
        for ch in range(3):
            feats.append(_channel_hist(arr[..., ch], bins=32))

        # Ratio-style features to mimic stain balance information
        denom = arr.sum(axis=2) + 1e-6
        r_ratio = arr[..., 0] / denom
        g_ratio = arr[..., 1] / denom
        b_ratio = arr[..., 2] / denom
        for ratio in (r_ratio, g_ratio, b_ratio):
            hist, _ = np.histogram(ratio, bins=16, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            feats.append(hist / (np.linalg.norm(hist) + 1e-8))

        # Gray statistics and texture
        grad_y, grad_x = np.gradient(gray)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_hist, _ = np.histogram(grad_mag, bins=32, range=(0.0, max(1.0, float(grad_mag.max()))))
        grad_hist = grad_hist.astype(np.float32)
        feats.append(grad_hist / (np.linalg.norm(grad_hist) + 1e-8))
        feats.append(_lbp_histogram(gray, bins=32))

        # Color moments across channels
        moments = []
        for ch in range(3):
            c = arr[..., ch].reshape(-1)
            moments.extend([
                float(c.mean() / 255.0),
                float(c.std() / 255.0),
                float(((c - c.mean()) ** 3).mean() / (255.0 ** 3 + 1e-8)),
            ])
        feats.append(np.asarray(moments, dtype=np.float32))

        vec = np.concatenate(feats, axis=0).astype(np.float32)
        if len(vec) >= self.embedding_dim:
            vec = vec[: self.embedding_dim]
        else:
            vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
        return vec / (np.linalg.norm(vec) + 1e-8)

    def encode_patch(self, patch_path: str | Path) -> np.ndarray:
        rgb = self._load_image(patch_path)
        return self._compose_features(rgb)

    def encode_patch_multiscale(self, patch_path: str | Path, scales: Iterable[int] = (64, 96, 128)) -> np.ndarray:
        vecs: list[np.ndarray] = []
        scales = tuple(scales)
        per_scale_dim = max(64, self.embedding_dim // max(1, len(scales)))
        for scale in scales:
            rgb = self._load_image(patch_path, image_size=int(scale))
            sub_backend = EmbeddingBackend(embedding_dim=per_scale_dim, image_size=int(scale))
            vecs.append(sub_backend._compose_features(rgb))
        vec = np.concatenate(vecs, axis=0).astype(np.float32)
        if len(vec) >= self.embedding_dim:
            vec = vec[: self.embedding_dim]
        else:
            vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
        return vec / (np.linalg.norm(vec) + 1e-8)


def compute_patch_embeddings(records: list[PatchRecord], backend: EmbeddingBackend) -> tuple[list[str], np.ndarray]:
    patch_ids: list[str] = []
    vectors: list[np.ndarray] = []
    for record in records:
        if not record.patch_path:
            continue
        patch_ids.append(record.patch_id)
        vectors.append(backend.encode_patch(record.patch_path))
    if not vectors:
        return [], np.empty((0, backend.embedding_dim), dtype=np.float32)
    return patch_ids, np.vstack(vectors)
