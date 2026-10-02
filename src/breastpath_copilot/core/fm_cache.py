"""Disk cache for foundation-model embeddings.

Layout (per cohort x FM):
    <root>/<cohort>/<fm_name>/
        embeddings.npy      # (N, D) float32 array
        manifest.parquet    # one row per image with metadata + row_idx
        metadata.json       # sidecar with FMEmbeddingMetadata schema

The cache is content-addressable on (cohort, fm_name) and idempotent: callers
can ask for embeddings for an arbitrary input manifest, and only the missing
rows are computed. Reads memory-map the .npy file so MoE / retrieval / agentic
code can slice without loading the whole matrix into RAM.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from breastpath_copilot.schemas import FMEmbeddingMetadata
from breastpath_copilot.utils.io import ensure_dir, save_json


_REQUIRED_MANIFEST_COLS = ("image_id", "image_path")


def _cache_dir(root: str | Path, cohort: str, fm_name: str) -> Path:
    return Path(root) / cohort.lower() / fm_name.lower()


def _hash_manifest(df: pd.DataFrame) -> str:
    cols = [c for c in ("image_id", "image_path") if c in df.columns]
    payload = df[cols].astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image_id values")


class FMEmbeddingCache:
    """Read/write a foundation-model embedding cache for one (cohort, FM) pair."""

    def __init__(self, root: str | Path, cohort: str, fm_name: str) -> None:
        self.root = Path(root)
        self.cohort = cohort.lower()
        self.fm_name = fm_name.lower()
        self.dir = _cache_dir(root, self.cohort, self.fm_name)
        self.embeddings_path = self.dir / "embeddings.npy"
        self.manifest_path = self.dir / "manifest.parquet"
        self.metadata_path = self.dir / "metadata.json"

    def exists(self) -> bool:
        return (
            self.embeddings_path.exists()
            and self.manifest_path.exists()
            and self.metadata_path.exists()
        )

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.manifest_path)

    def load_embeddings(self, mmap: bool = True) -> np.ndarray:
        if not self.embeddings_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        return np.load(self.embeddings_path, mmap_mode="r" if mmap else None)

    def load_metadata(self) -> FMEmbeddingMetadata | None:
        if not self.metadata_path.exists():
            return None
        with self.metadata_path.open("r", encoding="utf-8") as f:
            return FMEmbeddingMetadata.model_validate_json(f.read())

    def missing_rows(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Return manifest rows whose image_id is not yet in the cache."""
        _validate_manifest(manifest)
        cached = self.load_manifest()
        if cached.empty:
            return manifest.reset_index(drop=True)
        cached_ids = set(cached["image_id"].astype(str))
        mask = ~manifest["image_id"].astype(str).isin(cached_ids)
        return manifest.loc[mask].reset_index(drop=True)

    def write(
        self,
        manifest: pd.DataFrame,
        embeddings: np.ndarray,
        model_id: str,
        embedding_dim: int,
        *,
        model_revision: str | None = None,
        input_size: int = 224,
        preprocessing: str = "imagenet",
        pooling: str = "cls_token",
        fp16_inference: bool = True,
        notes: str = "",
    ) -> FMEmbeddingMetadata:
        """Append-or-create. If the cache already exists, new rows are appended
        and any rows whose image_id collides with the existing cache are skipped.
        """
        _validate_manifest(manifest)
        if len(manifest) != len(embeddings):
            raise ValueError(
                f"Manifest rows ({len(manifest)}) must match embeddings rows ({len(embeddings)})"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Embeddings shape {embeddings.shape} does not match expected dim {embedding_dim}"
            )

        ensure_dir(self.dir)
        existing_manifest = self.load_manifest()
        existing_embeddings = (
            self.load_embeddings(mmap=False) if self.embeddings_path.exists() else None
        )

        if existing_manifest.empty:
            new_manifest = manifest.copy()
            new_manifest["row_idx"] = np.arange(len(new_manifest), dtype=np.int64)
            new_embeddings = embeddings.astype(np.float32, copy=False)
        else:
            existing_ids = set(existing_manifest["image_id"].astype(str))
            mask = ~manifest["image_id"].astype(str).isin(existing_ids)
            new_rows = manifest.loc[mask].reset_index(drop=True)
            new_vecs = embeddings[mask.values]
            base = len(existing_manifest)
            new_rows["row_idx"] = np.arange(base, base + len(new_rows), dtype=np.int64)
            new_manifest = pd.concat([existing_manifest, new_rows], ignore_index=True)
            assert existing_embeddings is not None
            new_embeddings = np.vstack(
                [existing_embeddings.astype(np.float32, copy=False), new_vecs.astype(np.float32, copy=False)]
            )

        np.save(self.embeddings_path, new_embeddings)
        new_manifest.to_parquet(self.manifest_path, index=False)

        metadata = FMEmbeddingMetadata(
            cohort=self.cohort,
            fm_name=self.fm_name,
            model_id=model_id,
            model_revision=model_revision,
            embedding_dim=int(embedding_dim),
            num_records=int(len(new_manifest)),
            input_size=int(input_size),
            preprocessing=preprocessing,
            pooling=pooling,
            fp16_inference=bool(fp16_inference),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_manifest_hash=_hash_manifest(new_manifest),
            notes=notes,
        )
        save_json(metadata.model_dump(), self.metadata_path)
        return metadata

    def slice_by_image_ids(self, image_ids: Iterable[str]) -> tuple[pd.DataFrame, np.ndarray]:
        """Return manifest rows + embeddings (in the same order as ``image_ids``).

        Image ids that are not present in the cache are skipped.
        """
        cached = self.load_manifest()
        if cached.empty:
            return cached, np.empty((0, 0), dtype=np.float32)
        ids = list(image_ids)
        lookup = cached.set_index("image_id")
        present = [i for i in ids if i in lookup.index]
        if not present:
            return cached.iloc[0:0], np.empty((0, 0), dtype=np.float32)
        rows = lookup.loc[present].reset_index()
        idx = rows["row_idx"].to_numpy(dtype=np.int64)
        embeddings = self.load_embeddings(mmap=True)[idx]
        return rows, np.asarray(embeddings)
