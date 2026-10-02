"""Patient-disjoint nearest-neighbor index over MoE / FM embeddings.

A thin numpy-native index that operates on the (embedding_matrix, manifest
DataFrame) shape returned by :class:`breastpath_copilot.core.fm_cache.FMEmbeddingCache`
and by the MCP ``moe_forward_tool``. Distinct from the
:class:`breastpath_copilot.core.retrieval.RetrievalIndex` which expects a
DataFrame with ``emb_*`` columns and uses sklearn's NearestNeighbors -- we
keep that one for backwards-compat with existing scripts.

Design notes:

* **Cosine is the default metric.** Foundation-model embeddings (UNI,
  Virchow2, Phikon-v2, and our MoE projections) live on / near a sphere
  after LayerNorm; cosine similarity is more robust than L2 for them.
* **Patient-disjoint by construction.** Every query takes an explicit
  ``query_patient_id`` argument; all index rows with the same patient id
  are filtered out before top-k. This matches the patient-aware split
  protocol used throughout the Q1 evaluation.
* **No external FAISS dep.** For our scale (BreaKHis + BACH + BRACS ~13k
  RoIs) numpy matrix-multiply is fast enough and removes a heavy install.
  The internal interface keeps the door open for a FAISS-backed swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd


METRIC_TYPE = Literal["cosine", "l2"]


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


@dataclass
class RetrievalNeighbor:
    """One retrieved neighbor."""

    image_id: str
    score: float
    rank: int
    label: str | None = None
    subtype: str | None = None
    cohort: str | None = None
    patient_id: str | None = None
    extra: dict[str, object] | None = None


class MoERetrievalIndex:
    """Numpy-native nearest-neighbor index with patient-disjoint queries."""

    REQUIRED_COLS = ("image_id", "patient_id")

    def __init__(
        self,
        embeddings: np.ndarray,
        manifest: pd.DataFrame,
        metric: METRIC_TYPE = "cosine",
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if len(manifest) != len(embeddings):
            raise ValueError(
                f"manifest rows ({len(manifest)}) != embedding rows ({len(embeddings)})"
            )
        missing = [c for c in self.REQUIRED_COLS if c not in manifest.columns]
        if missing:
            raise ValueError(f"manifest must contain columns {missing}")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")

        self.metric: METRIC_TYPE = metric
        self.manifest = manifest.reset_index(drop=True).copy()
        self._image_ids = self.manifest["image_id"].astype(str).to_numpy()
        self._patient_ids = self.manifest["patient_id"].astype(str).to_numpy()
        raw = embeddings.astype(np.float32, copy=False)
        if metric == "cosine":
            self._embeddings = _l2_normalize(raw)
        else:
            self._embeddings = raw

    def __len__(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self._embeddings.shape[1])

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        *,
        query_patient_id: str | None = None,
        query_image_id: str | None = None,
        same_cohort_only: bool = False,
        query_cohort: str | None = None,
    ) -> list[RetrievalNeighbor]:
        """Top-k neighbors with patient-disjoint and (optional) cohort filtering.

        Parameters
        ----------
        query_embedding
            1-D array of length ``embedding_dim``. Normalised internally
            when ``metric='cosine'``.
        top_k
            Number of neighbors to return after filtering.
        query_patient_id
            All index rows with this patient_id are excluded (the patient-
            disjoint guarantee). ``None`` skips the filter (rarely correct
            in evaluation; use only for unit tests).
        query_image_id
            Excluded directly even if patient_id is missing.
        same_cohort_only
            If True, restrict neighbors to ``query_cohort``.
        """
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embedding_dim:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self.embedding_dim}"
            )
        if self.metric == "cosine":
            q = q / max(np.linalg.norm(q), 1e-8)
            scores = self._embeddings @ q  # higher = better
        else:
            diffs = self._embeddings - q[None, :]
            scores = -np.einsum("ij,ij->i", diffs, diffs)  # negative L2^2: higher = better

        mask = np.ones(len(self), dtype=bool)
        if query_patient_id is not None:
            mask &= self._patient_ids != str(query_patient_id)
        if query_image_id is not None:
            mask &= self._image_ids != str(query_image_id)
        if same_cohort_only and query_cohort is not None and "cohort" in self.manifest.columns:
            cohorts = self.manifest["cohort"].astype(str).to_numpy()
            mask &= cohorts == str(query_cohort)
        if not mask.any():
            return []
        filtered_scores = np.where(mask, scores, -np.inf)
        k = int(min(top_k, mask.sum()))
        # argpartition for speed on the unsorted top-k, then sort the k.
        if k < len(filtered_scores):
            part_idx = np.argpartition(-filtered_scores, k)[:k]
        else:
            part_idx = np.arange(len(filtered_scores))
        ordered = part_idx[np.argsort(-filtered_scores[part_idx])]

        results: list[RetrievalNeighbor] = []
        for rank, idx in enumerate(ordered.tolist(), start=1):
            row = self.manifest.iloc[idx]
            extra = {
                c: row[c]
                for c in self.manifest.columns
                if c not in {"image_id", "patient_id", "label", "subtype", "cohort"}
            }
            results.append(
                RetrievalNeighbor(
                    image_id=str(row["image_id"]),
                    score=float(scores[idx]),
                    rank=rank,
                    label=str(row["label"]) if "label" in self.manifest.columns else None,
                    subtype=str(row["subtype"]) if "subtype" in self.manifest.columns else None,
                    cohort=str(row["cohort"]) if "cohort" in self.manifest.columns else None,
                    patient_id=str(row["patient_id"]),
                    extra=extra or None,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "metric": self.metric,
                "manifest": self.manifest,
                "embeddings": self._embeddings,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MoERetrievalIndex":
        payload = joblib.load(path)
        # Bypass the normaliser since saved embeddings are already in their
        # storage form for the chosen metric.
        obj = cls.__new__(cls)
        obj.metric = payload["metric"]
        obj.manifest = payload["manifest"]
        obj._image_ids = obj.manifest["image_id"].astype(str).to_numpy()
        obj._patient_ids = obj.manifest["patient_id"].astype(str).to_numpy()
        obj._embeddings = payload["embeddings"].astype(np.float32, copy=False)
        return obj
