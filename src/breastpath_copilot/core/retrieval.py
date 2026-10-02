from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def feature_columns(df: pd.DataFrame, prefix: str = 'emb_') -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


class RetrievalIndex:
    def __init__(self, metric: str = 'cosine', n_neighbors: int = 5) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
        self.metadata: pd.DataFrame | None = None
        self.emb_cols: list[str] = []

    def fit(self, df: pd.DataFrame, emb_cols: Iterable[str] | None = None) -> None:
        self.emb_cols = list(emb_cols) if emb_cols is not None else feature_columns(df)
        self.metadata = df.reset_index(drop=True).copy()
        self.nn.fit(self.metadata[self.emb_cols].values)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        *,
        exclude_image_id: str | None = None,
        allowed_split: str | None = None,
    ) -> pd.DataFrame:
        if self.metadata is None:
            raise RuntimeError('Retrieval index not fitted.')
        n_neighbors = min(len(self.metadata), max(top_k + 5, top_k))
        distances, indices = self.nn.kneighbors(np.asarray(vector).reshape(1, -1), n_neighbors=n_neighbors)
        result = self.metadata.iloc[indices[0]].copy().reset_index(drop=True)
        result.insert(0, 'distance', distances[0])
        if exclude_image_id is not None and 'image_id' in result.columns:
            result = result[result['image_id'].astype(str) != str(exclude_image_id)]
        if allowed_split is not None and 'split' in result.columns:
            result = result[result['split'].astype(str) == str(allowed_split)]
        return result.head(top_k).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'metric': self.metric, 'n_neighbors': self.n_neighbors, 'nn': self.nn, 'metadata': self.metadata, 'emb_cols': self.emb_cols}, path)

    @classmethod
    def load(cls, path: str | Path) -> 'RetrievalIndex':
        payload = joblib.load(path)
        obj = cls(metric=payload['metric'], n_neighbors=payload['n_neighbors'])
        obj.nn = payload['nn']
        obj.metadata = payload['metadata']
        obj.emb_cols = payload['emb_cols']
        return obj
