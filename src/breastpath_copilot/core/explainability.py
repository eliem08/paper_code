from __future__ import annotations

import pandas as pd

from breastpath_copilot.schemas import SlidePrediction


def summarize_neighbors(neighbors: pd.DataFrame, top_k: int = 3) -> str:
    if neighbors is None or neighbors.empty:
        return 'No nearest-neighbor retrieval evidence was available.'
    cols = [c for c in ['label', 'subtype', 'magnification', 'distance'] if c in neighbors.columns]
    preview = neighbors[cols].head(top_k).to_dict(orient='records')
    return f'Nearest-neighbor evidence (top {min(top_k, len(neighbors))}): {preview}'


def generate_case_summary(prediction: SlidePrediction, neighbors: pd.DataFrame | None = None) -> str:
    top_class = prediction.predicted_label
    top_prob = prediction.probabilities.get(top_class, 0.0)
    suspicious_count = len(prediction.suspicious_patch_ids)
    retrieval_note = summarize_neighbors(neighbors)
    return (
        f"Slide-level prediction suggests '{top_class}' with probability {top_prob:.3f}. "
        f"The pipeline flagged {suspicious_count} suspicious patches for review. "
        f"{retrieval_note} "
        'This output is intended for research support and must be verified by a pathologist.'
    )
