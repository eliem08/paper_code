"""Operating-point metrics for early detection.

The five-class unified vocabulary is collapsed to a screening-relevant binary
question -- *does this case need clinical attention?* -- by treating
``{atypia, dcis, invasive}`` as positive and ``{normal, benign}`` as negative.
This binarisation is what makes "sensitivity at high specificity" meaningful
for an early-detection narrative: we want to know how many true lesions we
catch while holding the false-alarm rate low.

All functions are sklearn-free (trapezoidal AUROC, threshold sweep).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breastpath_copilot.training.datasets import UNIFIED_LABEL_TO_INDEX

# Positive = "needs attention" (pre-invasive or worse).
ATTENTION_POSITIVE_LABELS = ("atypia", "dcis", "invasive")
ATTENTION_NEGATIVE_LABELS = ("normal", "benign")


def attention_positive_indices(
    label_to_index: dict[str, int] | None = None,
) -> list[int]:
    mapping = dict(label_to_index or UNIFIED_LABEL_TO_INDEX)
    return [mapping[l] for l in ATTENTION_POSITIVE_LABELS if l in mapping]


def binary_attention_scores(
    probs: np.ndarray,
    label_to_index: dict[str, int] | None = None,
    positive_indices: list[int] | None = None,
) -> np.ndarray:
    """Collapse multiclass probabilities to a single P(needs attention).

    ``positive_indices`` overrides the default unified-vocab positives, so any
    label scheme (3/5/7-class) can define its own "needs attention" set.
    """
    probs = np.asarray(probs, dtype=np.float64)
    pos_idx = positive_indices if positive_indices is not None else attention_positive_indices(label_to_index)
    return probs[:, pos_idx].sum(axis=1)


def binary_attention_labels(
    labels: np.ndarray,
    label_to_index: dict[str, int] | None = None,
    positive_indices: list[int] | None = None,
) -> np.ndarray:
    pos = positive_indices if positive_indices is not None else attention_positive_indices(label_to_index)
    pos_idx = set(pos)
    labels = np.asarray(labels)
    return np.array([1 if int(l) in pos_idx else 0 for l in labels], dtype=np.int64)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUROC for binary labels. Returns 0.5 for degenerate input."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank-based (Mann-Whitney U) AUROC, robust to ties via average ranks.
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def multiclass_macro_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """One-vs-rest AUROC per class + macro average (matches leaderboard 'AUC').

    Classes absent from ``labels`` are skipped from the macro average.
    Returns {"macro_auroc": ..., "per_class": {c: auroc}}.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    per_class: dict[int, float] = {}
    for c in range(num_classes):
        y = (labels == c).astype(np.int64)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        per_class[c] = roc_auc(y, probs[:, c])
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.5
    return {"macro_auroc": macro, "per_class": {int(k): float(v) for k, v in per_class.items()}}


@dataclass
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float


def _sweep(y_true: np.ndarray, y_score: np.ndarray) -> list[OperatingPoint]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    thresholds = np.unique(y_score)
    # Add boundary thresholds so we can reach sens=1 / spec=1 extremes.
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    points: list[OperatingPoint] = []
    for t in thresholds:
        pred_pos = y_score >= t
        tp = int((pred_pos & (y_true == 1)).sum())
        tn = int((~pred_pos & (y_true == 0)).sum())
        points.append(
            OperatingPoint(
                threshold=float(t),
                sensitivity=tp / n_pos,
                specificity=tn / n_neg,
            )
        )
    return points


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_specificity: float = 0.95,
) -> OperatingPoint:
    """Max sensitivity among operating points with specificity >= target.

    Falls back to the highest-specificity point if none reaches the target.
    """
    points = _sweep(y_true, y_score)
    feasible = [p for p in points if p.specificity >= min_specificity]
    if feasible:
        return max(feasible, key=lambda p: p.sensitivity)
    return max(points, key=lambda p: p.specificity)


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_sensitivity: float = 0.95,
) -> OperatingPoint:
    """Max specificity among operating points with sensitivity >= target."""
    points = _sweep(y_true, y_score)
    feasible = [p for p in points if p.sensitivity >= min_sensitivity]
    if feasible:
        return max(feasible, key=lambda p: p.specificity)
    return max(points, key=lambda p: p.sensitivity)


def early_detection_operating_report(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    label_to_index: dict[str, int] | None = None,
    positive_indices: list[int] | None = None,
    specificity_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
    sensitivity_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, float | dict]:
    """Full early-detection operating-point report from multiclass probs.

    ``positive_indices`` lets any label scheme define its own "needs
    attention" set (e.g. 3-class: {atypical, malignant}; 7-class:
    {FEA, ADH, DCIS, IC}).
    """
    y_true = binary_attention_labels(labels, label_to_index, positive_indices)
    y_score = binary_attention_scores(probs, label_to_index, positive_indices)
    report: dict[str, float | dict] = {
        "auroc_attention": roc_auc(y_true, y_score),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    sens_at_spec: dict[str, dict] = {}
    for target in specificity_targets:
        op = sensitivity_at_specificity(y_true, y_score, target)
        sens_at_spec[f"spec>={target:.2f}"] = {
            "sensitivity": op.sensitivity,
            "specificity": op.specificity,
            "threshold": op.threshold,
        }
    spec_at_sens: dict[str, dict] = {}
    for target in sensitivity_targets:
        op = specificity_at_sensitivity(y_true, y_score, target)
        spec_at_sens[f"sens>={target:.2f}"] = {
            "sensitivity": op.sensitivity,
            "specificity": op.specificity,
            "threshold": op.threshold,
        }
    report["sensitivity_at_specificity"] = sens_at_spec
    report["specificity_at_sensitivity"] = spec_at_sens
    return report
