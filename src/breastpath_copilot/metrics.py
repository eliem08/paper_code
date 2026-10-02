from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    labels = ['benign', 'malignant']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape != (2, 2):
        raise ValueError(f'Expected 2x2 confusion matrix, got shape {cm.shape}')

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score((y_true == 'malignant').astype(int), y_score)

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, pos_label='malignant')),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'auc': float(auc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
