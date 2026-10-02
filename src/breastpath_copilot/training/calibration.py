"""Probability calibration utilities for the MoE classifier.

Calibration matters for an early-detection setting: a screening-style triage
decision is only reliable if the predicted probabilities mean what they
say. We provide:

* :func:`expected_calibration_error` -- ECE over equal-width confidence bins.
* :func:`brier_score` -- multiclass Brier score (mean squared error of the
  one-hot vs predicted probability vectors).
* :class:`TemperatureScaler` -- single-parameter temperature scaling fit on a
  held-out split by minimising NLL (Guo et al., 2017).
* :func:`reliability_table` -- per-bin accuracy/confidence for reliability
  diagrams.

All sklearn-free; depends only on numpy + torch (torch only for the scaler's
optimisation, which is a 1-D convex problem).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label ECE with equal-width confidence bins in [0, 1]."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or len(probs) == 0:
        return 0.0
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge in the last bin.
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int | None = None) -> float:
    """Multiclass Brier score = mean ||p - onehot(y)||^2 over samples."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or len(probs) == 0:
        return 0.0
    k = num_classes or probs.shape[1]
    onehot = np.zeros((len(labels), k), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


def reliability_table(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[ReliabilityBin]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        table.append(
            ReliabilityBin(
                lower=float(lo),
                upper=float(hi),
                count=count,
                avg_confidence=float(confidences[mask].mean()) if count else 0.0,
                accuracy=float(accuracies[mask].mean()) if count else 0.0,
            )
        )
    return table


class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., 2017).

    Fit ``T`` on validation logits by minimising NLL; apply by dividing logits
    by ``T`` before softmax. ``T > 1`` softens over-confident predictions.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.temperature: float = float(init_temperature)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> "TemperatureScaler":
        import torch

        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        log_T = torch.zeros(1, requires_grad=True)  # optimise in log-space -> T > 0
        optimizer = torch.optim.Adam([log_T], lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled = logits_t / log_T.exp()
            loss = loss_fn(scaled, labels_t)
            loss.backward()
            optimizer.step()
        self.temperature = float(log_T.exp().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-scaled probabilities."""
        return _softmax(np.asarray(logits, dtype=np.float64) / self.temperature)


def calibration_report(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    scaler: TemperatureScaler | None = None,
) -> dict[str, float]:
    """ECE + Brier before and (optionally) after temperature scaling."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    raw_probs = _softmax(logits)
    report = {
        "ece": expected_calibration_error(raw_probs, labels, n_bins=n_bins),
        "brier": brier_score(raw_probs, labels),
    }
    if scaler is not None:
        cal_probs = scaler.transform(logits)
        report["temperature"] = scaler.temperature
        report["ece_calibrated"] = expected_calibration_error(cal_probs, labels, n_bins=n_bins)
        report["brier_calibrated"] = brier_score(cal_probs, labels)
    return report
