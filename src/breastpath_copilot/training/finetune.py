"""Phikon-v2 fine-tuning loop for the hybrid MoE compute strategy.

In the Q1 design, UNI and Virchow2 are used as *frozen* feature extractors
(their embeddings are cached once), while Phikon-v2 is the single expert that
is **fine-tuned end-to-end** on the unified multi-cohort train pool. This
module provides:

* :class:`Phikon2Classifier` -- HF Phikon-v2 encoder + linear classification
  head, with ``mode`` in {``linear_probe``, ``full``}.
* :class:`Phikon2Finetuner` -- AdamW + cosine-with-warmup schedule, AMP,
  class-weighted cross-entropy, per-cohort validation, best-checkpoint
  saving keyed on a configurable monitor metric.

The encoder is *injectable*: tests pass a tiny fake backbone so the loop can
be exercised without downloading model weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from breastpath_copilot.training.datasets import (
    INDEX_TO_UNIFIED_LABEL,
    UNIFIED_LABEL_TO_INDEX,
    collate_batch,
)


PHIKON_V2_MODEL_ID = "owkin/phikon-v2"


@dataclass
class FinetuneConfig:
    mode: str = "full"  # "full" | "linear_probe"
    num_classes: int = 5
    input_size: int = 224
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    head_lr: float | None = None  # if set, separate (higher) LR for the head
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    monitor_metric: str = "macro_f1"  # key in the val-metrics dict
    monitor_mode: str = "max"  # "max" | "min"
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class Phikon2Classifier(nn.Module):
    """Phikon-v2 encoder + linear head.

    Parameters
    ----------
    num_classes
        Output class count (5 for the unified early-detection vocab).
    mode
        ``"full"`` fine-tunes the encoder; ``"linear_probe"`` freezes it.
    backbone
        Optional pre-built encoder. When ``None``, the HF Phikon-v2 model is
        loaded lazily. Tests inject a small stand-in exposing the same
        ``(pixel_values=...) -> last_hidden_state`` contract or a plain
        callable returning a feature tensor.
    hidden_size
        Encoder output dim. Defaults to 1024 (Phikon-v2 ViT-L). Must match an
        injected backbone's output.
    """

    def __init__(
        self,
        num_classes: int = 5,
        mode: str = "full",
        *,
        backbone: nn.Module | None = None,
        hidden_size: int = 1024,
        hf_token: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("full", "linear_probe"):
            raise ValueError(f"mode must be 'full' or 'linear_probe', got {mode!r}")
        self.mode = mode
        self.hidden_size = int(hidden_size)
        self._hf_token = hf_token
        self.backbone = backbone if backbone is not None else self._load_backbone()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, int(num_classes))
        if mode == "linear_probe":
            self.freeze_encoder()

    def _load_backbone(self) -> nn.Module:
        from transformers import AutoModel

        return AutoModel.from_pretrained(PHIKON_V2_MODEL_ID, token=self._hf_token)

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def encoder_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.classifier.parameters()

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        # HF ViT-style models expose last_hidden_state; take the CLS token.
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            first = out[0]
            return first[:, 0] if first.dim() == 3 else first
        raise TypeError(f"Unsupported backbone output type: {type(out)}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear_probe":
            with torch.no_grad():
                feats = self._encode(pixel_values)
        else:
            feats = self._encode(pixel_values)
        return self.classifier(self.dropout(feats))


class FMFineTuneClassifier(nn.Module):
    """Generic fine-tuning classifier for any registered FM backend.

    Reuses each backend's correct loading + pooling (UNI -> pooled 1024-d;
    Virchow2 -> CLS + mean-patch 2560-d; Phikon-v2 -> CLS 1024-d) and adds a
    linear head. Exposes the same interface as :class:`Phikon2Classifier` so it
    is a drop-in for :class:`Phikon2Finetuner`.
    """

    def __init__(
        self,
        backend_name: str,
        num_classes: int,
        *,
        mode: str = "full",
        hf_token: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("full", "linear_probe"):
            raise ValueError(f"mode must be 'full' or 'linear_probe', got {mode!r}")
        from breastpath_copilot.core.fm_embeddings import build_backend

        kwargs: dict = {"device": "cpu", "fp16": False, "hf_token": hf_token}
        if backend_name == "phikon_v2":
            kwargs["trainable"] = True  # don't freeze inside the backend
        self._backend = build_backend(backend_name, **kwargs)
        self.backend_name = backend_name
        self.mode = mode
        self.hidden_size = int(self._backend.embedding_dim)
        # Load the pretrained encoder and wire it into the backend so we can
        # reuse the backend's architecture-specific pooling in _forward.
        self.backbone = self._backend._load_model()
        self._backend._model = self.backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, int(num_classes))
        if mode == "linear_probe":
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def encoder_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.classifier.parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear_probe":
            with torch.no_grad():
                feats = self._backend._forward(pixel_values)
        else:
            feats = self._backend._forward(pixel_values)
        return self.classifier(self.dropout(feats))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Accuracy + macro/weighted precision/recall/F1 computed without sklearn.

    Macro averages over classes present in ``y_true``. Weighted averages by
    class support (matches the 'weighted F1' reported in the BRACS paper).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {
            "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
            "macro_f1": 0.0, "weighted_f1": 0.0,
        }
    accuracy = float((y_true == y_pred).mean())
    precisions, recalls, f1s = [], [], []
    weighted_f1_num = 0.0
    total_support = 0
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = tp + fn
        if support == 0:
            # Class absent from ground truth -> skip from macro average.
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_f1_num += support * f1
        total_support += support
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1": float(weighted_f1_num / total_support) if total_support else 0.0,
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@dataclass
class Phikon2Finetuner:
    model: Phikon2Classifier
    config: FinetuneConfig
    class_weights: torch.Tensor | None = None
    output_dir: Path | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device = self.config.resolved_device()
        self.model.to(self.device)
        torch.manual_seed(self.config.seed)

    # -- optimizer / schedule ------------------------------------------------ #

    def _build_optimizer(self) -> torch.optim.Optimizer:
        head_lr = self.config.head_lr or self.config.lr
        param_groups: list[dict] = [
            {"params": list(self.model.head_parameters()), "lr": head_lr}
        ]
        if self.config.mode == "full":
            param_groups.append(
                {"params": list(self.model.encoder_parameters()), "lr": self.config.lr}
            )
        return torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    def _lr_lambda(self, total_steps: int) -> Callable[[int], float]:
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        def fn(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return fn

    # -- loops --------------------------------------------------------------- #

    def _loss_fn(self) -> nn.Module:
        weight = self.class_weights.to(self.device) if self.class_weights is not None else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=self.config.label_smoothing)

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: "torch.cuda.amp.GradScaler",
        loss_fn: nn.Module,
    ) -> float:
        self.model.train()
        if self.config.mode == "linear_probe":
            self.model.backbone.eval()
        running = 0.0
        n = 0
        use_amp = self.config.use_amp and self.device.startswith("cuda")
        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label_idx"].to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = self.model(pixel_values)
                loss = loss_fn(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                if self.config.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                optimizer.step()
            scheduler.step()
            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        return running / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        all_true: list[int] = []
        all_pred: list[int] = []
        per_cohort: dict[str, list[tuple[int, int]]] = {}
        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label_idx"]
            logits = self.model(pixel_values)
            preds = logits.argmax(dim=-1).cpu().numpy()
            true = labels.numpy()
            all_true.extend(true.tolist())
            all_pred.extend(preds.tolist())
            for coh, t, p in zip(batch.get("cohort", ["?"] * len(true)), true, preds):
                per_cohort.setdefault(coh, []).append((int(t), int(p)))
        metrics = classification_metrics(
            np.array(all_true), np.array(all_pred), self.config.num_classes
        )
        cohort_metrics = {}
        for coh, pairs in per_cohort.items():
            t = np.array([a for a, _ in pairs])
            p = np.array([b for _, b in pairs])
            cohort_metrics[coh] = classification_metrics(t, p, self.config.num_classes)
        metrics["per_cohort"] = cohort_metrics
        return metrics

    # -- orchestration ------------------------------------------------------- #

    def _is_improved(self, current: float, best: float) -> bool:
        if self.config.monitor_mode == "max":
            return current > best
        return current < best

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict:
        optimizer = self._build_optimizer()
        total_steps = max(1, len(train_loader) * self.config.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda(total_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_amp and self.device.startswith("cuda"))
        loss_fn = self._loss_fn()

        best_metric = -math.inf if self.config.monitor_mode == "max" else math.inf
        best_epoch = -1
        best_path: Path | None = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(train_loader, optimizer, scheduler, scaler, loss_fn)
            record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                record["val"] = val_metrics
                monitor_val = float(val_metrics.get(self.config.monitor_metric, 0.0))
                if self._is_improved(monitor_val, best_metric):
                    best_metric = monitor_val
                    best_epoch = epoch
                    if self.output_dir is not None:
                        best_path = self.save_checkpoint("best.pt", extra={"epoch": epoch, "metric": monitor_val})
            self.history.append(record)

        summary = {
            "best_epoch": best_epoch,
            "best_metric": best_metric if best_epoch >= 0 else None,
            "monitor_metric": self.config.monitor_metric,
            "best_checkpoint": str(best_path) if best_path else None,
            "history": self.history,
        }
        if self.output_dir is not None:
            (Path(self.output_dir)).mkdir(parents=True, exist_ok=True)
            (Path(self.output_dir) / "training_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        return summary

    def save_checkpoint(self, filename: str, extra: dict | None = None) -> Path:
        assert self.output_dir is not None, "output_dir must be set to save checkpoints"
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "label_to_index": UNIFIED_LABEL_TO_INDEX,
            "index_to_label": INDEX_TO_UNIFIED_LABEL,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path
