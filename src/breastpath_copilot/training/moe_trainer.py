"""Train the MoE classification head on cached foundation-model embeddings.

Because UNI / Virchow2 / (fine-tuned) Phikon-v2 embeddings are pre-computed
and cached, the MoE head is tiny and trains in seconds on CPU. This module:

* aligns multiple per-FM caches by ``image_id`` into a single tensor bundle,
* trains :class:`MoEClassifier` with AdamW + class-weighted CE (+ the MoE
  auxiliary load-balance loss for ``soft_topk``),
* evaluates with classification metrics, calibration, and the early-detection
  operating-point report, optionally with temperature scaling fit on the
  validation split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.models.moe import MoEClassifier, MoEConfig
from breastpath_copilot.training.calibration import TemperatureScaler, calibration_report
from breastpath_copilot.training.datasets import UNIFIED_LABEL_TO_INDEX, compute_class_weights
from breastpath_copilot.training.finetune import classification_metrics
from breastpath_copilot.training.threshold_metrics import early_detection_operating_report


@dataclass
class EmbeddingBundle:
    """Aligned multi-FM embeddings + labels for a set of cases."""

    image_ids: list[str]
    expert_embeddings: dict[str, np.ndarray]  # fm_name -> (N, D_fm)
    labels: np.ndarray  # (N,) int label indices
    magnifications: list[str] | None
    cohorts: list[str]
    expert_dims: dict[str, int]

    def __len__(self) -> int:
        return len(self.image_ids)


def load_aligned_bundle(
    cache_root: str | Path,
    cohort: str,
    fm_names: list[str],
    *,
    manifest: pd.DataFrame | None = None,
    label_to_index: Mapping[str, int] | None = None,
) -> EmbeddingBundle:
    """Load and align cached embeddings for several FMs by image_id.

    If ``manifest`` is given, only its image_ids are used (and its label /
    magnification / cohort columns are authoritative); otherwise the first
    FM cache's manifest is used.
    """
    label_to_index = dict(label_to_index or UNIFIED_LABEL_TO_INDEX)
    caches = {fm: FMEmbeddingCache(cache_root, cohort, fm) for fm in fm_names}
    base_manifest = manifest if manifest is not None else caches[fm_names[0]].load_manifest()
    base_manifest = base_manifest[base_manifest["label"].astype(str).isin(label_to_index)]

    wanted_ids = base_manifest["image_id"].astype(str).tolist()
    aligned: dict[str, np.ndarray] = {}
    surviving: list[str] | None = None
    for fm in fm_names:
        rows, vecs = caches[fm].slice_by_image_ids(wanted_ids)
        ids = rows["image_id"].astype(str).tolist() if not rows.empty else []
        order = {iid: k for k, iid in enumerate(ids)}
        if surviving is None:
            surviving = ids
        else:
            surviving = [i for i in surviving if i in order]
    if not surviving:
        raise ValueError("No image_ids common to all requested FM caches.")

    # Second pass: reslice to the common surviving set in a stable order.
    for fm in fm_names:
        rows, vecs = caches[fm].slice_by_image_ids(surviving)
        idmap = {iid: k for k, iid in enumerate(rows["image_id"].astype(str).tolist())}
        order = [idmap[i] for i in surviving]
        aligned[fm] = np.asarray(vecs)[order]

    meta = base_manifest.set_index("image_id").loc[surviving].reset_index()
    labels = np.array([label_to_index[str(l)] for l in meta["label"]], dtype=np.int64)
    mags = [str(m) for m in meta["magnification"]] if "magnification" in meta.columns else None
    cohorts = [str(c) for c in meta["cohort"]] if "cohort" in meta.columns else [cohort] * len(surviving)
    expert_dims = {fm: int(aligned[fm].shape[1]) for fm in fm_names}
    return EmbeddingBundle(
        image_ids=list(surviving),
        expert_embeddings=aligned,
        labels=labels,
        magnifications=mags,
        cohorts=cohorts,
        expert_dims=expert_dims,
    )


@dataclass
class MoETrainConfig:
    routing: str = "soft_topk"
    shared_dim: int = 512
    top_k: int = 2
    load_balance_weight: float = 0.01
    per_dim_weights: bool = False
    num_classes: int = 5
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    use_class_weights: bool = True
    calibrate: bool = True
    loss_kind: str = "ce"   # "ce" (class-weighted) or "focal"
    focal_gamma: float = 1.5
    seed: int = 42
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_moe_config(self) -> MoEConfig:
        return MoEConfig(
            routing=self.routing,
            shared_dim=self.shared_dim,
            top_k=self.top_k,
            load_balance_weight=self.load_balance_weight,
            per_dim_weights=self.per_dim_weights,
        )


def _iter_minibatches(n: int, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


@dataclass
class MoETrainer:
    config: MoETrainConfig
    expert_dims: dict[str, int]
    class_weights: torch.Tensor | None = None
    # Optional explicit "needs attention" positive class indices for the binary
    # early-detection collapse. Defaults (None) to the 5-class unified vocab.
    attention_positive_indices: list[int] | None = None
    model: MoEClassifier = field(init=False)

    def __post_init__(self) -> None:
        self.device = self.config.resolved_device()
        torch.manual_seed(self.config.seed)
        self.model = MoEClassifier(
            self.expert_dims,
            num_classes=self.config.num_classes,
            config=self.config.to_moe_config(),
            dropout=self.config.dropout,
        ).to(self.device)

    def _to_tensors(self, bundle: EmbeddingBundle, idx: np.ndarray):
        experts = {
            fm: torch.as_tensor(bundle.expert_embeddings[fm][idx], dtype=torch.float32, device=self.device)
            for fm in self.expert_dims
        }
        labels = torch.as_tensor(bundle.labels[idx], dtype=torch.long, device=self.device)
        mags = [bundle.magnifications[i] for i in idx] if bundle.magnifications is not None else None
        return experts, labels, mags

    def fit(self, train: EmbeddingBundle, val: EmbeddingBundle | None = None) -> dict:
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        weight = self.class_weights.to(self.device) if self.class_weights is not None else None
        ce = nn.CrossEntropyLoss(weight=weight)

        def loss_fn(logits, labels):
            if self.config.loss_kind == "focal":
                # Class-weighted focal loss: (1-p_t)^gamma * CE. Helps the hard,
                # under-represented classes (ADH/FEA) that drag weighted-F1.
                ce_per = nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
                pt = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
                return (((1.0 - pt) ** self.config.focal_gamma) * ce_per).mean()
            elif self.config.loss_kind == "ordinal":
                # Squared Earth Mover's Distance for ordinal regression
                probs = torch.softmax(logits, dim=1)
                cdf_probs = torch.cumsum(probs, dim=1)
                true_one_hot = torch.zeros_like(logits)
                true_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                cdf_true = torch.cumsum(true_one_hot, dim=1)
                emd_sq = torch.sum((cdf_probs - cdf_true) ** 2, dim=1)
                if weight is not None:
                    batch_weights = weight[labels]
                    return (emd_sq * batch_weights).sum() / batch_weights.sum().clamp(min=1e-6)
                return emd_sq.mean()
            return ce(logits, labels)

        history: list[dict] = []
        for epoch in range(self.config.epochs):
            self.model.train()
            running = 0.0
            seen = 0
            for batch_idx in _iter_minibatches(
                len(train), self.config.batch_size, shuffle=True, seed=self.config.seed + epoch
            ):
                experts, labels, mags = self._to_tensors(train, batch_idx)
                opt.zero_grad(set_to_none=True)
                logits, moe_out = self.model(experts, magnifications=mags)
                loss = loss_fn(logits, labels) + moe_out.aux_loss
                loss.backward()
                opt.step()
                running += float(loss.item()) * len(batch_idx)
                seen += len(batch_idx)
            rec = {"epoch": epoch, "train_loss": running / max(1, seen)}
            history.append(rec)
        result: dict = {"history": history, "epochs": self.config.epochs}
        if val is not None:
            result["val"] = self.evaluate(val, fit_temperature_on_val=self.config.calibrate)
        return result

    @torch.no_grad()
    def predict_logits(self, bundle: EmbeddingBundle) -> np.ndarray:
        self.model.eval()
        out_chunks: list[np.ndarray] = []
        for batch_idx in _iter_minibatches(
            len(bundle), self.config.batch_size, shuffle=False, seed=0
        ):
            experts, _, mags = self._to_tensors(bundle, batch_idx)
            logits, _ = self.model(experts, magnifications=mags)
            out_chunks.append(logits.cpu().numpy())
        return np.concatenate(out_chunks, axis=0) if out_chunks else np.empty((0, self.config.num_classes))

    def evaluate(self, bundle: EmbeddingBundle, *, fit_temperature_on_val: bool = False) -> dict:
        logits = self.predict_logits(bundle)
        labels = bundle.labels
        preds = logits.argmax(axis=1)
        metrics = classification_metrics(labels, preds, self.config.num_classes)

        scaler = None
        if fit_temperature_on_val and len(bundle) > 1:
            scaler = TemperatureScaler().fit(logits, labels)
        metrics["calibration"] = calibration_report(logits, labels, scaler=scaler)

        probs = _softmax_np(logits)
        metrics["early_detection"] = early_detection_operating_report(
            probs, labels, positive_indices=self.attention_positive_indices
        )

        # Per-cohort breakdown
        per_cohort: dict[str, dict] = {}
        cohorts = np.array(bundle.cohorts)
        for coh in np.unique(cohorts):
            mask = cohorts == coh
            per_cohort[str(coh)] = classification_metrics(
                labels[mask], preds[mask], self.config.num_classes
            )
        metrics["per_cohort"] = per_cohort
        return metrics

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "expert_dims": self.expert_dims,
                "config": self.config.__dict__,
            },
            path,
        )
        return path


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_class_weights_from_bundle(
    bundle: EmbeddingBundle,
    index_to_label: dict[int, str],
    label_to_index: dict[str, int] | None = None,
) -> torch.Tensor:
    """Inverse-frequency class weights for the bundle's labels.

    Pass ``label_to_index`` for non-default (3/7-class) schemes so the weight
    vector has the right length; defaults to the 5-class unified vocab.
    """
    labels = [index_to_label[int(i)] for i in bundle.labels]
    li = label_to_index or {v: k for k, v in index_to_label.items()}
    return compute_class_weights(labels, label_to_index=li)
