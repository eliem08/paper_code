"""Tests for the Phikon-v2 fine-tuning components, using a tiny fake backbone
so nothing is downloaded and the loop runs on CPU in milliseconds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from breastpath_copilot.training.datasets import (  # noqa: E402
    BreastPathologyDataset,
    UNIFIED_LABEL_TO_INDEX,
    build_train_val_datasets,
    collate_batch,
    compute_class_weights,
)
from breastpath_copilot.training.finetune import (  # noqa: E402
    FinetuneConfig,
    Phikon2Classifier,
    Phikon2Finetuner,
    classification_metrics,
)


HIDDEN = 16


class _FakeBackboneOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeBackbone(nn.Module):
    """Mimics a HF ViT encoder: pixel_values -> object with last_hidden_state
    of shape (B, num_tokens, hidden). Uses a global-average-pool over the
    flattened image then projects to a token sequence so gradients flow."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.proj = nn.Linear(3, hidden)

    def forward(self, pixel_values: torch.Tensor) -> _FakeBackboneOutput:
        # (B, 3, H, W) -> (B, 3) global mean -> (B, hidden) -> (B, 2, hidden)
        pooled = pixel_values.mean(dim=(2, 3))  # (B, 3)
        feats = self.proj(pooled)  # (B, hidden)
        seq = torch.stack([feats, feats], dim=1)  # (B, 2, hidden) CLS + 1
        return _FakeBackboneOutput(seq)


def _make_manifest(tmp_path: Path, n_per_class: int = 4) -> pd.DataFrame:
    rows = []
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(exist_ok=True)
    for label in UNIFIED_LABEL_TO_INDEX:
        for i in range(n_per_class):
            p = img_dir / f"{label}_{i}.png"
            PILImage.new("RGB", (32, 32), color=(i * 10 % 255, 40, 120)).save(p)
            split = "val" if i == 0 else "train"
            rows.append(
                {
                    "image_path": str(p),
                    "image_id": f"{label}_{i}",
                    "patient_id": f"{label}_p{i}",
                    "label": label,
                    "subtype": f"{label}_sub",
                    "cohort": "breakhis" if i % 2 == 0 else "bracs",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _classifier() -> Phikon2Classifier:
    return Phikon2Classifier(
        num_classes=len(UNIFIED_LABEL_TO_INDEX),
        mode="full",
        backbone=_FakeBackbone(),
        hidden_size=HIDDEN,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def test_dataset_returns_pixel_values_and_label(tmp_path: Path):
    df = _make_manifest(tmp_path, n_per_class=2)
    ds = BreastPathologyDataset(df, input_size=32, train=False)
    item = ds[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["label_idx"].item() in range(5)
    assert item["image_id"]
    assert item["cohort"] in {"breakhis", "bracs"}


def test_dataset_drops_unknown_labels(tmp_path: Path):
    df = _make_manifest(tmp_path, n_per_class=1)
    df.loc[len(df)] = {
        "image_path": df.iloc[0]["image_path"],
        "image_id": "junk",
        "patient_id": "junk",
        "label": "not_a_real_label",
        "subtype": "x",
        "cohort": "bach",
        "split": "train",
    }
    ds = BreastPathologyDataset(df, input_size=32, train=False)
    assert len(ds) == len(df) - 1  # the junk row is dropped


def test_dataset_missing_columns_raises():
    bad = pd.DataFrame({"image_path": ["x"], "label": ["benign"]})
    with pytest.raises(ValueError):
        BreastPathologyDataset(bad)


def test_build_train_val_datasets_splits_on_column(tmp_path: Path):
    df = _make_manifest(tmp_path, n_per_class=4)
    train_ds, val_ds = build_train_val_datasets(df, input_size=32)
    # 1 of every 4 per class is 'val'.
    assert len(val_ds) == len(UNIFIED_LABEL_TO_INDEX) * 1
    assert len(train_ds) == len(UNIFIED_LABEL_TO_INDEX) * 3


def test_collate_batch_stacks_tensors(tmp_path: Path):
    df = _make_manifest(tmp_path, n_per_class=2)
    ds = BreastPathologyDataset(df, input_size=32, train=False)
    batch = collate_batch([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape == (3, 3, 32, 32)
    assert batch["label_idx"].shape == (3,)
    assert len(batch["cohort"]) == 3


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #

def test_compute_class_weights_inverse_frequency():
    labels = ["benign"] * 8 + ["dcis"] * 2  # benign over-represented
    weights = compute_class_weights(labels)
    bi = UNIFIED_LABEL_TO_INDEX["benign"]
    di = UNIFIED_LABEL_TO_INDEX["dcis"]
    # Rare class (dcis) must get a higher weight than the common one (benign).
    assert weights[di] > weights[bi]
    # Absent classes default to 1.0.
    assert weights[UNIFIED_LABEL_TO_INDEX["invasive"]] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_classification_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4])
    m = classification_metrics(y, y, num_classes=5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_classification_metrics_ignores_absent_classes_in_macro():
    # Only classes 0 and 1 present in ground truth.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = classification_metrics(y_true, y_pred, num_classes=5)
    assert m["macro_f1"] == pytest.approx(1.0)  # absent classes don't drag it to 2/5


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

def test_classifier_forward_shape():
    clf = _classifier()
    logits = clf(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 5)


def test_full_mode_encoder_receives_gradient():
    clf = _classifier()
    logits = clf(torch.randn(4, 3, 32, 32))
    loss = logits.sum()
    loss.backward()
    grad = clf.backbone.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_linear_probe_freezes_encoder():
    clf = Phikon2Classifier(
        num_classes=5, mode="linear_probe", backbone=_FakeBackbone(), hidden_size=HIDDEN
    )
    assert all(not p.requires_grad for p in clf.backbone.parameters())
    logits = clf(torch.randn(2, 3, 32, 32))
    logits.sum().backward()
    # Encoder grads stay None; head grads exist.
    assert clf.backbone.proj.weight.grad is None
    assert clf.classifier.weight.grad is not None


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Phikon2Classifier(num_classes=5, mode="bogus", backbone=_FakeBackbone(), hidden_size=HIDDEN)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

def _loaders(tmp_path: Path):
    df = _make_manifest(tmp_path, n_per_class=4)
    train_ds, val_ds = build_train_val_datasets(df, input_size=32)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader


def test_finetuner_one_step_updates_parameters(tmp_path: Path):
    clf = _classifier()
    cfg = FinetuneConfig(mode="full", epochs=1, batch_size=4, use_amp=False, device="cpu", num_workers=0)
    trainer = Phikon2Finetuner(model=clf, config=cfg)
    before = clf.classifier.weight.detach().clone()
    train_loader, val_loader = _loaders(tmp_path)
    summary = trainer.fit(train_loader, val_loader)
    after = clf.classifier.weight.detach().clone()
    assert not torch.allclose(before, after), "classifier weights should update"
    assert "history" in summary and len(summary["history"]) == 1
    assert "val" in summary["history"][0]


def test_finetuner_saves_best_checkpoint(tmp_path: Path):
    clf = _classifier()
    out = tmp_path / "ckpt"
    cfg = FinetuneConfig(mode="full", epochs=2, batch_size=4, use_amp=False, device="cpu", num_workers=0)
    trainer = Phikon2Finetuner(model=clf, config=cfg, output_dir=out)
    train_loader, val_loader = _loaders(tmp_path)
    summary = trainer.fit(train_loader, val_loader)
    assert summary["best_epoch"] >= 0
    assert summary["best_checkpoint"] is not None
    assert (out / "best.pt").exists()
    assert (out / "training_summary.json").exists()
    payload = torch.load(out / "best.pt", map_location="cpu")
    assert "model_state_dict" in payload
    assert payload["label_to_index"] == UNIFIED_LABEL_TO_INDEX


def test_finetuner_per_cohort_metrics_present(tmp_path: Path):
    clf = _classifier()
    cfg = FinetuneConfig(mode="full", epochs=1, batch_size=4, use_amp=False, device="cpu", num_workers=0)
    trainer = Phikon2Finetuner(model=clf, config=cfg)
    train_loader, val_loader = _loaders(tmp_path)
    metrics = trainer.evaluate(val_loader)
    assert "per_cohort" in metrics
    assert set(metrics["per_cohort"]).issubset({"breakhis", "bracs", "bach"})


def test_checkpoint_encoder_roundtrips_into_phikon_backend(tmp_path: Path):
    """A saved finetuner checkpoint's backbone.* keys can be loaded into a
    matching encoder via the PhikonV2Backend helper (using a fake encoder)."""
    from breastpath_copilot.core.fm_embeddings import PhikonV2Backend

    clf = _classifier()
    out = tmp_path / "ckpt"
    cfg = FinetuneConfig(mode="full", epochs=1, batch_size=4, use_amp=False, device="cpu", num_workers=0)
    trainer = Phikon2Finetuner(model=clf, config=cfg, output_dir=out)
    ckpt = trainer.save_checkpoint("best.pt", extra={"epoch": 0})

    # Load encoder weights into a fresh fake backbone via the static helper.
    fresh = _FakeBackbone()
    PhikonV2Backend._load_finetuned_encoder(fresh, str(ckpt))
    # Weights should now match the trained classifier's backbone.
    torch.testing.assert_close(fresh.proj.weight, clf.backbone.proj.weight)


def test_checkpoint_without_backbone_keys_raises(tmp_path: Path):
    from breastpath_copilot.core.fm_embeddings import PhikonV2Backend

    bad = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {"classifier.weight": torch.zeros(5, HIDDEN)}}, bad)
    with pytest.raises(ValueError):
        PhikonV2Backend._load_finetuned_encoder(_FakeBackbone(), str(bad))
