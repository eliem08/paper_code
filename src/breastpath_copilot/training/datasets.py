"""PyTorch dataset over the unified multi-cohort manifest.

The dataset reads images on-demand and applies a deterministic ImageNet-style
preprocessing pipeline matched to the ViT-Large/16 input contract used by
Phikon-v2 (224x224, ImageNet mean/std). Augmentations are kept conservative
because H&E hue jitter is risky for diagnostic-classification fine-tuning.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# BRACS RoIs can exceed 100 megapixels; these are trusted local files, so
# disable PIL's decompression-bomb guard to avoid crashing fine-tuning runs.
Image.MAX_IMAGE_PIXELS = None


UNIFIED_LABEL_TO_INDEX: dict[str, int] = {
    "normal": 0,
    "benign": 1,
    "atypia": 2,
    "dcis": 3,
    "invasive": 4,
}
INDEX_TO_UNIFIED_LABEL: dict[int, str] = {v: k for k, v in UNIFIED_LABEL_TO_INDEX.items()}


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _train_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class BreastPathologyDataset(Dataset):
    """Dataset over the unified-manifest schema (cohort + image_path + label).

    Rows whose ``label`` is not in :data:`UNIFIED_LABEL_TO_INDEX` are dropped.
    """

    REQUIRED_COLS = ("image_path", "label", "image_id", "patient_id", "cohort")

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        input_size: int = 224,
        train: bool = False,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in manifest.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        self.label_to_index = dict(label_to_index or UNIFIED_LABEL_TO_INDEX)
        df = manifest[manifest["label"].astype(str).isin(self.label_to_index)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows match label_to_index keys {sorted(self.label_to_index)}"
            )
        self.manifest = df
        self.input_size = int(input_size)
        self.train = bool(train)
        self.transform = _train_transform(input_size) if train else _eval_transform(input_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[int(idx)]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.transform(image)
        label_idx = self.label_to_index[str(row["label"])]
        return {
            "pixel_values": pixel_values,
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "cohort": str(row["cohort"]),
            "label": str(row["label"]),
            "subtype": str(row.get("subtype", "")) if "subtype" in self.manifest.columns else "",
        }


def collate_batch(items: Iterable[dict]) -> dict:
    """Default collate that stacks tensor fields and keeps metadata as lists."""
    items = list(items)
    if not items:
        return {}
    out: dict = {
        "pixel_values": torch.stack([it["pixel_values"] for it in items], dim=0),
        "label_idx": torch.stack([it["label_idx"] for it in items], dim=0),
    }
    for k in ("image_id", "patient_id", "cohort", "label", "subtype"):
        out[k] = [it[k] for it in items]
    return out


def compute_class_weights(
    labels: Iterable[str],
    label_to_index: dict[str, int] | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Inverse-frequency class weights normalised to mean 1 over present classes."""
    label_to_index = dict(label_to_index or UNIFIED_LABEL_TO_INDEX)
    counts = Counter(str(x) for x in labels if str(x) in label_to_index)
    num_classes = len(label_to_index)
    weights = torch.ones(num_classes, dtype=torch.float32)
    present = [label_to_index[k] for k in counts]
    for label, count in counts.items():
        weights[label_to_index[label]] = 1.0 / (count + eps)
    if present:
        # Normalise so the mean weight over *present* classes is 1, then keep
        # absent classes pinned at a neutral 1.0 (they contribute no loss).
        present_idx = torch.tensor(present, dtype=torch.long)
        present_mean = weights[present_idx].mean().clamp(min=eps)
        weights[present_idx] = weights[present_idx] / present_mean
        absent = [i for i in range(num_classes) if i not in set(present)]
        if absent:
            weights[torch.tensor(absent, dtype=torch.long)] = 1.0
    return weights


def build_train_val_datasets(
    manifest: pd.DataFrame,
    *,
    val_split_col: str = "split",
    val_value: str = "val",
    train_value: str = "train",
    input_size: int = 224,
    label_to_index: dict[str, int] | None = None,
) -> tuple[BreastPathologyDataset, BreastPathologyDataset]:
    """Split a manifest by an existing ``split`` column.

    Training rows are selected by ``split == train_value`` (NOT ``!= val_value``)
    so a held-out ``test`` split is never leaked into fine-tuning. For
    multi-cohort training where some cohorts lack a native split, pre-compute
    the split column before calling this helper.
    """
    if val_split_col not in manifest.columns:
        raise ValueError(f"Manifest missing required split column {val_split_col!r}")
    train_df = manifest[manifest[val_split_col].astype(str) == str(train_value)].reset_index(drop=True)
    val_df = manifest[manifest[val_split_col].astype(str) == str(val_value)].reset_index(drop=True)
    return (
        BreastPathologyDataset(
            train_df, input_size=input_size, train=True, label_to_index=label_to_index
        ),
        BreastPathologyDataset(
            val_df, input_size=input_size, train=False, label_to_index=label_to_index
        ),
    )
