"""Fine-tuning utilities for the Q1 pathology MoE pipeline.

Two public surfaces:

* :class:`BreastPathologyDataset` -- a PyTorch ``Dataset`` over the unified
  multi-cohort manifest, returning ``(pixel_values, label_idx, metadata)``
  tuples ready for a HuggingFace ViT-class encoder.
* :class:`Phikon2Classifier` + :class:`Phikon2Finetuner` -- the model
  wrapper and training loop that implement the *hybrid* compute strategy:
  UNI and Virchow2 stay frozen on cached embeddings, while Phikon-v2 is
  fine-tuned end-to-end on the unified train pool.
"""

from breastpath_copilot.training.datasets import (
    BreastPathologyDataset,
    UNIFIED_LABEL_TO_INDEX,
    build_train_val_datasets,
    compute_class_weights,
)
from breastpath_copilot.training.finetune import (
    FinetuneConfig,
    FMFineTuneClassifier,
    Phikon2Classifier,
    Phikon2Finetuner,
)

__all__ = [
    "BreastPathologyDataset",
    "FinetuneConfig",
    "FMFineTuneClassifier",
    "Phikon2Classifier",
    "Phikon2Finetuner",
    "UNIFIED_LABEL_TO_INDEX",
    "build_train_val_datasets",
    "compute_class_weights",
]
