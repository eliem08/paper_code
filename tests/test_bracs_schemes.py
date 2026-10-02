"""Tests for BRACS label schemes, weighted-F1, and the multi-seed runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402
from breastpath_copilot.training.bracs_schemes import (  # noqa: E402
    SCHEME_3CLASS,
    SCHEME_7CLASS,
    get_scheme,
    remap_bracs_manifest,
)
from breastpath_copilot.training.finetune import classification_metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Schemes + remap
# --------------------------------------------------------------------------- #

def test_scheme_sizes_and_attention_positives():
    assert SCHEME_3CLASS.num_classes == 3
    assert SCHEME_7CLASS.num_classes == 7
    # 3-class: atypical+malignant are "needs attention".
    assert sorted(SCHEME_3CLASS.attention_positive_indices) == sorted(
        [SCHEME_3CLASS.label_to_index["atypical"], SCHEME_3CLASS.label_to_index["malignant"]]
    )
    # 7-class: FEA, ADH, DCIS, IC.
    pos = {SCHEME_7CLASS.index_to_label[i] for i in SCHEME_7CLASS.attention_positive_indices}
    assert pos == {"FEA", "ADH", "DCIS", "IC"}


def test_get_scheme_aliases():
    assert get_scheme("3class").num_classes == 3
    assert get_scheme("7-class").num_classes == 7
    assert get_scheme("5_class").num_classes == 5
    with pytest.raises(KeyError):
        get_scheme("9class")


def _bracs_manifest(n_per_subtype: int = 4) -> pd.DataFrame:
    subtypes = [
        "normal", "pathological_benign", "usual_ductal_hyperplasia",
        "flat_epithelial_atypia", "atypical_ductal_hyperplasia",
        "ductal_carcinoma_in_situ", "invasive_carcinoma",
    ]
    rows = []
    for st in subtypes:
        for i in range(n_per_subtype):
            rows.append(
                {
                    "image_id": f"bracs_{st}_{i}",
                    "image_path": "/x",
                    "patient_id": f"p_{st}_{i}",
                    "label": "placeholder",
                    "subtype": st,
                    "magnification": "unknown",
                    "cohort": "bracs",
                    "split": "train" if i < 2 else ("val" if i == 2 else "test"),
                }
            )
    return pd.DataFrame(rows)


def test_remap_to_7class_gives_native_codes():
    m = _bracs_manifest()
    out = remap_bracs_manifest(m, SCHEME_7CLASS)
    assert set(out["label"]) == {"N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"}


def test_remap_to_3class_groups_correctly():
    m = _bracs_manifest()
    out = remap_bracs_manifest(m, SCHEME_3CLASS)
    # benign = N+PB+UDH (3 subtypes x4), atypical = FEA+ADH (2x4), malignant = DCIS+IC (2x4)
    counts = out["label"].value_counts().to_dict()
    assert counts["benign"] == 12
    assert counts["atypical"] == 8
    assert counts["malignant"] == 8


def test_remap_requires_subtype_column():
    with pytest.raises(ValueError):
        remap_bracs_manifest(pd.DataFrame({"label": ["x"]}), SCHEME_3CLASS)


# --------------------------------------------------------------------------- #
# Weighted-F1
# --------------------------------------------------------------------------- #

def test_weighted_f1_present_and_correct():
    # 8 class-0 (all predicted 0) + 2 class-1 (also predicted 0).
    # class0: tp=8, fp=2, fn=0 -> P=0.8, R=1.0, F1=2*.8*1/1.8=0.8889
    # class1: tp=0, fp=0, fn=2 -> F1=0
    # weighted = (8*0.8889 + 2*0)/10 = 0.7111 ; macro = (0.8889+0)/2 = 0.4444
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 8 + [0] * 2)
    m = classification_metrics(y_true, y_pred, num_classes=3)
    assert m["weighted_f1"] == pytest.approx(0.7111, abs=1e-3)
    assert m["macro_f1"] == pytest.approx(0.4444, abs=1e-3)


# --------------------------------------------------------------------------- #
# Multi-seed runner end-to-end (synthetic, separable)
# --------------------------------------------------------------------------- #

def _seed_bracs_cache(cache_root: Path, fm_dims: dict[str, int]):
    """Class-separable BRACS embeddings with all 7 subtypes + official splits."""
    subtypes = [
        "normal", "pathological_benign", "usual_ductal_hyperplasia",
        "flat_epithelial_atypia", "atypical_ductal_hyperplasia",
        "ductal_carcinoma_in_situ", "invasive_carcinoma",
    ]
    rows_meta = []
    per = 12
    for si, st in enumerate(subtypes):
        for i in range(per):
            split = "train" if i < 8 else ("val" if i < 10 else "test")
            rows_meta.append((f"bracs_{st}_{i}", f"p_{st}_{i}", st, split, si))
    n = len(rows_meta)
    rng = np.random.default_rng(0)
    for fm, dim in fm_dims.items():
        cache = FMEmbeddingCache(cache_root, "bracs", fm)
        base = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
        for k, (_iid, _pid, _st, _sp, si) in enumerate(rows_meta):
            base[k, si % dim] += 3.0  # separable by subtype
        df = pd.DataFrame(
            {
                "image_id": [r[0] for r in rows_meta],
                "image_path": ["/x"] * n,
                "patient_id": [r[1] for r in rows_meta],
                "label": ["placeholder"] * n,
                "subtype": [r[2] for r in rows_meta],
                "magnification": ["unknown"] * n,
                "cohort": ["bracs"] * n,
                "split": [r[3] for r in rows_meta],
            }
        )
        cache.write(df, base, model_id=f"d/{fm}", embedding_dim=dim)
    return df[["image_id", "patient_id", "subtype", "split", "label", "image_path", "magnification", "cohort"]]



