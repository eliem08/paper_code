"""Tests for the two-experiment Q1 runner (synthetic caches, CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402
from breastpath_copilot.training.datasets import UNIFIED_LABEL_TO_INDEX  # noqa: E402
from breastpath_copilot.training.experiment_runner import (  # noqa: E402
    concat_bundles,
    default_config_grid,
    restrict_experts,
    run_train_bracs_breakhis_test_bach,
    run_within_bracs,
    subset_bundle,
)
from breastpath_copilot.training.moe_trainer import MoETrainConfig, load_aligned_bundle  # noqa: E402


LABEL_NAMES = {v: k for k, v in UNIFIED_LABEL_TO_INDEX.items()}


def _seed_cohort(
    cache_root: Path,
    cohort: str,
    fm_dims: dict[str, int],
    labels: list[str],
    *,
    seed: int,
    splits: list[str] | None = None,
):
    n = len(labels)
    rng = np.random.default_rng(seed)
    ids = [f"{cohort}_{i}" for i in range(n)]
    mags = rng.choice(["40X", "100X", "200X", "400X"], size=n)
    label_idx = np.array([UNIFIED_LABEL_TO_INDEX[l] for l in labels])
    for fm, dim in fm_dims.items():
        cache = FMEmbeddingCache(cache_root, cohort, fm)
        base = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
        for i, li in enumerate(label_idx):
            base[i, li % dim] += 3.0  # class-separable signal
        cols = {
            "image_id": ids,
            "image_path": ["/x"] * n,
            "patient_id": [f"{cohort}_p{i}" for i in range(n)],
            "label": labels,
            "subtype": [f"sub_{l}" for l in labels],
            "magnification": mags,
            "cohort": [cohort] * n,
        }
        if splits is not None:
            cols["split"] = splits
        cache.write(cache_root and pd.DataFrame(cols), base, model_id=f"d/{fm}", embedding_dim=dim)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_default_config_grid_shape():
    grid = default_config_grid(["uni", "virchow2", "phikon_v2"])
    names = [g[0] for g in grid]
    assert names == [
        "single:uni",
        "single:virchow2",
        "single:phikon_v2",
        "moe:hard_mag",
        "moe:soft_topk",
        "moe:stacking",
    ]
    # single configs carry exactly one fm; moe configs carry all three.
    assert all(len(g[1]) == 1 for g in grid[:3])
    assert all(len(g[1]) == 3 for g in grid[3:])


def test_restrict_and_subset_and_concat(tmp_path: Path):
    cache_root = tmp_path / "cache"
    labels = ["benign", "invasive", "dcis", "normal"]
    _seed_cohort(cache_root, "bracs", {"fm_a": 8, "fm_b": 12}, labels, seed=0)
    bundle = load_aligned_bundle(cache_root, "bracs", ["fm_a", "fm_b"])

    one = restrict_experts(bundle, ["fm_a"])
    assert set(one.expert_embeddings) == {"fm_a"}
    assert len(one) == len(bundle)

    half = subset_bundle(bundle, np.array([True, False, True, False]))
    assert len(half) == 2

    cat = concat_bundles([half, half])
    assert len(cat) == 4
    assert cat.expert_dims == bundle.expert_dims


# --------------------------------------------------------------------------- #
# Experiment 1: within-BRACS
# --------------------------------------------------------------------------- #

def test_run_within_bracs_uses_official_split(tmp_path: Path):
    cache_root = tmp_path / "cache"
    # 5 classes x 6 each, deterministic splits.
    labels, splits = [], []
    for lab in ["normal", "benign", "atypia", "dcis", "invasive"]:
        labels += [lab] * 6
        splits += ["train", "train", "train", "train", "val", "test"]
    _seed_cohort(cache_root, "bracs", {"uni": 8, "virchow2": 8, "phikon_v2": 8}, labels, seed=1, splits=splits)
    manifest = FMEmbeddingCache(cache_root, "bracs", "uni").load_manifest()
    # The manifest written by the cache carries the split column.
    assert "split" in manifest.columns

    cfg = MoETrainConfig(shared_dim=16, epochs=5, device="cpu")
    df, results = run_within_bracs(cache_root, ["uni", "virchow2", "phikon_v2"], manifest, base_config=cfg)
    # 3 single-FM + 3 MoE configs = 6 rows.
    assert len(df) == 6
    assert set(df["experiment"]) == {"within_bracs_5class"}
    # test split has 1 per class = 5 cases.
    assert (df["test_n"] == 5).all()
    assert {"accuracy", "auroc_attention", "ece"}.issubset(df.columns)


# --------------------------------------------------------------------------- #
# Experiment 2: BACH external
# --------------------------------------------------------------------------- #

def test_run_bach_external_train_pool_covers_bach_classes(tmp_path: Path):
    cache_root = tmp_path / "cache"
    dims = {"uni": 8, "virchow2": 8, "phikon_v2": 8}
    # BRACS: all 5 classes. BreaKHis: benign/invasive. BACH: normal/benign/dcis/invasive.
    _seed_cohort(cache_root, "bracs", dims, ["normal", "benign", "atypia", "dcis", "invasive"] * 4, seed=2)
    _seed_cohort(cache_root, "breakhis", dims, ["benign", "invasive"] * 10, seed=3)
    _seed_cohort(cache_root, "bach", dims, ["normal", "benign", "dcis", "invasive"] * 3, seed=4)

    cfg = MoETrainConfig(shared_dim=16, epochs=5, device="cpu")
    df, results = run_train_bracs_breakhis_test_bach(
        cache_root, ["uni", "virchow2", "phikon_v2"], base_config=cfg
    )
    assert len(df) == 6
    assert set(df["experiment"]) == {"train_bracs_breakhis_test_bach"}
    # train_n = BRACS(20) + BreaKHis(20) = 40; test_n = BACH(12).
    assert (df["train_n"] == 40).all()
    assert (df["test_n"] == 12).all()
    # AUROC for the binary attention collapse should be computable.
    assert df["auroc_attention"].notna().all()


def test_within_bracs_moe_beats_chance_on_separable_data(tmp_path: Path):
    cache_root = tmp_path / "cache"
    labels, splits = [], []
    for lab in ["normal", "benign", "atypia", "dcis", "invasive"]:
        labels += [lab] * 20
        splits += ["train"] * 16 + ["val"] * 2 + ["test"] * 2
    _seed_cohort(cache_root, "bracs", {"uni": 8, "virchow2": 8, "phikon_v2": 8}, labels, seed=5, splits=splits)
    manifest = FMEmbeddingCache(cache_root, "bracs", "uni").load_manifest()
    cfg = MoETrainConfig(shared_dim=32, epochs=40, lr=5e-3, device="cpu")
    df, _ = run_within_bracs(cache_root, ["uni", "virchow2", "phikon_v2"], manifest, base_config=cfg)
    # On class-separable embeddings, the soft_topk MoE should clear chance (0.2).
    moe_row = df[df["config"] == "moe:soft_topk"].iloc[0]
    assert moe_row["accuracy"] > 0.5
