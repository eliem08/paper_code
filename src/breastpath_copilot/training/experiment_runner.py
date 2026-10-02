"""Q1 experiment runners (revised design, 2026-06-02).

Two experiments that play to the cohorts' actual label spaces instead of a
degenerate 3-way leave-one-cohort-out:

* :func:`run_within_bracs` -- fine-grained 5-class early detection on BRACS
  using its official patient-disjoint train/val/test split.
* :func:`run_train_bracs_breakhis_test_bach` -- cross-cohort generalization:
  train on the pooled BRACS+BreaKHis embeddings, test on BACH (the one clean
  rotation, since the train pool covers every BACH class).

Each experiment evaluates a fixed config grid: three single-FM baselines
(each FM alone behind a linear head) and the three MoE routings over all FMs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from breastpath_copilot.training.datasets import INDEX_TO_UNIFIED_LABEL
from breastpath_copilot.training.moe_trainer import (
    EmbeddingBundle,
    MoETrainConfig,
    MoETrainer,
    build_class_weights_from_bundle,
    load_aligned_bundle,
)


def restrict_experts(bundle: EmbeddingBundle, fm_names: list[str]) -> EmbeddingBundle:
    """Return a copy of the bundle keeping only the named experts."""
    missing = [fm for fm in fm_names if fm not in bundle.expert_embeddings]
    if missing:
        raise KeyError(f"bundle is missing experts {missing}")
    return EmbeddingBundle(
        image_ids=bundle.image_ids,
        expert_embeddings={fm: bundle.expert_embeddings[fm] for fm in fm_names},
        labels=bundle.labels,
        magnifications=bundle.magnifications,
        cohorts=bundle.cohorts,
        expert_dims={fm: bundle.expert_dims[fm] for fm in fm_names},
    )


def subset_bundle(bundle: EmbeddingBundle, mask: np.ndarray) -> EmbeddingBundle:
    idx = np.where(mask)[0]
    return EmbeddingBundle(
        image_ids=[bundle.image_ids[i] for i in idx],
        expert_embeddings={fm: arr[idx] for fm, arr in bundle.expert_embeddings.items()},
        labels=bundle.labels[idx],
        magnifications=[bundle.magnifications[i] for i in idx] if bundle.magnifications is not None else None,
        cohorts=[bundle.cohorts[i] for i in idx],
        expert_dims=dict(bundle.expert_dims),
    )


def concat_bundles(bundles: list[EmbeddingBundle]) -> EmbeddingBundle:
    base_dims = bundles[0].expert_dims
    fm_names = list(base_dims)
    for b in bundles[1:]:
        if b.expert_dims != base_dims:
            raise ValueError(f"expert dim mismatch: {base_dims} vs {b.expert_dims}")
    any_mag = any(b.magnifications is not None for b in bundles)
    image_ids: list[str] = []
    cohorts: list[str] = []
    mags: list[str] = []
    labels: list[np.ndarray] = []
    experts: dict[str, list[np.ndarray]] = {fm: [] for fm in fm_names}
    for b in bundles:
        image_ids.extend(b.image_ids)
        cohorts.extend(b.cohorts)
        labels.append(b.labels)
        if any_mag:
            mags.extend(b.magnifications if b.magnifications is not None else ["unknown"] * len(b))
        for fm in fm_names:
            experts[fm].append(b.expert_embeddings[fm])
    return EmbeddingBundle(
        image_ids=image_ids,
        expert_embeddings={fm: np.concatenate(v, axis=0) for fm, v in experts.items()},
        labels=np.concatenate(labels, axis=0),
        magnifications=mags if any_mag else None,
        cohorts=cohorts,
        expert_dims=base_dims,
    )


# Config grid: (name, fm_names, routing).
def default_config_grid(all_fms: list[str]) -> list[tuple[str, list[str], str]]:
    grid: list[tuple[str, list[str], str]] = [
        (f"single:{fm}", [fm], "stacking") for fm in all_fms
    ]
    for routing in ("hard_mag", "soft_topk", "stacking"):
        grid.append((f"moe:{routing}", list(all_fms), routing))
    return grid


@dataclass
class ExperimentResult:
    experiment: str
    config_name: str
    routing: str
    fms: list[str]
    train_n: int
    test_n: int
    metrics: dict


def _flatten(exp: str, name: str, routing: str, fms: list[str], train_n: int, test_n: int, m: dict) -> dict:
    ed = m.get("early_detection", {})
    cal = m.get("calibration", {})
    sens = ed.get("sensitivity_at_specificity", {})
    row = {
        "experiment": exp,
        "config": name,
        "routing": routing,
        "n_fms": len(fms),
        "train_n": train_n,
        "test_n": test_n,
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "macro_recall": m.get("macro_recall"),
        "auroc_attention": ed.get("auroc_attention"),
        "sens@spec0.95": (sens.get("spec>=0.95") or {}).get("sensitivity"),
        "ece": cal.get("ece"),
        "ece_calibrated": cal.get("ece_calibrated"),
    }
    return row


def _train_eval_grid(
    experiment: str,
    full_train: EmbeddingBundle,
    full_test: EmbeddingBundle,
    all_fms: list[str],
    base_config: MoETrainConfig,
    *,
    val: EmbeddingBundle | None = None,
) -> tuple[list[dict], list[ExperimentResult]]:
    rows: list[dict] = []
    results: list[ExperimentResult] = []
    for name, fms, routing in default_config_grid(all_fms):
        train_b = restrict_experts(full_train, fms)
        test_b = restrict_experts(full_test, fms)
        cfg = replace(base_config, routing=routing)
        weights = (
            build_class_weights_from_bundle(train_b, INDEX_TO_UNIFIED_LABEL)
            if cfg.use_class_weights
            else None
        )
        trainer = MoETrainer(config=cfg, expert_dims=train_b.expert_dims, class_weights=weights)
        trainer.fit(train_b, val=restrict_experts(val, fms) if val is not None else None)
        metrics = trainer.evaluate(test_b, fit_temperature_on_val=cfg.calibrate)
        rows.append(_flatten(experiment, name, routing, fms, len(train_b), len(test_b), metrics))
        results.append(
            ExperimentResult(experiment, name, routing, fms, len(train_b), len(test_b), metrics)
        )
    return rows, results


def run_within_bracs(
    cache_root: str | Path,
    fm_names: list[str],
    bracs_manifest: pd.DataFrame,
    *,
    base_config: MoETrainConfig | None = None,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Experiment 1: fine-grained 5-class on BRACS official splits."""
    cfg = base_config or MoETrainConfig()
    bundle = load_aligned_bundle(cache_root, "bracs", fm_names, manifest=bracs_manifest)
    # Map image_id -> split from the manifest.
    split_by_id = dict(
        zip(bracs_manifest["image_id"].astype(str), bracs_manifest["split"].astype(str))
    )
    ids = np.array(bundle.image_ids)
    split_arr = np.array([split_by_id.get(i, "train") for i in ids])
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"
    train_b = subset_bundle(bundle, train_mask)
    val_b = subset_bundle(bundle, val_mask) if val_mask.any() else None
    test_b = subset_bundle(bundle, test_mask)
    rows, results = _train_eval_grid(
        "within_bracs_5class", train_b, test_b, fm_names, cfg, val=val_b
    )
    return pd.DataFrame(rows), results


def run_train_bracs_breakhis_test_bach(
    cache_root: str | Path,
    fm_names: list[str],
    *,
    bracs_manifest: pd.DataFrame | None = None,
    breakhis_manifest: pd.DataFrame | None = None,
    bach_manifest: pd.DataFrame | None = None,
    base_config: MoETrainConfig | None = None,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Experiment 2: train on BRACS+BreaKHis, test on BACH (clean rotation)."""
    cfg = base_config or MoETrainConfig()
    bracs = load_aligned_bundle(cache_root, "bracs", fm_names, manifest=bracs_manifest)
    breakhis = load_aligned_bundle(cache_root, "breakhis", fm_names, manifest=breakhis_manifest)
    bach = load_aligned_bundle(cache_root, "bach", fm_names, manifest=bach_manifest)
    train_b = concat_bundles([bracs, breakhis])
    rows, results = _train_eval_grid(
        "train_bracs_breakhis_test_bach", train_b, bach, fm_names, cfg
    )
    return pd.DataFrame(rows), results
