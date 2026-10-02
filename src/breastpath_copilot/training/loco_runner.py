"""Leave-one-cohort-out (LOCO) orchestration + ablation table.

Runs the headline Q1 experiment: for each held-out cohort and each MoE
routing strategy, train on the pooled remaining cohorts and evaluate on the
held-out one. Produces a tidy long-format ablation table (one row per
held_out x routing) with classification, calibration, and early-detection
operating-point metrics.

All embedding loading goes through cached FM features, so a full LOCO sweep
over 3 cohorts x 3 routings is fast once the caches exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from breastpath_copilot.training.datasets import (
    INDEX_TO_UNIFIED_LABEL,
    UNIFIED_LABEL_TO_INDEX,
)
from breastpath_copilot.training.moe_trainer import (
    EmbeddingBundle,
    MoETrainConfig,
    MoETrainer,
    build_class_weights_from_bundle,
    load_aligned_bundle,
)


DEFAULT_ROUTINGS = ("hard_mag", "soft_topk", "stacking")


def _subset_bundle(bundle: EmbeddingBundle, mask: np.ndarray) -> EmbeddingBundle:
    idx = np.where(mask)[0]
    return EmbeddingBundle(
        image_ids=[bundle.image_ids[i] for i in idx],
        expert_embeddings={fm: arr[idx] for fm, arr in bundle.expert_embeddings.items()},
        labels=bundle.labels[idx],
        magnifications=[bundle.magnifications[i] for i in idx] if bundle.magnifications is not None else None,
        cohorts=[bundle.cohorts[i] for i in idx],
        expert_dims=dict(bundle.expert_dims),
    )


def load_multicohort_bundle(
    cache_root: str | Path,
    cohorts: list[str],
    fm_names: list[str],
    *,
    per_cohort_manifest: dict[str, pd.DataFrame] | None = None,
) -> EmbeddingBundle:
    """Concatenate aligned bundles across cohorts into one bundle.

    Each cohort has its own per-FM cache directory (cache_root/<cohort>/<fm>).
    """
    bundles: list[EmbeddingBundle] = []
    for cohort in cohorts:
        manifest = (per_cohort_manifest or {}).get(cohort)
        bundles.append(
            load_aligned_bundle(cache_root, cohort, fm_names, manifest=manifest)
        )
    # Validate consistent expert dims across cohorts.
    base_dims = bundles[0].expert_dims
    for b in bundles[1:]:
        if b.expert_dims != base_dims:
            raise ValueError(
                f"Inconsistent expert dims across cohorts: {base_dims} vs {b.expert_dims}"
            )
    image_ids: list[str] = []
    labels_parts: list[np.ndarray] = []
    cohorts_all: list[str] = []
    mags_all: list[str] = []
    any_mag = any(b.magnifications is not None for b in bundles)
    expert_concat: dict[str, list[np.ndarray]] = {fm: [] for fm in fm_names}
    for b in bundles:
        image_ids.extend(b.image_ids)
        labels_parts.append(b.labels)
        cohorts_all.extend(b.cohorts)
        if any_mag:
            mags_all.extend(b.magnifications if b.magnifications is not None else ["unknown"] * len(b))
        for fm in fm_names:
            expert_concat[fm].append(b.expert_embeddings[fm])
    return EmbeddingBundle(
        image_ids=image_ids,
        expert_embeddings={fm: np.concatenate(parts, axis=0) for fm, parts in expert_concat.items()},
        labels=np.concatenate(labels_parts, axis=0),
        magnifications=mags_all if any_mag else None,
        cohorts=cohorts_all,
        expert_dims=base_dims,
    )


@dataclass
class LOCOResult:
    held_out_cohort: str
    routing: str
    train_n: int
    test_n: int
    metrics: dict


def _flatten_metrics(held_out: str, routing: str, train_n: int, test_n: int, metrics: dict) -> dict:
    ed = metrics.get("early_detection", {})
    cal = metrics.get("calibration", {})
    sens_block = ed.get("sensitivity_at_specificity", {})
    row = {
        "held_out_cohort": held_out,
        "routing": routing,
        "train_n": train_n,
        "test_n": test_n,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "auroc_attention": ed.get("auroc_attention"),
        "ece": cal.get("ece"),
        "brier": cal.get("brier"),
        "ece_calibrated": cal.get("ece_calibrated"),
        "temperature": cal.get("temperature"),
    }
    for key, block in sens_block.items():
        row[f"sens@{key}"] = block.get("sensitivity")
    return row


def run_loco_ablation(
    cache_root: str | Path,
    cohorts: list[str],
    fm_names: list[str],
    *,
    routings: Iterable[str] = DEFAULT_ROUTINGS,
    base_config: MoETrainConfig | None = None,
    output_dir: str | Path | None = None,
    per_cohort_manifest: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[LOCOResult]]:
    """Run the full LOCO x routing sweep. Returns (ablation_df, raw_results)."""
    routings = list(routings)
    full_bundle = load_multicohort_bundle(
        cache_root, cohorts, fm_names, per_cohort_manifest=per_cohort_manifest
    )
    cohort_arr = np.array(full_bundle.cohorts)

    rows: list[dict] = []
    results: list[LOCOResult] = []
    for held_out in cohorts:
        test_mask = cohort_arr == held_out
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        train_bundle = _subset_bundle(full_bundle, train_mask)
        test_bundle = _subset_bundle(full_bundle, test_mask)
        class_weights = build_class_weights_from_bundle(train_bundle, INDEX_TO_UNIFIED_LABEL)

        for routing in routings:
            cfg = _clone_config(base_config, routing)
            trainer = MoETrainer(
                config=cfg,
                expert_dims=full_bundle.expert_dims,
                class_weights=class_weights if cfg.use_class_weights else None,
            )
            trainer.fit(train_bundle, val=None)
            metrics = trainer.evaluate(test_bundle, fit_temperature_on_val=cfg.calibrate)
            results.append(
                LOCOResult(held_out, routing, len(train_bundle), len(test_bundle), metrics)
            )
            rows.append(_flatten_metrics(held_out, routing, len(train_bundle), len(test_bundle), metrics))
            if output_dir is not None:
                ckpt = Path(output_dir) / f"moe_{routing}_holdout_{held_out}.pt"
                trainer.save(ckpt)

    df = pd.DataFrame(rows)
    if output_dir is not None and not df.empty:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "ablation_table.csv", index=False)
    return df, results


def _clone_config(base: MoETrainConfig | None, routing: str) -> MoETrainConfig:
    if base is None:
        return MoETrainConfig(routing=routing)
    data = dict(base.__dict__)
    data["routing"] = routing
    return MoETrainConfig(**data)
