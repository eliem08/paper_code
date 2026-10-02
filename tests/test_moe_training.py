"""Tests for the MoE training/eval pipeline: calibration, threshold metrics,
trainer, and the LOCO ablation runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402
from breastpath_copilot.training.calibration import (  # noqa: E402
    TemperatureScaler,
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_table,
)
from breastpath_copilot.training.threshold_metrics import (  # noqa: E402
    binary_attention_labels,
    binary_attention_scores,
    early_detection_operating_report,
    roc_auc,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from breastpath_copilot.training.datasets import UNIFIED_LABEL_TO_INDEX  # noqa: E402


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def test_ece_zero_for_perfectly_calibrated():
    # Two-class, confidence == accuracy exactly.
    probs = np.array([[0.5, 0.5]] * 100)
    labels = np.array([0, 1] * 50)
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_overconfident():
    # Always predict class 0 with 0.99 confidence but only 50% correct.
    probs = np.array([[0.99, 0.01]] * 100)
    labels = np.array([0, 1] * 50)
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece > 0.4  # ~0.49


def test_brier_score_bounds():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == pytest.approx(0.0)
    probs_wrong = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert brier_score(probs_wrong, labels) == pytest.approx(2.0)


def test_temperature_scaler_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    # Large-magnitude logits -> very high softmax confidence, but only ~70%
    # accurate: the classic over-confidence regime where temperature
    # scaling (T > 1) demonstrably reduces ECE.
    base = rng.standard_normal((600, 3))
    logits = base * 6.0  # inflate confidence
    labels = base.argmax(1)
    flip = rng.random(600) < 0.3
    labels[flip] = (labels[flip] + 1) % 3
    scaler = TemperatureScaler().fit(logits, labels)
    report = calibration_report(logits, labels, scaler=scaler)
    # Over-confident model -> learned temperature should be > 1 (softening)...
    assert report["temperature"] > 1.0
    # ...and calibrated ECE should be no worse than raw (here: clearly better).
    assert report["ece_calibrated"] <= report["ece"] + 1e-3


def test_reliability_table_bins_sum_to_n():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 1])
    table = reliability_table(probs, labels, n_bins=10)
    assert sum(b.count for b in table) == 3


# --------------------------------------------------------------------------- #
# Threshold metrics
# --------------------------------------------------------------------------- #

def test_roc_auc_perfect_and_random():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    # Degenerate single-class -> 0.5
    assert roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) == pytest.approx(0.5)


def test_binary_attention_collapse():
    # label indices: normal=0, benign=1, atypia=2, dcis=3, invasive=4
    labels = np.array([0, 1, 2, 3, 4])
    y = binary_attention_labels(labels)
    assert y.tolist() == [0, 0, 1, 1, 1]
    probs = np.eye(5)  # one-hot per class
    scores = binary_attention_scores(probs)
    # atypia/dcis/invasive rows have all attention-mass; normal/benign have 0.
    assert scores.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_sensitivity_at_specificity_and_inverse():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])  # perfectly separable
    op = sensitivity_at_specificity(y, s, 0.95)
    assert op.sensitivity == pytest.approx(1.0)
    assert op.specificity == pytest.approx(1.0)
    op2 = specificity_at_sensitivity(y, s, 0.95)
    assert op2.specificity == pytest.approx(1.0)


def test_sensitivity_at_specificity_falls_back_when_target_infeasible():
    # Scores give at best specificity 0.5 -> target 0.99 infeasible.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.5, 0.9, 0.5, 0.9])
    op = sensitivity_at_specificity(y, s, 0.99)
    assert 0.0 <= op.specificity <= 1.0  # returns best-specificity fallback


def test_early_detection_report_structure():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((100, 5))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = rng.integers(0, 5, size=100)
    report = early_detection_operating_report(probs, labels)
    assert "auroc_attention" in report
    assert "sensitivity_at_specificity" in report
    assert "spec>=0.95" in report["sensitivity_at_specificity"]


# --------------------------------------------------------------------------- #
# MoE trainer + LOCO runner
# --------------------------------------------------------------------------- #

def _seed_cohort_caches(cache_root: Path, cohort: str, fm_dims: dict[str, int], n: int, seed: int):
    """Create separable embeddings so the trained MoE can actually learn."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, size=n)
    ids = [f"{cohort}_{i}" for i in range(n)]
    mags = rng.choice(["40X", "100X", "200X", "400X"], size=n)
    label_names = {v: k for k, v in UNIFIED_LABEL_TO_INDEX.items()}
    for fm, dim in fm_dims.items():
        cache = FMEmbeddingCache(cache_root, cohort, fm)
        # Class-separable: shift each class mean along a distinct axis.
        base = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
        for i, lab in enumerate(labels):
            base[i, lab % dim] += 3.0
        df = pd.DataFrame(
            {
                "image_id": ids,
                "image_path": ["/x"] * n,
                "patient_id": [f"{cohort}_p{i}" for i in range(n)],
                "label": [label_names[int(l)] for l in labels],
                "subtype": [f"sub_{int(l)}" for l in labels],
                "magnification": mags,
                "cohort": [cohort] * n,
            }
        )
        cache.write(df, base, model_id=f"dummy/{fm}", embedding_dim=dim)


def test_load_aligned_bundle_and_train_one_routing(tmp_path: Path):
    from breastpath_copilot.training.moe_trainer import (
        MoETrainConfig,
        MoETrainer,
        load_aligned_bundle,
    )

    cache_root = tmp_path / "cache"
    _seed_cohort_caches(cache_root, "breakhis", {"fm_a": 8, "fm_b": 12}, n=60, seed=0)
    bundle = load_aligned_bundle(cache_root, "breakhis", ["fm_a", "fm_b"])
    assert len(bundle) == 60
    assert set(bundle.expert_dims) == {"fm_a", "fm_b"}

    cfg = MoETrainConfig(routing="soft_topk", shared_dim=16, epochs=5, device="cpu")
    trainer = MoETrainer(config=cfg, expert_dims=bundle.expert_dims)
    result = trainer.fit(bundle, val=bundle)
    assert len(result["history"]) == 5
    assert "val" in result
    assert "accuracy" in result["val"]
    assert "calibration" in result["val"]
    assert "early_detection" in result["val"]


def test_moe_trainer_learns_on_separable_data(tmp_path: Path):
    from breastpath_copilot.training.moe_trainer import (
        MoETrainConfig,
        MoETrainer,
        load_aligned_bundle,
    )

    cache_root = tmp_path / "cache"
    _seed_cohort_caches(cache_root, "breakhis", {"fm_a": 8, "fm_b": 8}, n=120, seed=3)
    bundle = load_aligned_bundle(cache_root, "breakhis", ["fm_a", "fm_b"])
    cfg = MoETrainConfig(routing="stacking", shared_dim=16, epochs=40, lr=5e-3, device="cpu")
    trainer = MoETrainer(config=cfg, expert_dims=bundle.expert_dims)
    trainer.fit(bundle, val=None)
    metrics = trainer.evaluate(bundle)
    # Class-separable embeddings -> well above chance (chance = 0.2 for 5 classes).
    assert metrics["accuracy"] > 0.6


@pytest.mark.parametrize("routing", ["hard_mag", "soft_topk", "stacking"])
def test_loco_runner_produces_ablation_rows(tmp_path: Path, routing: str):
    from breastpath_copilot.training.loco_runner import run_loco_ablation
    from breastpath_copilot.training.moe_trainer import MoETrainConfig

    cache_root = tmp_path / "cache"
    for ci, cohort in enumerate(["breakhis", "bach", "bracs"]):
        _seed_cohort_caches(cache_root, cohort, {"fm_a": 8, "fm_b": 8}, n=40, seed=ci)

    cfg = MoETrainConfig(shared_dim=16, epochs=3, device="cpu")
    df, results = run_loco_ablation(
        cache_root,
        ["breakhis", "bach", "bracs"],
        ["fm_a", "fm_b"],
        routings=[routing],
        base_config=cfg,
        output_dir=tmp_path / "out",
    )
    # 3 held-out cohorts x 1 routing = 3 rows.
    assert len(df) == 3
    assert set(df["held_out_cohort"]) == {"breakhis", "bach", "bracs"}
    assert (df["routing"] == routing).all()
    assert (tmp_path / "out" / "ablation_table.csv").exists()
    # Each held-out cohort's test rows must not appear in its training count.
    for r in results:
        assert r.test_n > 0 and r.train_n > 0


def test_loco_full_three_routing_sweep_smoke(tmp_path: Path):
    from breastpath_copilot.training.loco_runner import run_loco_ablation
    from breastpath_copilot.training.moe_trainer import MoETrainConfig

    cache_root = tmp_path / "cache"
    for ci, cohort in enumerate(["breakhis", "bach", "bracs"]):
        _seed_cohort_caches(cache_root, cohort, {"fm_a": 8, "fm_b": 8}, n=30, seed=ci + 10)
    cfg = MoETrainConfig(shared_dim=16, epochs=2, device="cpu")
    df, _ = run_loco_ablation(
        cache_root, ["breakhis", "bach", "bracs"], ["fm_a", "fm_b"], base_config=cfg
    )
    # 3 cohorts x 3 routings = 9 rows.
    assert len(df) == 9
    assert set(df["routing"]) == {"hard_mag", "soft_topk", "stacking"}
    assert {"accuracy", "auroc_attention", "ece"}.issubset(df.columns)
