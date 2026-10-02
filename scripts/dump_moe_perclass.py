"""Reproduce the headline MoE-only deep ensemble and dump per-class F1,
confusion matrix, per-seed wF1 and ECE (pre/post temperature) for the paper.

Uses ONLY cached FM embeddings (uni, virchow2, phikon_v2_ft) -- same config as
scripts/run_supremacy_benchmark.py (soft_topk, shared_dim=512, top_k=2,
epochs=120, lr=1e-3, ce, 5 seeds). No graphs, no fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (
    MoETrainConfig,
    MoETrainer,
    build_class_weights_from_bundle,
    load_aligned_bundle,
)

MANIFESTS = Path("artifacts/manifests")
CACHE = Path("artifacts/fm_cache")
OUT = Path("artifacts/q1_results")
SEEDS = [0, 1, 2, 3, 4]
FMS = ["uni", "virchow2", "phikon_v2_ft"]


def _split_masks(bundle, manifest):
    split_by_id = dict(zip(manifest["image_id"].astype(str), manifest["split"].astype(str)))
    arr = np.array([split_by_id.get(i, "train") for i in bundle.image_ids])
    return arr == "train", arr == "val", arr == "test"


def ece(probs, y, n_bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def fit_temperature(logits, y):
    """Single-temperature scaling on validation logits."""
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=200)
    lg = torch.from_numpy(logits).float()
    yt = torch.from_numpy(y).long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / t.clamp(min=1e-2), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=1e-2).item())


def run(nc, device):
    sch = get_scheme(f"{nc}class")
    bracs_m = pd.read_parquet(MANIFESTS / "bracs.parquet")
    remapped = remap_bracs_manifest(bracs_m, sch)
    bundle = load_aligned_bundle(CACHE, "bracs", FMS, manifest=remapped, label_to_index=sch.label_to_index)
    tr, va, te = _split_masks(bundle, remapped)
    b_tr, b_va, b_te = subset_bundle(bundle, tr), subset_bundle(bundle, va), subset_bundle(bundle, te)
    y_te, y_va = np.asarray(b_te.labels), np.asarray(b_va.labels)

    probs_te, logits_te, logits_va, per_seed_wf1 = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=nc, seed=seed, device=device)
        w = build_class_weights_from_bundle(b_tr, sch.index_to_label, sch.label_to_index)
        moe = MoETrainer(config=cfg, expert_dims=b_tr.expert_dims, class_weights=w)
        moe.fit(b_tr)
        lt = moe.predict_logits(b_te)
        lv = moe.predict_logits(b_va)
        logits_te.append(lt)
        logits_va.append(lv)
        pt = F.softmax(torch.from_numpy(lt), 1).numpy()
        probs_te.append(pt)
        per_seed_wf1.append(f1_score(y_te, pt.argmax(1), average="weighted"))
        print(f"[{nc}c] seed {seed}: wF1 {per_seed_wf1[-1]:.4f}", flush=True)

    p_te = np.mean(probs_te, axis=0)
    pred = p_te.argmax(1)
    labels = [sch.index_to_label[i] for i in range(nc)]

    wf1 = f1_score(y_te, pred, average="weighted")
    acc = float((pred == y_te).mean())
    auroc = roc_auc_score(y_te, p_te, multi_class="ovr")
    prec, rec, f1, sup = precision_recall_fscore_support(y_te, pred, labels=list(range(nc)), zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=list(range(nc)))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    # calibration: ensemble logits (mean), temperature from val
    mean_logits_va = np.mean(logits_va, axis=0)
    mean_logits_te = np.mean(logits_te, axis=0)
    T = fit_temperature(mean_logits_va, y_va)
    p_uncal = F.softmax(torch.from_numpy(mean_logits_te), 1).numpy()
    p_cal = F.softmax(torch.from_numpy(mean_logits_te) / T, 1).numpy()
    ece_pre, ece_post = ece(p_uncal, y_te), ece(p_cal, y_te)

    out = {
        "n_classes": nc,
        "labels": labels,
        "support": [int(s) for s in sup],
        "per_seed_wf1": [round(float(x), 4) for x in per_seed_wf1],
        "ensemble": {"wF1": round(float(wf1), 4), "acc": round(acc, 4), "auroc": round(float(auroc), 4)},
        "per_class": {labels[i]: {"precision": round(float(prec[i]), 3),
                                  "recall": round(float(rec[i]), 3),
                                  "f1": round(float(f1[i]), 3),
                                  "support": int(sup[i])} for i in range(nc)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rownorm": np.round(cm_norm, 3).tolist(),
        "temperature": round(T, 3),
        "ece": {"uncalibrated": round(ece_pre, 3), "calibrated": round(ece_post, 3)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"moe_perclass_{nc}class.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for nc in (7, 3):
        run(nc, device)


if __name__ == "__main__":
    main()
