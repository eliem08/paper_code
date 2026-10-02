"""Compute multiclass macro one-vs-rest AUROC (the leaderboard 'AUC' metric)
for our within-BRACS models, so results compare to the wizwand BRACS
leaderboard (7-class AUC ~0.889-0.893, RoI AUROC 90.78, 3-class AUC 93.6).

Reports accuracy + weighted-F1 + macro-AUROC for frozen and hybrid FM sets,
across 3/5/7-class schemes, averaged over seeds.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import restrict_experts, subset_bundle
from breastpath_copilot.training.finetune import classification_metrics
from breastpath_copilot.training.moe_trainer import (
    MoETrainConfig,
    MoETrainer,
    build_class_weights_from_bundle,
    load_aligned_bundle,
)
from breastpath_copilot.training.threshold_metrics import multiclass_macro_auroc

CACHE = "artifacts/fm_cache"
FROZEN = ["uni", "virchow2", "phikon_v2"]
HYBRID = ["uni", "virchow2", "phikon_v2_ft"]
SEEDS = [0, 1, 2]
EPOCHS = 200
SCHEMES = ["3class", "5class", "7class"]
# Best-performing routing per the overnight run = stacking (stable, top acc).
ROUTING = "stacking"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run():
    bracs = pd.read_parquet("artifacts/manifests/bracs.parquet")
    rows = []
    for scheme_name in SCHEMES:
        scheme = get_scheme(scheme_name)
        remapped = remap_bracs_manifest(bracs, scheme)
        split = dict(zip(remapped["image_id"].astype(str), remapped["split"].astype(str)))
        for tag, fms in [("frozen", FROZEN), ("hybrid", HYBRID)]:
            bundle = load_aligned_bundle(CACHE, "bracs", fms, manifest=remapped, label_to_index=scheme.label_to_index)
            arr = np.array([split.get(i, "train") for i in bundle.image_ids])
            tr = subset_bundle(bundle, arr == "train")
            te = subset_bundle(bundle, arr == "test")
            accs, wf1s, aurocs = [], [], []
            for seed in SEEDS:
                cfg = MoETrainConfig(routing=ROUTING, shared_dim=512, top_k=2,
                                     epochs=EPOCHS, lr=1e-3, num_classes=scheme.num_classes,
                                     seed=seed, device="auto")
                w = build_class_weights_from_bundle(tr, scheme.index_to_label, scheme.label_to_index)
                tr2 = restrict_experts(tr, fms)
                te2 = restrict_experts(te, fms)
                trainer = MoETrainer(config=cfg, expert_dims=tr2.expert_dims, class_weights=w,
                                     attention_positive_indices=scheme.attention_positive_indices)
                trainer.fit(tr2, val=None)
                logits = trainer.predict_logits(te2)
                probs = _softmax(logits)
                preds = probs.argmax(1)
                m = classification_metrics(te2.labels, preds, scheme.num_classes)
                auc = multiclass_macro_auroc(probs, te2.labels, scheme.num_classes)
                accs.append(m["accuracy"]); wf1s.append(m["weighted_f1"]); aurocs.append(auc["macro_auroc"])
            rows.append({
                "scheme": scheme_name, "fmset": tag, "routing": ROUTING,
                "acc_mean": mean(accs), "acc_std": pstdev(accs) if len(accs) > 1 else 0.0,
                "wf1_mean": mean(wf1s),
                "macro_auroc_mean": mean(aurocs), "macro_auroc_std": pstdev(aurocs) if len(aurocs) > 1 else 0.0,
            })
            print(f"[{scheme_name}/{tag}] acc {rows[-1]['acc_mean']:.3f}  wF1 {rows[-1]['wf1_mean']:.3f}  "
                  f"macroAUROC {rows[-1]['macro_auroc_mean']:.4f}±{rows[-1]['macro_auroc_std']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path("artifacts/q1_results/bracs_macro_auroc.csv")
    df.to_csv(out, index=False)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    run()
