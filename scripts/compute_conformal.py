"""Conformal prediction layer (a rigorous, distribution-free uncertainty
signal). Split-conformal (LAC) calibrated
on validation gives test prediction SETS with a coverage guarantee; set size is
a triage signal -- singletons are auto-decided, larger sets are escalated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (MoETrainConfig, MoETrainer,
                                                     build_class_weights_from_bundle, load_aligned_bundle)

CACHE = Path("artifacts/fm_cache"); MAN = Path("artifacts/manifests"); OUT = Path("artifacts/q1_results")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
SEEDS = [0, 1, 2, 3, 4]


def ens_probs(nc, dev):
    sch = get_scheme(f"{nc}class"); m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    b = load_aligned_bundle(CACHE, "bracs", FMS, manifest=m, label_to_index=sch.label_to_index)
    s = dict(zip(m["image_id"].astype(str), m["split"].astype(str)))
    a = np.array([s.get(i, "train") for i in b.image_ids])
    tr, va, te = subset_bundle(b, a == "train"), subset_bundle(b, a == "val"), subset_bundle(b, a == "test")
    w = build_class_weights_from_bundle(tr, sch.index_to_label, sch.label_to_index).to(dev)
    Pv, Pt = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=nc, seed=seed, device=dev)
        moe = MoETrainer(config=cfg, expert_dims=tr.expert_dims, class_weights=w)
        moe.fit(tr)
        Pv.append(F.softmax(torch.from_numpy(moe.predict_logits(va)), 1).numpy())
        Pt.append(F.softmax(torch.from_numpy(moe.predict_logits(te)), 1).numpy())
    return np.mean(Pv, 0), np.asarray(va.labels), np.mean(Pt, 0), np.asarray(te.labels)


def conformal(nc, dev):
    Pv, yv, Pt, yt = ens_probs(nc, dev)
    out = {"n_classes": nc, "alpha": {}}
    for alpha in (0.10, 0.05):
        scores = 1.0 - Pv[np.arange(len(yv)), yv]            # LAC nonconformity on cal
        n = len(yv)
        qlevel = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat = float(np.quantile(scores, qlevel, method="higher"))
        sets = Pt >= (1.0 - qhat)                            # boolean (N, C)
        empty = ~sets.any(1)
        if empty.any():
            sets[empty, Pt[empty].argmax(1)] = True          # never empty: keep top-1
        sizes = sets.sum(1)
        covered = sets[np.arange(len(yt)), yt]
        singleton = sizes == 1
        pred = Pt.argmax(1)
        out["alpha"][f"{alpha:.2f}"] = {
            "target_coverage": round(1 - alpha, 3),
            "empirical_coverage": round(float(covered.mean()), 4),
            "avg_set_size": round(float(sizes.mean()), 3),
            "singleton_rate": round(float(singleton.mean()), 4),
            "acc_on_singletons": round(float((pred[singleton] == yt[singleton]).mean()) if singleton.any() else 0.0, 4),
            "escalate_rate": round(float((~singleton).mean()), 4),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"conformal_{nc}class.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{nc}-class] conformal (LAC, split):", flush=True)
    for a, r in out["alpha"].items():
        print(f"  alpha={a} target={r['target_coverage']} -> cov {r['empirical_coverage']} | "
              f"set {r['avg_set_size']} | singletons {r['singleton_rate']} "
              f"(acc {r['acc_on_singletons']}) | escalate {r['escalate_rate']}", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for nc in (7, 3):
        conformal(nc, dev)


if __name__ == "__main__":
    main()
