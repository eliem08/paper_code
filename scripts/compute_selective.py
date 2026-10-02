"""Selective-prediction / deferral analysis (the reliable decision-support
angle). Using the MoE engine's
deep-ensembled, temperature-scaled test probabilities, we compute the
risk-coverage curve: as the agent defers the least-confident cases to a human,
accuracy on the retained (auto-decided) cases rises.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (MoETrainConfig, MoETrainer,
                                                     build_class_weights_from_bundle, load_aligned_bundle)

CACHE = Path("artifacts/fm_cache"); MAN = Path("artifacts/manifests"); OUT = Path("artifacts/q1_results")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
SEEDS = [0, 1, 2, 3, 4]


def fit_T(logits, y):
    t = nn.Parameter(torch.ones(1)); opt = torch.optim.LBFGS([t], lr=0.01, max_iter=200)
    lg, yt = torch.from_numpy(logits).float(), torch.from_numpy(y).long()
    def cl():
        opt.zero_grad(); loss = F.cross_entropy(lg / t.clamp(min=1e-2), yt); loss.backward(); return loss
    opt.step(cl); return float(t.detach().clamp(min=1e-2).item())


def run(nc, dev):
    sch = get_scheme(f"{nc}class"); m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    b = load_aligned_bundle(CACHE, "bracs", FMS, manifest=m, label_to_index=sch.label_to_index)
    s = dict(zip(m["image_id"].astype(str), m["split"].astype(str)))
    a = np.array([s.get(i, "train") for i in b.image_ids])
    tr, va, te = subset_bundle(b, a == "train"), subset_bundle(b, a == "val"), subset_bundle(b, a == "test")
    w = build_class_weights_from_bundle(tr, sch.index_to_label, sch.label_to_index).to(dev)
    yte, yva = np.asarray(te.labels), np.asarray(va.labels)
    lt, lv = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=nc, seed=seed, device=dev)
        moe = MoETrainer(config=cfg, expert_dims=tr.expert_dims, class_weights=w)
        moe.fit(tr)
        lt.append(moe.predict_logits(te)); lv.append(moe.predict_logits(va))
    mlt, mlv = np.mean(lt, 0), np.mean(lv, 0)
    T = fit_T(mlv, yva)
    probs = F.softmax(torch.from_numpy(mlt) / T, 1).numpy()
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == yte).astype(float)
    order = np.argsort(-conf)  # most confident first
    curve = []
    for cov in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(cov * len(yte))))
        keep = order[:k]
        acc = float(correct[keep].mean())
        curve.append({"coverage": cov, "n_auto": k, "deferred": len(yte) - k,
                      "accuracy_retained": round(acc, 4),
                      "conf_threshold": round(float(conf[order[k - 1]]), 3)})
    out = {"n_classes": nc, "temperature": round(T, 3),
           "accuracy_full": round(float(correct.mean()), 4), "risk_coverage": curve}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"selective_{nc}class.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{nc}-class] full acc {out['accuracy_full']:.4f}  (T={T:.2f})", flush=True)
    for r in curve:
        print(f"  coverage {int(r['coverage']*100):3d}%  ->  acc {r['accuracy_retained']:.4f}"
              f"  (defer {r['deferred']}, conf>={r['conf_threshold']})", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for nc in (7, 3):
        run(nc, dev)


if __name__ == "__main__":
    main()
