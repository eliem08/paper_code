"""Uncertainty-aware MoE: the engine emits not just a class
distribution but two principled uncertainty signals --
 (i) epistemic mutual information from deep-ensemble disagreement (BALD):
       MI = H(mean_s p_s) - mean_s H(p_s)
 (ii) routing entropy of the averaged top-k gate weights.
We compare deferral policies (defer the least-confident / most-uncertain cases)
on the risk-coverage curve to show uncertainty-aware deferral beats plain
softmax confidence.
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


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis)


def gate_weights(model, bundle, dev, bs=256):
    """Run the MoE head and collect per-case gate weights (B, M)."""
    model.eval(); outs = []
    n = len(bundle)
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        experts = {fm: torch.as_tensor(bundle.expert_embeddings[fm][idx], dtype=torch.float32, device=dev)
                   for fm in bundle.expert_dims}
        with torch.no_grad():
            _, out = model(experts, magnifications=None)
        outs.append(out.gate_weights.cpu().numpy())
    return np.concatenate(outs, 0)


def acc_at(correct, score_keep_high, covs):
    """Accuracy on retained when keeping the top-coverage by score (higher=keep)."""
    order = np.argsort(-score_keep_high)
    res = {}
    for c in covs:
        k = max(1, int(round(c * len(correct))))
        res[c] = float(correct[order[:k]].mean())
    return res


def run(nc, dev):
    sch = get_scheme(f"{nc}class"); m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    b = load_aligned_bundle(CACHE, "bracs", FMS, manifest=m, label_to_index=sch.label_to_index)
    s = dict(zip(m["image_id"].astype(str), m["split"].astype(str)))
    a = np.array([s.get(i, "train") for i in b.image_ids])
    tr, te = subset_bundle(b, a == "train"), subset_bundle(b, a == "test")
    w = build_class_weights_from_bundle(tr, sch.index_to_label, sch.label_to_index).to(dev)
    yte = np.asarray(te.labels)
    P, G = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=nc, seed=seed, device=dev)
        moe = MoETrainer(config=cfg, expert_dims=tr.expert_dims, class_weights=w)
        moe.fit(tr)
        P.append(F.softmax(torch.from_numpy(moe.predict_logits(te)), 1).numpy())
        G.append(gate_weights(moe.model, te, dev))
    P = np.stack(P)                       # (S, N, C)
    ens = P.mean(0)                       # (N, C)
    pred = ens.argmax(1); correct = (pred == yte).astype(float)
    conf = ens.max(1)                     # softmax confidence
    mi = entropy(ens) - entropy(P, axis=2).mean(0)   # epistemic (BALD)
    gate_ent = entropy(np.stack(G).mean(0))          # routing entropy
    # hybrid: rank-average of (high conf) and (low mi)
    def rank(x):
        r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r / (len(x) - 1)
    hybrid = rank(conf) + (1 - rank(mi))
    covs = [1.0, 0.9, 0.8, 0.7]
    policies = {"confidence": conf, "epistemic_MI": -mi, "gate_entropy": -gate_ent, "hybrid": hybrid}
    out = {"n_classes": nc, "acc_full": round(float(correct.mean()), 4), "policies": {}}
    print(f"\n[{nc}-class]  full acc {out['acc_full']:.4f}", flush=True)
    for name, sc in policies.items():
        at = acc_at(correct, sc, covs)
        out["policies"][name] = {str(c): round(at[c], 4) for c in covs}
        print(f"  defer by {name:13s}: " + "  ".join(f"{int(c*100)}%={at[c]:.4f}" for c in covs), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"uncertainty_deferral_{nc}class.json").write_text(json.dumps(out, indent=2))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for nc in (7, 3):
        run(nc, dev)


if __name__ == "__main__":
    main()
