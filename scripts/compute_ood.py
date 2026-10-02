"""Out-of-distribution detection: does the uncertainty-aware engine
know when a case is unlike its BRACS training data? We train the BRACS 7-class
MoE, then measure its epistemic uncertainty (ensemble mutual information) and
softmax confidence on in-distribution BRACS-test vs out-of-distribution BACH and
BreaKHis. High uncertainty on OOD => the agent escalates it. We report the AUROC
of OOD-vs-ID separation for each signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (MoETrainConfig, MoETrainer,
                                                     build_class_weights_from_bundle, load_aligned_bundle)

CACHE = Path("artifacts/fm_cache"); MAN = Path("artifacts/manifests"); OUT = Path("artifacts/q1_results")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
SEEDS = [0, 1, 2, 3, 4]


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis)


def load_cohort(cohort):
    """Align the 3 experts by image_id for a whole cohort (labels ignored)."""
    caches = {f: FMEmbeddingCache(CACHE, cohort, f) for f in FMS}
    base = caches[FMS[0]].load_manifest()
    ids = base["image_id"].astype(str).tolist()
    surv = None
    slices = {}
    for f in FMS:
        rows, vecs = caches[f].slice_by_image_ids(ids)
        rid = rows["image_id"].astype(str).tolist()
        slices[f] = (rid, np.asarray(vecs))
        surv = set(rid) if surv is None else (surv & set(rid))
    surv = [i for i in ids if i in surv]
    aligned = {}
    for f in FMS:
        rid, vecs = slices[f]
        pos = {i: k for k, i in enumerate(rid)}
        aligned[f] = vecs[[pos[i] for i in surv]].astype(np.float32)
    return aligned


def probs_for(models, emb, dev, bs=512):
    n = len(next(iter(emb.values())))
    per = []
    for model in models:
        model.eval(); outs = []
        for s in range(0, n, bs):
            ed = {f: torch.tensor(emb[f][s:s + bs], device=dev) for f in FMS}
            with torch.no_grad():
                logits, _ = model(ed, magnifications=None)
            outs.append(F.softmax(logits, 1).cpu().numpy())
        per.append(np.concatenate(outs, 0))
    P = np.stack(per)             # (S, N, C)
    ens = P.mean(0)
    mi = entropy(ens) - entropy(P, axis=2).mean(0)
    conf = ens.max(1)
    return mi, conf


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sch = get_scheme("7class"); m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    b = load_aligned_bundle(CACHE, "bracs", FMS, manifest=m, label_to_index=sch.label_to_index)
    s = dict(zip(m["image_id"].astype(str), m["split"].astype(str)))
    a = np.array([s.get(i, "train") for i in b.image_ids])
    tr, te = subset_bundle(b, a == "train"), subset_bundle(b, a == "test")
    w = build_class_weights_from_bundle(tr, sch.index_to_label, sch.label_to_index).to(dev)
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=7, seed=seed, device=dev)
        moe = MoETrainer(config=cfg, expert_dims=tr.expert_dims, class_weights=w)
        moe.fit(tr); models.append(moe.model)

    id_emb = {f: te.expert_embeddings[f].astype(np.float32) for f in FMS}
    mi_id, conf_id = probs_for(models, id_emb, dev)
    res = {"id_cohort": "bracs-test", "n_id": int(len(mi_id)), "ood": {}}
    print(f"ID bracs-test: MI mean {mi_id.mean():.4f}  conf mean {conf_id.mean():.4f}  (n={len(mi_id)})", flush=True)
    for cohort in ("bach", "breakhis"):
        emb = load_cohort(cohort)
        mi_o, conf_o = probs_for(models, emb, dev)
        y = np.r_[np.zeros(len(mi_id)), np.ones(len(mi_o))]
        auroc_mi = roc_auc_score(y, np.r_[mi_id, mi_o])
        auroc_conf = roc_auc_score(y, np.r_[-conf_id, -conf_o])   # low conf => OOD
        res["ood"][cohort] = {"n": int(len(mi_o)),
                              "mi_mean": round(float(mi_o.mean()), 4),
                              "conf_mean": round(float(conf_o.mean()), 4),
                              "ood_auroc_mi": round(float(auroc_mi), 4),
                              "ood_auroc_conf": round(float(auroc_conf), 4)}
        print(f"OOD {cohort:9s}: MI mean {mi_o.mean():.4f}  conf mean {conf_o.mean():.4f}  "
              f"| OOD-AUROC  MI={auroc_mi:.4f}  conf={auroc_conf:.4f}  (n={len(mi_o)})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ood_detection.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
