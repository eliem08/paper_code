"""In-domain BACH 4-class evaluation of the MoE.

BACH (ICIAR-2018) has 400 microscopy ROIs, 100 each of normal/benign/in-situ/
invasive, with no official split -> stratified 5-fold CV. MoE (soft_topk) over
frozen UNI + Virchow2 + Phikon-v2, 3-seed deep ensemble per fold. Reports
pooled out-of-fold accuracy / weighted-F1 / macro-AUROC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (MoETrainConfig, MoETrainer,
                                                     build_class_weights_from_bundle, load_aligned_bundle)

CACHE = Path("artifacts/fm_cache")
FMS = ["uni", "virchow2", "phikon_v2"]
L2I = {"normal": 0, "benign": 1, "dcis": 2, "invasive": 3}
I2L = {v: k for k, v in L2I.items()}
SEEDS = [0, 1, 2]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = pd.read_parquet(CACHE / "bach" / "uni" / "manifest.parquet")
    man = man[man["label"].astype(str).isin(L2I)].reset_index(drop=True)
    bundle = load_aligned_bundle(CACHE, "bach", FMS, manifest=man, label_to_index=L2I)
    y = np.asarray(bundle.labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros((len(y), 4), dtype=np.float32)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr_mask = np.zeros(len(y), bool); tr_mask[tr_idx] = True
        b_tr, b_te = subset_bundle(bundle, tr_mask), subset_bundle(bundle, ~tr_mask)
        w = build_class_weights_from_bundle(b_tr, I2L, L2I).to(dev)
        probs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                                 lr=1e-3, loss_kind="ce", num_classes=4, seed=seed, device=dev)
            moe = MoETrainer(config=cfg, expert_dims=b_tr.expert_dims, class_weights=w)
            moe.fit(b_tr)
            probs.append(F.softmax(torch.from_numpy(moe.predict_logits(b_te)), 1).numpy())
        p = np.mean(probs, 0)
        oof_prob[te_idx] = p; oof_pred[te_idx] = p.argmax(1)
        print(f"  fold {fold}: acc {(p.argmax(1)==y[te_idx]).mean():.4f}", flush=True)
    acc = float((oof_pred == y).mean())
    wf1 = f1_score(y, oof_pred, average="weighted")
    auroc = roc_auc_score(y, oof_prob, multi_class="ovr")
    res = (f"BACH 4-class (in-domain 5-fold CV) MoE soft_topk over {FMS}\n"
           f"acc {acc:.4f} | wF1 {wf1:.4f} | macro-AUROC {auroc:.4f}\n")
    print(res, flush=True)
    Path("artifacts/q1_results/bach_moe_kfold.txt").write_text(res)


if __name__ == "__main__":
    main()
