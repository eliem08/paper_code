"""Real ensemble-size curve for the 7-class MoE deep ensemble: train the same
5 seeds used elsewhere in the paper (Table 8), then compute weighted-F1 for
every non-empty subset of size S=1..5 of those 5 seeds' test-set probability
vectors, averaged by simple probability averaging. Reports mean +/- std wF1
per subset size S, so the paper's ensemble-size claim is backed by an actual
curve rather than asserted from 5 individual seeds + the full ensemble alone.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = 7
    sch = get_scheme(f"{nc}class")
    bracs_m = pd.read_parquet(MANIFESTS / "bracs.parquet")
    remapped = remap_bracs_manifest(bracs_m, sch)
    bundle = load_aligned_bundle(CACHE, "bracs", FMS, manifest=remapped, label_to_index=sch.label_to_index)
    tr, va, te = _split_masks(bundle, remapped)
    b_tr, b_te = subset_bundle(bundle, tr), subset_bundle(bundle, te)
    y_te = np.asarray(b_te.labels)

    probs_te = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                             lr=1e-3, loss_kind="ce", num_classes=nc, seed=seed, device=device)
        w = build_class_weights_from_bundle(b_tr, sch.index_to_label, sch.label_to_index)
        moe = MoETrainer(config=cfg, expert_dims=b_tr.expert_dims, class_weights=w)
        moe.fit(b_tr)
        lt = moe.predict_logits(b_te)
        pt = F.softmax(torch.from_numpy(lt), 1).numpy()
        probs_te.append(pt)
        wf1 = f1_score(y_te, pt.argmax(1), average="weighted")
        print(f"seed {seed}: wF1 {wf1:.4f}", flush=True)

    curve = {}
    for s in range(1, 6):
        vals = []
        for combo in itertools.combinations(range(5), s):
            p = np.mean([probs_te[i] for i in combo], axis=0)
            vals.append(f1_score(y_te, p.argmax(1), average="weighted"))
        curve[s] = {"mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n_subsets": len(vals),
                    "values": [round(float(v), 4) for v in vals]}
        print(f"S={s}: wF1 {curve[s]['mean']:.4f} +/- {curve[s]['std']:.4f} "
              f"over {curve[s]['n_subsets']} subsets", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ensemble_size_curve_7class.json", "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\nwrote -> {OUT / 'ensemble_size_curve_7class.json'}", flush=True)


if __name__ == "__main__":
    main()
