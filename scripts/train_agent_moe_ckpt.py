"""Train a 3-class BRACS MoE head and save its raw state_dict so the agent's
moe_forward_tool can load it (it calls head.load_state_dict(torch.load(path)))."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest
from breastpath_copilot.training.experiment_runner import subset_bundle
from breastpath_copilot.training.moe_trainer import (MoETrainConfig, MoETrainer,
                                                     build_class_weights_from_bundle, load_aligned_bundle)

CACHE = Path("artifacts/fm_cache"); MAN = Path("artifacts/manifests")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
OUT = Path("artifacts/agent"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sch = get_scheme("3class")
    m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    b = load_aligned_bundle(CACHE, "bracs", FMS, manifest=m, label_to_index=sch.label_to_index)
    s = dict(zip(m["image_id"].astype(str), m["split"].astype(str)))
    a = np.array([s.get(i, "train") for i in b.image_ids])
    b_tr = subset_bundle(b, a == "train")
    w = build_class_weights_from_bundle(b_tr, sch.index_to_label, sch.label_to_index).to(dev)
    cfg = MoETrainConfig(routing="soft_topk", shared_dim=512, top_k=2, epochs=120,
                         lr=1e-3, loss_kind="ce", num_classes=3, seed=0, device=dev)
    moe = MoETrainer(config=cfg, expert_dims=b_tr.expert_dims, class_weights=w)
    moe.fit(b_tr)
    ckpt = OUT / "moe_bracs_3class.pt"
    torch.save(moe.model.state_dict(), ckpt)  # raw state_dict for moe_forward_tool
    print(f"saved {ckpt} | expert_dims={b_tr.expert_dims} | classes={sch.index_to_label}", flush=True)


if __name__ == "__main__":
    main()
