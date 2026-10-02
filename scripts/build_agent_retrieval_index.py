"""Build a patient-disjoint MoE retrieval index over BRACS TRAIN UNI embeddings
so the agent's retrieve_neighbors tool returns real similar prior cases. The
tool queries with the first FM (uni) embedding, so the index lives in that space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.core.retrieval_index import MoERetrievalIndex

CACHE = "artifacts/fm_cache"; MAN = Path("artifacts/manifests")
OUT = Path("artifacts/agent/retrieval_index.joblib")


def main():
    bracs = pd.read_parquet(MAN / "bracs.parquet")
    train_ids = bracs.loc[bracs["split"].astype(str) == "train", "image_id"].astype(str).tolist()
    cache = FMEmbeddingCache(CACHE, "bracs", "uni")
    rows, vecs = cache.slice_by_image_ids(train_ids)
    ids = rows["image_id"].astype(str).tolist()
    meta = bracs.set_index("image_id").loc[ids].reset_index()
    manifest = pd.DataFrame({
        "image_id": meta["image_id"].astype(str),
        "patient_id": meta["patient_id"].astype(str),
        "label": meta["label"].astype(str),
        "subtype": meta["subtype"].astype(str),
        "cohort": meta["cohort"].astype(str),
    })
    idx = MoERetrievalIndex(np.asarray(vecs, dtype=np.float32), manifest, metric="cosine")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    idx.save(OUT)
    print(f"saved index: {len(idx)} train cases, dim={idx.embedding_dim} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
