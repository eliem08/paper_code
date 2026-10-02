"""MCP-callable tools exposed by the pathology server.

This module is the public surface area that an external MCP client (or the
agentic harness in :mod:`breastpath_copilot.agents`) calls into. Each tool is
a typed Python function with Pydantic request / response models so it can be
schema-validated by an MCP protocol wrapper layer without changes.

Slide-level tool:
* :func:`process_slide` -- end-to-end slide-level inference.

Q1 extension tools:
* :func:`extract_fm_embeddings_tool`     -- batch FM embedding extraction.
* :func:`slice_fm_cache_tool`            -- fetch cached FM embeddings by id.
* :func:`build_unified_manifest_tool`    -- multi-cohort manifest assembly.
* :func:`cohort_split_tool`              -- leave-one-cohort-out splits.
* :func:`moe_forward_tool`               -- MoE forward pass on cached features.
* :func:`list_fm_backends_tool`          -- discovery: available FM experts.
* :func:`list_moe_routings_tool`         -- discovery: available routings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from breastpath_copilot.core.aggregation import aggregate_patch_predictions
from breastpath_copilot.core.embeddings import EmbeddingBackend, compute_patch_embeddings
from breastpath_copilot.core.explainability import generate_case_summary
from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.core.fm_embeddings import BACKEND_REGISTRY, FMBackend, build_backend
from breastpath_copilot.core.retrieval import RetrievalIndex
from breastpath_copilot.core.patching import extract_grid_patches
from breastpath_copilot.core.qc import run_qc
from breastpath_copilot.data import derive_slide_id_from_path, parse_breakhis_image_path
from breastpath_copilot.data_cohorts import (
    COHORT_BUILDERS,
    build_cross_cohort_manifest,
    leave_one_cohort_out_splits,
)
from breastpath_copilot.models.classifier import PatchClassifier
from breastpath_copilot.schemas import (
    CacheSliceRequest,
    CacheSliceResult,
    CohortSplitRequest,
    CohortSplitResult,
    FMCacheLocator,
    FMExtractRequest,
    FMExtractResult,
    MoEForwardRequest,
    MoEForwardResult,
    PatchPrediction,
    SlideInput,
    ToolListing,
)
from breastpath_copilot.utils.io import ensure_dir, save_dataframe


# --------------------------------------------------------------------------- #
# Slide-level tool
# --------------------------------------------------------------------------- #

def process_slide(
    image_path: str | Path,
    slide_id: str | None,
    output_dir: str | Path,
    class_names: list[str],
    classifier_path: str | Path | None = None,
    retrieval_index_path: str | Path | None = None,
) -> dict:
    image = Image.open(image_path).convert('RGB')
    breakhis_info = None
    try:
        breakhis_info = parse_breakhis_image_path(image_path)
    except Exception:
        pass

    effective_slide_id = slide_id or (derive_slide_id_from_path(image_path) if breakhis_info else Path(image_path).stem)
    slide = SlideInput(
        slide_path=Path(image_path),
        slide_id=effective_slide_id,
        patient_id=breakhis_info.patient_id if breakhis_info else None,
        magnification=breakhis_info.magnification if breakhis_info else '40X',
    )
    qc = run_qc(slide, image)

    patch_dir = ensure_dir(Path(output_dir) / 'patches' / effective_slide_id)
    records = extract_grid_patches(slide, image, patch_dir)

    backend = EmbeddingBackend(embedding_dim=256)
    patch_ids, vectors = compute_patch_embeddings(records, backend)

    if classifier_path and Path(classifier_path).exists() and len(vectors) > 0:
        clf = PatchClassifier.load(classifier_path)
        probas = clf.predict_proba(vectors)
        preds = clf.predict(vectors)
        patch_predictions = [
            PatchPrediction(
                slide_id=effective_slide_id,
                patch_id=patch_ids[i],
                predicted_label=str(preds[i]),
                probabilities={label: float(probas[i][j]) for j, label in enumerate(clf.model.classes_)},
            )
            for i in range(len(patch_ids))
        ]
    else:
        patch_predictions = []

    neighbors_df = pd.DataFrame()
    if retrieval_index_path and Path(retrieval_index_path).exists() and len(vectors) > 0:
        retrieval_index = RetrievalIndex.load(retrieval_index_path)
        slide_vec = vectors.mean(axis=0)
        neighbors_df = retrieval_index.query(slide_vec, top_k=5)

    slide_pred = aggregate_patch_predictions(effective_slide_id, patch_predictions, class_names)
    slide_pred.summary = generate_case_summary(slide_pred, neighbors_df)

    patch_df = pd.DataFrame([r.model_dump() for r in records])
    emb_df = pd.DataFrame(vectors) if len(vectors) > 0 else pd.DataFrame()
    if not emb_df.empty:
        emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
        emb_df.insert(0, 'patch_id', patch_ids)
        emb_df.insert(0, 'slide_id', effective_slide_id)

    meta = {}
    if breakhis_info:
        meta = {
            'label': breakhis_info.label,
            'subtype': breakhis_info.subtype,
            'magnification': breakhis_info.magnification,
            'patient_id': breakhis_info.patient_id,
        }

    return {
        'qc': qc.model_dump(),
        'slide_prediction': slide_pred.model_dump(),
        'patches': patch_df,
        'embeddings': emb_df,
        'slide_id': effective_slide_id,
        'metadata': meta,
        'neighbors': neighbors_df,
    }


# --------------------------------------------------------------------------- #
# Discovery tools
# --------------------------------------------------------------------------- #

def list_fm_backends_tool() -> list[ToolListing]:
    """Return discovery metadata for the registered FM experts."""
    return [
        ToolListing(
            name=cls.name,
            description=f"Pathology FM expert: {cls.model_id}",
            metadata={
                "model_id": cls.model_id,
                "embedding_dim": cls.embedding_dim,
                "input_size": cls.input_size,
                "requires_hf_token": cls.requires_hf_token,
                "license_note": cls.license_note,
            },
        )
        for cls in BACKEND_REGISTRY.values()
    ]


def list_moe_routings_tool() -> list[ToolListing]:
    """Return discovery metadata for the available MoE routing strategies."""
    return [
        ToolListing(
            name="hard_mag",
            description=(
                "Magnification-conditioned hard routing via a fixed lookup table. "
                "No learning; falls back to uniform for unknown magnifications."
            ),
            metadata={"learnable_router": False, "aux_loss": False},
        ),
        ToolListing(
            name="soft_topk",
            description=(
                "Learned soft routing with top-k selection (default k=2). Dropless, "
                "with a Switch-style load-balance auxiliary loss. Design follows "
                "Li et al., 'Slicing and Dicing' (arXiv 2605.11689)."
            ),
            metadata={"learnable_router": True, "aux_loss": True, "default_top_k": 2},
        ),
        ToolListing(
            name="stacking",
            description=(
                "Always-on weighted ensemble baseline. Per-expert or per-dim "
                "softmax weights. Included as the routing-vs-no-routing baseline."
            ),
            metadata={"learnable_router": False, "aux_loss": False},
        ),
    ]


def list_cohorts_tool() -> list[ToolListing]:
    """Return discovery metadata for supported cohorts."""
    descriptions = {
        "breakhis": "BreaKHis — binary benign/malignant with subtype + magnification.",
        "bach": "BACH (ICIAR 2018) — Normal / Benign / In-Situ / Invasive 4-class.",
        "bracs": "BRACS — Normal / PB / UDH / FEA / ADH / DCIS / IC; the early-detection backbone.",
    }
    return [
        ToolListing(name=k, description=descriptions.get(k, ""), metadata={})
        for k in COHORT_BUILDERS
    ]


# --------------------------------------------------------------------------- #
# Cohort + manifest tools
# --------------------------------------------------------------------------- #

def build_unified_manifest_tool(
    cohort_roots: Mapping[str, str | Path],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a unified multi-cohort manifest with the shared schema."""
    manifest = build_cross_cohort_manifest({k: Path(v) for k, v in cohort_roots.items()})
    if output_path is not None:
        save_dataframe(manifest, Path(output_path))
    return manifest


def cohort_split_tool(req: CohortSplitRequest) -> CohortSplitResult:
    """Apply leave-one-cohort-out: hold one cohort fully as the test set."""
    manifest = (
        pd.read_parquet(req.manifest_path)
        if req.manifest_path.suffix == ".parquet"
        else pd.read_csv(req.manifest_path)
    )
    train_df, test_df = None, None
    for held, train, test in leave_one_cohort_out_splits(manifest):
        if held == req.held_out_cohort:
            train_df, test_df = train, test
            break
    if train_df is None:
        raise ValueError(
            f"held_out_cohort {req.held_out_cohort!r} not present in manifest; "
            f"found {sorted(manifest['cohort'].unique())}"
        )

    train_path: Path | None = None
    test_path: Path | None = None
    if req.output_dir is not None:
        out = ensure_dir(req.output_dir)
        train_path = out / f"train_holdout_{req.held_out_cohort}.parquet"
        test_path = out / f"test_holdout_{req.held_out_cohort}.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

    return CohortSplitResult(
        held_out_cohort=req.held_out_cohort,
        train_count=int(len(train_df)),
        test_count=int(len(test_df)),
        train_path=train_path,
        test_path=test_path,
    )


# --------------------------------------------------------------------------- #
# FM cache tools
# --------------------------------------------------------------------------- #

def _resolve_extract_manifest(req: FMExtractRequest) -> pd.DataFrame:
    if req.manifest_path is not None:
        path = req.manifest_path
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if req.root is None:
        raise ValueError("FMExtractRequest needs either root or manifest_path")
    builder = COHORT_BUILDERS.get(req.cohort.lower())
    if builder is None:
        raise ValueError(
            f"Unknown cohort {req.cohort!r}. Available: {sorted(COHORT_BUILDERS)}"
        )
    return builder(req.root)


def extract_fm_embeddings_tool(
    req: FMExtractRequest,
    backend: FMBackend | None = None,
) -> FMExtractResult:
    """Extract FM embeddings for the pending rows of a cohort manifest."""
    manifest = _resolve_extract_manifest(req)
    cache = FMEmbeddingCache(req.cache_root, cohort=req.cohort, fm_name=req.fm_name)
    pending = cache.missing_rows(manifest)
    num_existing = int(len(manifest) - len(pending))
    if req.limit and len(pending) > req.limit:
        pending = pending.head(req.limit).reset_index(drop=True)
    if backend is None:
        backend_kwargs: dict = dict(device=req.device, batch_size=req.batch_size, fp16=req.fp16)
        if req.local_checkpoint is not None:
            backend_kwargs["local_checkpoint"] = str(req.local_checkpoint)
        backend = build_backend(req.backend_name or req.fm_name, **backend_kwargs)

    if not pending.empty:
        paths = pending["image_path"].astype(str).tolist()
        if req.multipatch:
            vectors = backend.encode_paths_multipatch(
                paths, patch_size=req.patch_size, max_patches=req.max_patches,
                tissue_only=req.tissue_only,
            )
        else:
            vectors = backend.encode_paths(paths)
        if vectors.shape != (len(pending), backend.embedding_dim):
            raise RuntimeError(
                f"Unexpected backend output shape {vectors.shape}, expected "
                f"({len(pending)}, {backend.embedding_dim})"
            )
        cache.write(
            pending,
            vectors.astype(np.float32, copy=False),
            model_id=backend.model_id,
            embedding_dim=backend.embedding_dim,
            input_size=backend.input_size,
            pooling="cls+mean_patch" if backend.name == "virchow2" else "cls_token",
            fp16_inference=backend.fp16,
            notes=backend.license_note,
        )

    return FMExtractResult(
        cache=FMCacheLocator(
            cache_root=req.cache_root, cohort=req.cohort, fm_name=req.fm_name
        ),
        num_existing=num_existing,
        num_added=int(len(pending)),
        embedding_dim=int(backend.embedding_dim),
        model_id=backend.model_id,
    )


def slice_fm_cache_tool(
    req: CacheSliceRequest,
) -> tuple[CacheSliceResult, pd.DataFrame, np.ndarray]:
    """Fetch cached embeddings + manifest rows for a list of image_ids.

    Returns (summary, manifest_rows, embeddings). Missing image_ids are
    reported in :attr:`CacheSliceResult.image_ids_missing` and silently
    skipped in the returned arrays.
    """
    cache = FMEmbeddingCache(req.cache.cache_root, req.cache.cohort, req.cache.fm_name)
    rows, vectors = cache.slice_by_image_ids(req.image_ids)
    found = list(rows["image_id"]) if not rows.empty else []
    missing = [i for i in req.image_ids if i not in set(found)]
    summary = CacheSliceResult(
        image_ids_found=found,
        image_ids_missing=missing,
        embedding_shape=tuple(vectors.shape) if vectors.size else (0, 0),
        cache=req.cache,
    )
    return summary, rows, np.asarray(vectors)


# --------------------------------------------------------------------------- #
# MoE forward tool
# --------------------------------------------------------------------------- #

def _align_caches_to_image_ids(
    fm_caches: list[FMCacheLocator],
    image_ids: list[str],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, int], pd.DataFrame]:
    """Slice each FM cache to the intersection of image_ids.

    Returns the surviving ordered image_ids, a mapping from fm_name to its
    embedding matrix (rows aligned to surviving ids), a mapping from fm_name
    to its embedding dim, and the metadata frame from the first cache.
    """
    per_fm_vectors: dict[str, np.ndarray] = {}
    per_fm_index: dict[str, dict[str, int]] = {}
    per_fm_dim: dict[str, int] = {}
    first_manifest: pd.DataFrame | None = None
    for loc in fm_caches:
        cache = FMEmbeddingCache(loc.cache_root, loc.cohort, loc.fm_name)
        manifest = cache.load_manifest()
        embeddings = cache.load_embeddings(mmap=False)
        per_fm_vectors[loc.fm_name] = embeddings
        per_fm_index[loc.fm_name] = {
            str(row["image_id"]): int(row["row_idx"]) for _, row in manifest.iterrows()
        }
        per_fm_dim[loc.fm_name] = int(embeddings.shape[1]) if embeddings.size else 0
        if first_manifest is None:
            first_manifest = manifest
    surviving = [
        i for i in image_ids if all(i in per_fm_index[loc.fm_name] for loc in fm_caches)
    ]
    aligned: dict[str, np.ndarray] = {}
    for loc in fm_caches:
        idx = [per_fm_index[loc.fm_name][i] for i in surviving]
        aligned[loc.fm_name] = (
            per_fm_vectors[loc.fm_name][idx] if idx else np.empty((0, per_fm_dim[loc.fm_name]), dtype=np.float32)
        )
    meta_frame = (
        first_manifest.set_index("image_id").loc[surviving].reset_index()
        if first_manifest is not None and surviving
        else pd.DataFrame()
    )
    return surviving, aligned, per_fm_dim, meta_frame


def moe_forward_tool(req: MoEForwardRequest) -> MoEForwardResult:
    """Run an MoE head forward pass on cached FM embeddings.

    The head is built from ``req.moe_config``. If ``checkpoint_path`` is set,
    a state dict is loaded; otherwise the head uses random init (useful for
    shape / discovery calls before training has finished).
    """
    import torch

    from breastpath_copilot.models.moe import MoEClassifier, MoEConfig, build_moe

    if not req.fm_caches:
        raise ValueError("fm_caches must contain at least one entry")
    if not req.image_ids:
        raise ValueError("image_ids must be non-empty")

    surviving, aligned, dims, meta_frame = _align_caches_to_image_ids(
        req.fm_caches, req.image_ids
    )
    if not surviving:
        return MoEForwardResult(
            image_ids=[],
            routed_embedding_shape=(0, req.moe_config.shared_dim),
            gate_weights=[],
            chosen_experts=[],
            expert_names=[loc.fm_name for loc in req.fm_caches],
            logits=None,
        )

    expert_dims = {loc.fm_name: dims[loc.fm_name] for loc in req.fm_caches}
    # Clamp top_k to the number of available experts. Asking for top-2 routing
    # when only one FM cache is supplied should degrade gracefully to top-1
    # rather than raising -- this happens routinely with single-FM contexts.
    effective_top_k = max(1, min(req.moe_config.top_k, len(expert_dims)))
    moe_cfg = MoEConfig(
        routing=req.moe_config.routing,
        shared_dim=req.moe_config.shared_dim,
        top_k=effective_top_k,
        gate_hidden_dim=req.moe_config.gate_hidden_dim,
        load_balance_weight=req.moe_config.load_balance_weight,
        per_dim_weights=req.moe_config.per_dim_weights,
        layer_norm=req.moe_config.layer_norm,
    )

    expert_tensors = {
        name: torch.as_tensor(vec, dtype=torch.float32) for name, vec in aligned.items()
    }
    mags = req.magnifications
    if mags is None and "magnification" in meta_frame.columns:
        mags = [str(m) for m in meta_frame["magnification"].tolist()]

    if req.classifier_num_classes is not None:
        head = MoEClassifier(
            expert_dims, num_classes=req.classifier_num_classes, config=moe_cfg
        )
        if req.checkpoint_path is not None:
            head.load_state_dict(torch.load(req.checkpoint_path, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            logits_t, out = head(expert_tensors, magnifications=mags)
        logits_list: list[list[float]] | None = logits_t.tolist()
    else:
        head = build_moe(expert_dims, moe_cfg)
        if req.checkpoint_path is not None:
            head.load_state_dict(torch.load(req.checkpoint_path, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            out = head(expert_tensors, magnifications=mags)
        logits_list = None

    return MoEForwardResult(
        image_ids=surviving,
        routed_embedding_shape=tuple(out.routed_embedding.shape),
        gate_weights=out.gate_weights.tolist(),
        chosen_experts=out.chosen_experts.tolist(),
        expert_names=list(expert_dims.keys()),
        logits=logits_list,
    )
