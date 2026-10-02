from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SlideInput(BaseModel):
    slide_path: Path
    slide_id: str
    patient_id: str | None = None
    stain_type: Literal["H&E", "IHC", "OTHER"] = "H&E"
    magnification: str = "40X"


class QCResult(BaseModel):
    slide_id: str
    passed: bool
    blur_score: float
    mean_brightness: float
    notes: list[str] = Field(default_factory=list)


class PatchRecord(BaseModel):
    slide_id: str
    patch_id: str
    x: int
    y: int
    width: int
    height: int
    tissue_ratio: float
    patch_path: str | None = None


class EmbeddingBatch(BaseModel):
    slide_id: str
    patch_ids: list[str]
    embedding_dim: int


class PatchPrediction(BaseModel):
    slide_id: str
    patch_id: str
    predicted_label: str
    probabilities: dict[str, float]


class SlidePrediction(BaseModel):
    slide_id: str
    predicted_label: str
    probabilities: dict[str, float]
    suspicious_patch_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class FMEmbeddingMetadata(BaseModel):
    """Sidecar describing a cached foundation-model embedding bundle."""

    cohort: Literal["breakhis", "bach", "bracs"] | str
    fm_name: str
    model_id: str
    model_revision: str | None = None
    embedding_dim: int
    num_records: int
    input_size: int = 224
    preprocessing: str = "imagenet"
    pooling: str = "cls_token"
    fp16_inference: bool = True
    created_at: str = ""
    source_manifest_hash: str | None = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# MCP tool I/O schemas (Q1 extension)
# --------------------------------------------------------------------------- #

class FMCacheLocator(BaseModel):
    """Identifies a foundation-model embedding cache on disk."""

    cache_root: Path
    cohort: str
    fm_name: str


class FMExtractRequest(BaseModel):
    """Request to extract foundation-model embeddings for one cohort x FM.

    ``fm_name`` is the cache identifier (the on-disk folder). ``backend_name``,
    if set, selects which registered backend actually computes the embeddings
    — this lets a fine-tuned variant be cached under a distinct name (e.g.
    fm_name='phikon_v2_ft', backend_name='phikon_v2', local_checkpoint=...).
    """

    cohort: Literal["breakhis", "bach", "bracs"] | str
    fm_name: str
    backend_name: str | None = None
    local_checkpoint: Path | None = None
    root: Path | None = None
    manifest_path: Path | None = None
    cache_root: Path = Path("artifacts/fm_cache")
    batch_size: int = 32
    device: str = "auto"
    fp16: bool = True
    limit: int = 0
    # Multi-patch tiling: when multipatch=True, each RoI is tiled into
    # native-resolution patches, each embedded, and mean-pooled to one vector
    # (preserves cellular detail vs resizing the whole RoI to 224).
    multipatch: bool = False
    patch_size: int = 224
    max_patches: int = 16
    tissue_only: bool = True


class FMExtractResult(BaseModel):
    cache: FMCacheLocator
    num_existing: int
    num_added: int
    embedding_dim: int
    model_id: str


class CacheSliceRequest(BaseModel):
    cache: FMCacheLocator
    image_ids: list[str]


class CacheSliceResult(BaseModel):
    image_ids_found: list[str]
    image_ids_missing: list[str]
    embedding_shape: tuple[int, int]
    cache: FMCacheLocator


class CohortSplitRequest(BaseModel):
    """Leave-one-cohort-out split of a unified multi-cohort manifest."""

    manifest_path: Path
    held_out_cohort: str
    output_dir: Path | None = None


class CohortSplitResult(BaseModel):
    held_out_cohort: str
    train_count: int
    test_count: int
    train_path: Path | None = None
    test_path: Path | None = None


class MoEHeadConfig(BaseModel):
    """JSON-serialisable mirror of models.moe.MoEConfig."""

    routing: Literal["hard_mag", "soft_topk", "stacking"]
    shared_dim: int = 512
    top_k: int = 2
    gate_hidden_dim: int | None = None
    load_balance_weight: float = 0.01
    per_dim_weights: bool = False
    layer_norm: bool = True


class MoEForwardRequest(BaseModel):
    """Run the MoE head on cached embeddings for a batch of image ids."""

    fm_caches: list[FMCacheLocator]
    image_ids: list[str]
    moe_config: MoEHeadConfig
    checkpoint_path: Path | None = None
    magnifications: list[str] | None = None
    classifier_num_classes: int | None = None


class MoEForwardResult(BaseModel):
    image_ids: list[str]
    routed_embedding_shape: tuple[int, int]
    gate_weights: list[list[float]]
    chosen_experts: list[list[int]]
    expert_names: list[str]
    logits: list[list[float]] | None = None


class ToolListing(BaseModel):
    """Discovery: human-readable names + metadata for available choices."""

    name: str
    description: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
