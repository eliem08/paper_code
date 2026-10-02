"""Integration tests for the Q1-extension MCP tools in api/tools.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breastpath_copilot.api.tools import (  # noqa: E402
    build_unified_manifest_tool,
    cohort_split_tool,
    extract_fm_embeddings_tool,
    list_cohorts_tool,
    list_fm_backends_tool,
    list_moe_routings_tool,
    moe_forward_tool,
    slice_fm_cache_tool,
)
from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402
from breastpath_copilot.core.fm_embeddings import FMBackend  # noqa: E402
from breastpath_copilot.schemas import (  # noqa: E402
    CacheSliceRequest,
    CohortSplitRequest,
    FMCacheLocator,
    FMExtractRequest,
    MoEForwardRequest,
    MoEHeadConfig,
)


class _DummyFM(FMBackend):
    name = "dummy_fm"
    model_id = "dummy/test"
    embedding_dim = 16
    input_size = 32
    requires_hf_token = False

    def _load_model(self):  # pragma: no cover - bypassed
        raise NotImplementedError

    def _forward(self, batch):  # pragma: no cover - bypassed
        raise NotImplementedError

    def encode_images(self, images):
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.random.default_rng(42).standard_normal((len(images), self.embedding_dim)).astype(np.float32)

    def encode_paths(self, paths):
        paths = list(paths)
        return np.random.default_rng(42).standard_normal((len(paths), self.embedding_dim)).astype(np.float32)


def _make_breakhis_tree(tmp_path: Path) -> Path:
    """Create a synthetic BreaKHis-style folder with two patients."""
    root = tmp_path / "breakhis"
    paths = [
        root
        / "histology_slides/breast/benign/SOB/adenosis"
        / "SOB_B_A-14-22549A/40X/SOB_B_A-14-22549A-40-001.png",
        root
        / "histology_slides/breast/benign/SOB/adenosis"
        / "SOB_B_A-14-22549A/40X/SOB_B_A-14-22549A-40-002.png",
        root
        / "histology_slides/breast/malignant/SOB/ductal_carcinoma"
        / "SOB_M_DC-14-2523/40X/SOB_M_DC-14-2523-40-001.png",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image as PILImage

        PILImage.new("RGB", (32, 32), color=(120, 60, 200)).save(p)
    return root


def _make_bach_tree(tmp_path: Path) -> Path:
    root = tmp_path / "bach"
    layout = {
        "Normal": ["n001.png", "n002.png"],
        "Benign": ["b001.png"],
        "InSitu": ["is001.png"],
        "Invasive": ["iv001.png"],
    }
    for cls, files in layout.items():
        (root / "Photos" / cls).mkdir(parents=True, exist_ok=True)
        for fname in files:
            from PIL import Image as PILImage

            PILImage.new("RGB", (32, 32), color=(50, 200, 50)).save(root / "Photos" / cls / fname)
    return root


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_list_fm_backends_includes_three_pathology_experts():
    listings = list_fm_backends_tool()
    names = {item.name for item in listings}
    assert {"uni", "virchow2", "phikon_v2"} <= names


def test_list_moe_routings_includes_three_strategies():
    listings = list_moe_routings_tool()
    names = [item.name for item in listings]
    assert names == ["hard_mag", "soft_topk", "stacking"]


def test_list_cohorts_covers_three_cohorts():
    listings = list_cohorts_tool()
    assert {item.name for item in listings} == {"breakhis", "bach", "bracs"}


# --------------------------------------------------------------------------- #
# Manifest + cohort split
# --------------------------------------------------------------------------- #

def test_build_unified_manifest_combines_breakhis_and_bach(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    bach_root = _make_bach_tree(tmp_path)
    output_path = tmp_path / "unified.parquet"
    manifest = build_unified_manifest_tool(
        {"breakhis": breakhis_root, "bach": bach_root}, output_path=output_path
    )
    assert output_path.exists()
    cohorts = set(manifest["cohort"])
    assert cohorts == {"breakhis", "bach"}
    # BACH InSitu must be present as the 'dcis' early-detection class.
    bach_rows = manifest[manifest["cohort"] == "bach"]
    assert "dcis" in set(bach_rows["label"])


def test_cohort_split_holds_out_named_cohort(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    bach_root = _make_bach_tree(tmp_path)
    manifest_path = tmp_path / "unified.parquet"
    build_unified_manifest_tool(
        {"breakhis": breakhis_root, "bach": bach_root}, output_path=manifest_path
    )
    req = CohortSplitRequest(
        manifest_path=manifest_path,
        held_out_cohort="bach",
        output_dir=tmp_path / "splits",
    )
    result = cohort_split_tool(req)
    assert result.held_out_cohort == "bach"
    assert result.train_count > 0 and result.test_count > 0
    assert result.train_path is not None and result.train_path.exists()
    assert result.test_path is not None and result.test_path.exists()

    train_df = pd.read_parquet(result.train_path)
    test_df = pd.read_parquet(result.test_path)
    assert (train_df["cohort"] == "bach").sum() == 0
    assert (test_df["cohort"] == "bach").all()


def test_cohort_split_unknown_cohort_raises(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    manifest_path = tmp_path / "unified.parquet"
    build_unified_manifest_tool({"breakhis": breakhis_root}, output_path=manifest_path)
    with pytest.raises(ValueError):
        cohort_split_tool(
            CohortSplitRequest(manifest_path=manifest_path, held_out_cohort="bach")
        )


# --------------------------------------------------------------------------- #
# FM cache extract + slice
# --------------------------------------------------------------------------- #

def test_extract_fm_embeddings_tool_writes_cache(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    backend = _DummyFM()
    req = FMExtractRequest(
        cohort="breakhis",
        fm_name="dummy_fm",
        root=breakhis_root,
        cache_root=tmp_path / "cache",
    )
    result = extract_fm_embeddings_tool(req, backend=backend)
    assert result.num_added == 3
    assert result.num_existing == 0
    assert result.embedding_dim == 16

    # Re-run is idempotent -> nothing added.
    result2 = extract_fm_embeddings_tool(req, backend=backend)
    assert result2.num_existing == 3
    assert result2.num_added == 0


def test_slice_fm_cache_tool_returns_aligned_rows(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    backend = _DummyFM()
    extract_fm_embeddings_tool(
        FMExtractRequest(
            cohort="breakhis",
            fm_name="dummy_fm",
            root=breakhis_root,
            cache_root=tmp_path / "cache",
        ),
        backend=backend,
    )
    cache = FMEmbeddingCache(tmp_path / "cache", "breakhis", "dummy_fm")
    all_ids = list(cache.load_manifest()["image_id"])
    requested = [all_ids[2], all_ids[0], "definitely_missing"]
    summary, rows, vectors = slice_fm_cache_tool(
        CacheSliceRequest(
            cache=FMCacheLocator(
                cache_root=tmp_path / "cache", cohort="breakhis", fm_name="dummy_fm"
            ),
            image_ids=requested,
        )
    )
    assert summary.image_ids_found == [all_ids[2], all_ids[0]]
    assert summary.image_ids_missing == ["definitely_missing"]
    assert vectors.shape == (2, backend.embedding_dim)
    assert list(rows["image_id"]) == [all_ids[2], all_ids[0]]


# --------------------------------------------------------------------------- #
# MoE forward through MCP tool
# --------------------------------------------------------------------------- #

def test_moe_forward_tool_runs_on_three_caches(tmp_path: Path):
    breakhis_root = _make_breakhis_tree(tmp_path)
    cache_root = tmp_path / "cache"
    # Three "FM" caches with different embedding dims to mimic UNI/Virchow2/Phikon-v2.
    for fm_name, dim in [("fm_a", 8), ("fm_b", 16), ("fm_c", 12)]:
        cache = FMEmbeddingCache(cache_root, "breakhis", fm_name)
        df = pd.DataFrame(
            {
                "image_id": ["img_1", "img_2", "img_3"],
                "image_path": [str(breakhis_root)] * 3,
                "magnification": ["40X", "100X", "400X"],
            }
        )
        vectors = np.random.default_rng(dim).standard_normal((3, dim)).astype(np.float32)
        cache.write(df, vectors, model_id=f"dummy/{fm_name}", embedding_dim=dim)

    locators = [
        FMCacheLocator(cache_root=cache_root, cohort="breakhis", fm_name=name)
        for name in ("fm_a", "fm_b", "fm_c")
    ]

    req = MoEForwardRequest(
        fm_caches=locators,
        image_ids=["img_1", "img_2", "img_3", "img_missing"],
        moe_config=MoEHeadConfig(routing="soft_topk", shared_dim=24, top_k=2),
    )
    result = moe_forward_tool(req)

    assert result.image_ids == ["img_1", "img_2", "img_3"]
    assert result.routed_embedding_shape == (3, 24)
    assert result.expert_names == ["fm_a", "fm_b", "fm_c"]
    assert all(abs(sum(row) - 1.0) < 1e-4 for row in result.gate_weights)
    assert all(sum(1 for w in row if w > 0) == 2 for row in result.gate_weights)
    assert result.logits is None


def test_moe_forward_tool_returns_logits_when_classifier_requested(tmp_path: Path):
    cache_root = tmp_path / "cache"
    for fm_name, dim in [("fm_a", 8), ("fm_b", 16)]:
        cache = FMEmbeddingCache(cache_root, "breakhis", fm_name)
        df = pd.DataFrame(
            {"image_id": ["img_x", "img_y"], "image_path": ["/dummy", "/dummy"]}
        )
        vectors = np.random.default_rng(0).standard_normal((2, dim)).astype(np.float32)
        cache.write(df, vectors, model_id=f"dummy/{fm_name}", embedding_dim=dim)

    locators = [
        FMCacheLocator(cache_root=cache_root, cohort="breakhis", fm_name=name)
        for name in ("fm_a", "fm_b")
    ]
    req = MoEForwardRequest(
        fm_caches=locators,
        image_ids=["img_x", "img_y"],
        moe_config=MoEHeadConfig(routing="stacking", shared_dim=12),
        classifier_num_classes=5,
    )
    result = moe_forward_tool(req)
    assert result.logits is not None
    assert len(result.logits) == 2 and len(result.logits[0]) == 5


def test_moe_forward_tool_handles_no_intersecting_ids(tmp_path: Path):
    cache_root = tmp_path / "cache"
    cache = FMEmbeddingCache(cache_root, "breakhis", "fm_a")
    df = pd.DataFrame({"image_id": ["a"], "image_path": ["/x"]})
    cache.write(df, np.zeros((1, 4), dtype=np.float32), model_id="d", embedding_dim=4)

    req = MoEForwardRequest(
        fm_caches=[FMCacheLocator(cache_root=cache_root, cohort="breakhis", fm_name="fm_a")],
        image_ids=["nope"],
        moe_config=MoEHeadConfig(routing="soft_topk", shared_dim=8, top_k=1),
    )
    result = moe_forward_tool(req)
    assert result.image_ids == []
    assert result.routed_embedding_shape == (0, 8)


def test_moe_forward_tool_clamps_top_k_to_single_expert(tmp_path: Path):
    """A single FM cache with top_k=2 must degrade to top-1, not crash."""
    cache_root = tmp_path / "cache"
    cache = FMEmbeddingCache(cache_root, "breakhis", "fm_only")
    df = pd.DataFrame(
        {"image_id": ["i1", "i2"], "image_path": ["/x", "/y"], "magnification": ["40X", "400X"]}
    )
    vecs = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
    cache.write(df, vecs, model_id="dummy/fm_only", embedding_dim=8)

    req = MoEForwardRequest(
        fm_caches=[FMCacheLocator(cache_root=cache_root, cohort="breakhis", fm_name="fm_only")],
        image_ids=["i1", "i2"],
        moe_config=MoEHeadConfig(routing="soft_topk", shared_dim=8, top_k=2),  # 2 > 1 expert
        classifier_num_classes=5,
    )
    result = moe_forward_tool(req)  # must not raise
    assert result.image_ids == ["i1", "i2"]
    assert result.expert_names == ["fm_only"]
    # Only one expert -> each row routes to exactly that one expert.
    assert all(sum(1 for w in row if w > 0) == 1 for row in result.gate_weights)


def test_moe_forward_tool_rejects_empty_inputs():
    with pytest.raises(ValueError):
        moe_forward_tool(
            MoEForwardRequest(
                fm_caches=[],
                image_ids=["a"],
                moe_config=MoEHeadConfig(routing="soft_topk"),
            )
        )
    with pytest.raises(ValueError):
        moe_forward_tool(
            MoEForwardRequest(
                fm_caches=[
                    FMCacheLocator(cache_root=Path("/tmp"), cohort="x", fm_name="y")
                ],
                image_ids=[],
                moe_config=MoEHeadConfig(routing="soft_topk"),
            )
        )
