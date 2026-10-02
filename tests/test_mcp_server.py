"""Tests for the real MCP protocol server wrapper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mcp")
pytest.importorskip("torch")

from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402
from breastpath_copilot.mcp_server import build_server  # noqa: E402


EXPECTED_TOOLS = {
    "list_fm_backends",
    "list_moe_routings",
    "list_cohorts",
    "build_unified_manifest",
    "cohort_split",
    "extract_fm_embeddings",
    "slice_fm_cache",
    "moe_forward",
}


def test_server_builds_with_name():
    server = build_server("test-pathology")
    assert server.name == "test-pathology"


def test_server_registers_all_tools():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS


def test_every_tool_has_input_schema():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    for t in tools:
        assert t.inputSchema is not None
        assert t.inputSchema.get("type") == "object"


def test_discovery_tools_callable_through_server():
    server = build_server()
    # call_tool returns the tool output; discovery tools need no args.
    result = asyncio.run(server.call_tool("list_moe_routings", {}))
    # FastMCP returns a (content, structured) tuple or content list depending
    # on version; normalise to a string blob for the assertion.
    blob = str(result)
    assert "soft_topk" in blob
    assert "hard_mag" in blob
    assert "stacking" in blob


def test_moe_forward_tool_through_server(tmp_path: Path):
    cache_root = tmp_path / "cache"
    for fm, dim in [("fm_a", 8), ("fm_b", 12)]:
        cache = FMEmbeddingCache(cache_root, "breakhis", fm)
        df = pd.DataFrame(
            {
                "image_id": ["i1", "i2", "i3"],
                "image_path": ["/x"] * 3,
                "patient_id": ["p1", "p2", "p3"],
                "label": ["benign", "invasive", "dcis"],
                "magnification": ["40X", "400X", "200X"],
                "cohort": ["breakhis"] * 3,
            }
        )
        vecs = np.random.default_rng(dim).standard_normal((3, dim)).astype(np.float32)
        cache.write(df, vecs, model_id=f"dummy/{fm}", embedding_dim=dim)

    server = build_server()
    args = {
        "fm_caches": [
            {"cache_root": str(cache_root), "cohort": "breakhis", "fm_name": "fm_a"},
            {"cache_root": str(cache_root), "cohort": "breakhis", "fm_name": "fm_b"},
        ],
        "image_ids": ["i1", "i2", "i3"],
        "routing": "soft_topk",
        "shared_dim": 16,
        "top_k": 2,
    }
    result = asyncio.run(server.call_tool("moe_forward", args))
    blob = str(result)
    assert "fm_a" in blob and "fm_b" in blob
    # routed_embedding_shape should reflect the 3 input cases.
    assert "routed_embedding_shape" in blob


def test_cohort_split_tool_through_server(tmp_path: Path):
    # Build a tiny unified manifest with two cohorts.
    manifest = pd.DataFrame(
        {
            "image_path": ["/a", "/b", "/c", "/d"],
            "image_id": ["a", "b", "c", "d"],
            "patient_id": ["p1", "p2", "p3", "p4"],
            "label": ["benign", "invasive", "dcis", "atypia"],
            "subtype": ["x", "y", "z", "w"],
            "magnification": ["unknown"] * 4,
            "cohort": ["breakhis", "breakhis", "bach", "bach"],
        }
    )
    manifest_path = tmp_path / "unified.parquet"
    manifest.to_parquet(manifest_path, index=False)

    server = build_server()
    result = asyncio.run(
        server.call_tool(
            "cohort_split",
            {
                "manifest_path": str(manifest_path),
                "held_out_cohort": "bach",
                "output_dir": str(tmp_path / "splits"),
            },
        )
    )
    blob = str(result)
    assert "bach" in blob
    assert (tmp_path / "splits").exists()
