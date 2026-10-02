"""Real Model Context Protocol server for the Pathology MCP tool layer.

This wraps the typed tools in :mod:`breastpath_copilot.api.tools` with the
official ``mcp`` Python SDK (FastMCP), so the server speaks the actual MCP
protocol over stdio (and can be mounted over HTTP/SSE), exposing pathology
operations as protocol-native callable tools.

Run (stdio)::

    python -m breastpath_copilot.mcp_server
    # or, after install:
    breastpath-mcp

The heavy ML imports (torch, the MoE forward path) are deferred into the tool
bodies so that ``build_server()`` and tool discovery stay light-weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - guard for missing dep
    raise RuntimeError(
        "The 'mcp' package is required to run the protocol server. "
        "Install via `pip install mcp`."
    ) from exc

from breastpath_copilot.api.tools import (
    build_unified_manifest_tool,
    cohort_split_tool,
    extract_fm_embeddings_tool,
    list_cohorts_tool,
    list_fm_backends_tool,
    list_moe_routings_tool,
    moe_forward_tool,
    slice_fm_cache_tool,
)
from breastpath_copilot.schemas import (
    CacheSliceRequest,
    CohortSplitRequest,
    FMCacheLocator,
    FMExtractRequest,
    MoEForwardRequest,
    MoEHeadConfig,
)


def build_server(name: str = "pathology-mcp") -> "FastMCP":
    """Construct and return the FastMCP server with all tools registered."""
    mcp = FastMCP(name)

    # ----------------------- discovery tools ----------------------------- #

    @mcp.tool(description="List the registered pathology foundation-model experts.")
    def list_fm_backends() -> list[dict[str, Any]]:
        return [item.model_dump() for item in list_fm_backends_tool()]

    @mcp.tool(description="List the available MoE routing strategies.")
    def list_moe_routings() -> list[dict[str, Any]]:
        return [item.model_dump() for item in list_moe_routings_tool()]

    @mcp.tool(description="List the supported pathology cohorts (BreaKHis, BACH, BRACS).")
    def list_cohorts() -> list[dict[str, Any]]:
        return [item.model_dump() for item in list_cohorts_tool()]

    # ----------------------- cohort / manifest --------------------------- #

    @mcp.tool(
        description=(
            "Build a unified multi-cohort manifest from dataset roots. "
            "cohort_roots maps cohort name -> dataset root path. "
            "Optionally writes the manifest to output_path."
        )
    )
    def build_unified_manifest(
        cohort_roots: dict[str, str], output_path: str | None = None
    ) -> dict[str, Any]:
        df = build_unified_manifest_tool(cohort_roots, output_path=output_path)
        return {
            "num_rows": int(len(df)),
            "cohorts": sorted(df["cohort"].unique().tolist()),
            "label_counts": df["label"].value_counts().to_dict(),
            "output_path": output_path,
        }

    @mcp.tool(
        description=(
            "Leave-one-cohort-out split: hold one cohort fully as the test set. "
            "Reads a unified manifest (parquet/csv) and writes train/test parquets."
        )
    )
    def cohort_split(
        manifest_path: str, held_out_cohort: str, output_dir: str | None = None
    ) -> dict[str, Any]:
        result = cohort_split_tool(
            CohortSplitRequest(
                manifest_path=Path(manifest_path),
                held_out_cohort=held_out_cohort,
                output_dir=Path(output_dir) if output_dir else None,
            )
        )
        return result.model_dump()

    # ----------------------- FM cache ------------------------------------ #

    @mcp.tool(
        description=(
            "Extract foundation-model embeddings for a cohort and cache them. "
            "Idempotent: only computes embeddings for image_ids not yet cached. "
            "Requires the FM weights to be available (and HF_TOKEN for gated models)."
        )
    )
    def extract_fm_embeddings(
        cohort: str,
        fm_name: str,
        root: str | None = None,
        manifest_path: str | None = None,
        cache_root: str = "artifacts/fm_cache",
        batch_size: int = 32,
        device: str = "auto",
        fp16: bool = True,
        limit: int = 0,
    ) -> dict[str, Any]:
        req = FMExtractRequest(
            cohort=cohort,
            fm_name=fm_name,
            root=Path(root) if root else None,
            manifest_path=Path(manifest_path) if manifest_path else None,
            cache_root=Path(cache_root),
            batch_size=batch_size,
            device=device,
            fp16=fp16,
            limit=limit,
        )
        return extract_fm_embeddings_tool(req).model_dump()

    @mcp.tool(
        description=(
            "Fetch cached FM embeddings for a list of image_ids. Returns the "
            "found/missing ids and the embedding matrix shape (the embedding "
            "values themselves are large and are not returned over the protocol)."
        )
    )
    def slice_fm_cache(
        cache_root: str, cohort: str, fm_name: str, image_ids: list[str]
    ) -> dict[str, Any]:
        summary, _rows, _vecs = slice_fm_cache_tool(
            CacheSliceRequest(
                cache=FMCacheLocator(cache_root=Path(cache_root), cohort=cohort, fm_name=fm_name),
                image_ids=image_ids,
            )
        )
        return summary.model_dump()

    # ----------------------- MoE forward --------------------------------- #

    @mcp.tool(
        description=(
            "Run the MoE head on cached FM embeddings for a batch of image_ids. "
            "fm_caches is a list of {cache_root, cohort, fm_name}. Returns routed "
            "embedding shape, gate weights over experts, chosen experts, and "
            "(if classifier_num_classes is set) class logits."
        )
    )
    def moe_forward(
        fm_caches: list[dict[str, str]],
        image_ids: list[str],
        routing: str = "soft_topk",
        shared_dim: int = 512,
        top_k: int = 2,
        checkpoint_path: str | None = None,
        classifier_num_classes: int | None = None,
        magnifications: list[str] | None = None,
    ) -> dict[str, Any]:
        locators = [
            FMCacheLocator(
                cache_root=Path(c["cache_root"]), cohort=c["cohort"], fm_name=c["fm_name"]
            )
            for c in fm_caches
        ]
        req = MoEForwardRequest(
            fm_caches=locators,
            image_ids=image_ids,
            moe_config=MoEHeadConfig(routing=routing, shared_dim=shared_dim, top_k=top_k),
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            classifier_num_classes=classifier_num_classes,
            magnifications=magnifications,
        )
        return moe_forward_tool(req).model_dump()

    return mcp


def main() -> None:
    """Entry point: run the server over stdio."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
