"""Extract foundation-model embeddings for a cohort and persist them to disk.

Usage::

    python scripts/extract_fm_embeddings.py \
        --cohort bracs \
        --root /data/BRACS \
        --fm phikon_v2 \
        --cache-root artifacts/fm_cache \
        --batch-size 32

The cache layout is described in :mod:`breastpath_copilot.core.fm_cache`.
Re-running with the same arguments only computes embeddings for image_ids that
are not yet present in the cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich import print

from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.core.fm_embeddings import BACKEND_REGISTRY, build_backend
from breastpath_copilot.data_cohorts import COHORT_BUILDERS

app = typer.Typer(add_completion=False)


def _build_manifest(cohort: str, root: Path, manifest: Path | None) -> pd.DataFrame:
    if manifest is not None:
        return pd.read_parquet(manifest) if manifest.suffix == ".parquet" else pd.read_csv(manifest)
    key = cohort.lower()
    if key not in COHORT_BUILDERS:
        raise typer.BadParameter(
            f"Unknown cohort {cohort!r}. Available: {sorted(COHORT_BUILDERS)}"
        )
    return COHORT_BUILDERS[key](root)


@app.command()
def main(
    cohort: str = typer.Option(..., help="One of: breakhis, bach, bracs"),
    fm: str = typer.Option(..., help=f"FM backend short name. One of: {sorted(BACKEND_REGISTRY)}"),
    root: Path = typer.Option(None, help="Cohort dataset root (skip if --manifest is provided)"),
    manifest: Path = typer.Option(None, help="Pre-built manifest (CSV or parquet) overriding --root"),
    cache_root: Path = typer.Option(Path("artifacts/fm_cache"), help="FM embedding cache root"),
    batch_size: int = typer.Option(32, help="Inference batch size"),
    device: str = typer.Option("auto", help="Device: auto | cuda | cuda:N | cpu"),
    fp16: bool = typer.Option(True, help="Use float16 inference on CUDA"),
    limit: int = typer.Option(0, help="Optional cap on number of new images (0 = no cap)"),
) -> None:
    if root is None and manifest is None:
        raise typer.BadParameter("Pass either --root or --manifest")

    df = _build_manifest(cohort, root, manifest)
    print(f"[bold cyan]Manifest[/bold cyan] for {cohort}: {len(df)} rows")

    cache = FMEmbeddingCache(cache_root, cohort=cohort, fm_name=fm)
    pending = cache.missing_rows(df)
    print(
        f"[bold]Cache[/bold] at {cache.dir}: "
        f"{len(df) - len(pending)} already cached, {len(pending)} pending"
    )
    if limit and len(pending) > limit:
        pending = pending.head(limit).reset_index(drop=True)
        print(f"[yellow]Capping to first {limit} new images[/yellow]")
    if pending.empty:
        print("[green]Nothing to do.[/green]")
        return

    backend = build_backend(fm, device=device, batch_size=batch_size, fp16=fp16)
    print(
        f"[bold magenta]Backend[/bold magenta] {backend.name} ({backend.model_id}), "
        f"dim={backend.embedding_dim}, input={backend.input_size}, device={backend.device}"
    )

    paths = pending["image_path"].astype(str).tolist()
    vectors = backend.encode_paths(paths)
    if vectors.shape != (len(pending), backend.embedding_dim):
        raise RuntimeError(
            f"Unexpected embedding shape {vectors.shape}, "
            f"expected ({len(pending)}, {backend.embedding_dim})"
        )

    metadata = cache.write(
        pending,
        vectors.astype(np.float32, copy=False),
        model_id=backend.model_id,
        embedding_dim=backend.embedding_dim,
        input_size=backend.input_size,
        pooling="cls+mean_patch" if backend.name == "virchow2" else "cls_token",
        fp16_inference=backend.fp16,
        notes=backend.license_note,
    )
    print(
        f"[bold green]Wrote[/bold green] {len(pending)} new rows -> total {metadata.num_records}"
    )


if __name__ == "__main__":
    app()
