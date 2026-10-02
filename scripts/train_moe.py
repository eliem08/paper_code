"""Run the leave-one-cohort-out MoE ablation and emit the Q1 results table.

Prerequisite: FM embeddings cached for every (cohort, FM) pair via
``scripts/extract_fm_embeddings.py`` (or the ``extract-fm`` server command).

Example::

    python scripts/train_moe.py \
        --cache-root artifacts/fm_cache \
        --cohort breakhis --cohort bach --cohort bracs \
        --fm uni --fm virchow2 --fm phikon_v2 \
        --output-dir artifacts/moe_loco \
        --epochs 50

Produces ``artifacts/moe_loco/ablation_table.csv`` with one row per
(held_out_cohort x routing) and per-routing checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cache_root: Path = typer.Option(Path("artifacts/fm_cache")),
    cohort: list[str] = typer.Option(..., "--cohort", help="Cohort name (repeat for each)."),
    fm: list[str] = typer.Option(..., "--fm", help="FM short name (repeat for each)."),
    output_dir: Path = typer.Option(Path("artifacts/moe_loco")),
    routings: list[str] = typer.Option(
        ["hard_mag", "soft_topk", "stacking"], "--routing", help="Routings to benchmark."
    ),
    epochs: int = typer.Option(50),
    shared_dim: int = typer.Option(512),
    top_k: int = typer.Option(2),
    lr: float = typer.Option(1e-3),
    batch_size: int = typer.Option(256),
    no_class_weights: bool = typer.Option(False, "--no-class-weights"),
    no_calibrate: bool = typer.Option(False, "--no-calibrate"),
    device: str = typer.Option("auto"),
) -> None:
    from breastpath_copilot.training.loco_runner import run_loco_ablation
    from breastpath_copilot.training.moe_trainer import MoETrainConfig

    base_cfg = MoETrainConfig(
        shared_dim=shared_dim,
        top_k=top_k,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_class_weights=not no_class_weights,
        calibrate=not no_calibrate,
        device=device,
    )
    print(
        f"[bold cyan]LOCO sweep[/bold cyan] cohorts={list(cohort)} fms={list(fm)} "
        f"routings={list(routings)}"
    )
    df, _results = run_loco_ablation(
        cache_root,
        list(cohort),
        list(fm),
        routings=routings,
        base_config=base_cfg,
        output_dir=output_dir,
    )
    if df.empty:
        print("[yellow]No results produced (check that caches exist for every cohort x FM).[/yellow]")
        raise typer.Exit(1)

    table = Table(title="LOCO ablation (held-out cohort x routing)")
    for col in ["held_out_cohort", "routing", "test_n", "accuracy", "macro_f1", "auroc_attention", "ece"]:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(
            str(row["held_out_cohort"]),
            str(row["routing"]),
            str(int(row["test_n"])),
            f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "-",
            f"{row['macro_f1']:.3f}" if row["macro_f1"] is not None else "-",
            f"{row['auroc_attention']:.3f}" if row["auroc_attention"] is not None else "-",
            f"{row['ece']:.3f}" if row["ece"] is not None else "-",
        )
    print(table)
    print(f"[bold green]Wrote[/bold green] {output_dir / 'ablation_table.csv'}")


if __name__ == "__main__":
    app()
