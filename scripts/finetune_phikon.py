"""Fine-tune Phikon-v2 on the unified multi-cohort manifest.

This is the *trainable expert* of the hybrid MoE compute strategy: UNI and
Virchow2 are cached frozen, while Phikon-v2 is fine-tuned here. The resulting
checkpoint can then be fed to ``PhikonV2Backend(local_checkpoint=...)`` to
re-extract embeddings for the fine-tuned expert.

Example (single cohort with native split column, e.g. BRACS)::

    python scripts/finetune_phikon.py \
        --manifest artifacts/manifests/bracs_unified.parquet \
        --output-dir artifacts/phikon_ft/bracs \
        --mode full --epochs 15 --batch-size 32 --lr 1e-5 --head-lr 1e-4

Example (leave-one-cohort-out: train on BreaKHis+BACH, validate on a held
slice; build the manifest + split column beforehand with the cohort tools)::

    python scripts/finetune_phikon.py \
        --manifest artifacts/manifests/loco_holdout_bracs_train.parquet \
        --val-manifest artifacts/manifests/loco_holdout_bracs_val.parquet \
        --output-dir artifacts/phikon_ft/holdout_bracs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich import print

app = typer.Typer(add_completion=False)


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@app.command()
def main(
    manifest: Path = typer.Option(..., help="Train manifest (or full manifest with a split column)."),
    val_manifest: Path = typer.Option(None, help="Optional separate validation manifest."),
    output_dir: Path = typer.Option(..., help="Where to write checkpoints + summary."),
    split_col: str = typer.Option("split", help="Split column used when --val-manifest is omitted."),
    val_value: str = typer.Option("val", help="Value in split_col marking validation rows."),
    mode: str = typer.Option("full", help="full | linear_probe"),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(1e-5, help="Encoder LR (full mode)."),
    head_lr: float = typer.Option(1e-4, help="Classifier-head LR."),
    weight_decay: float = typer.Option(0.05),
    input_size: int = typer.Option(224),
    num_workers: int = typer.Option(4),
    no_amp: bool = typer.Option(False, "--no-amp", help="Disable mixed precision."),
    no_class_weights: bool = typer.Option(False, "--no-class-weights", help="Disable inverse-freq class weights."),
    hf_token: str = typer.Option(None, help="HF token if Phikon-v2 requires auth."),
    device: str = typer.Option("auto"),
) -> None:
    import torch
    from torch.utils.data import DataLoader

    from breastpath_copilot.training import (
        BreastPathologyDataset,
        FinetuneConfig,
        Phikon2Classifier,
        Phikon2Finetuner,
        UNIFIED_LABEL_TO_INDEX,
        build_train_val_datasets,
        compute_class_weights,
    )
    from breastpath_copilot.training.datasets import collate_batch

    full_df = _read_manifest(manifest)
    if val_manifest is not None:
        train_ds = BreastPathologyDataset(full_df, input_size=input_size, train=True)
        val_df = _read_manifest(val_manifest)
        val_ds = BreastPathologyDataset(val_df, input_size=input_size, train=False)
    else:
        train_ds, val_ds = build_train_val_datasets(
            full_df, val_split_col=split_col, val_value=val_value, input_size=input_size
        )
    print(f"[bold cyan]Train[/bold cyan] {len(train_ds)} | [bold cyan]Val[/bold cyan] {len(val_ds)}")

    class_weights = None
    if not no_class_weights:
        class_weights = compute_class_weights(train_ds.manifest["label"].tolist())
        print(f"[dim]class weights:[/dim] {class_weights.tolist()}")

    cfg = FinetuneConfig(
        mode=mode,
        num_classes=len(UNIFIED_LABEL_TO_INDEX),
        input_size=input_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        use_amp=not no_amp,
        num_workers=num_workers,
        device=device,
    )
    model = Phikon2Classifier(num_classes=cfg.num_classes, mode=mode, hf_token=hf_token)
    trainer = Phikon2Finetuner(
        model=model, config=cfg, class_weights=class_weights, output_dir=output_dir
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_batch, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_batch, pin_memory=True,
    )

    summary = trainer.fit(train_loader, val_loader)
    print(
        f"[bold green]Done.[/bold green] best_epoch={summary['best_epoch']} "
        f"{summary['monitor_metric']}={summary['best_metric']}"
    )
    if summary["best_checkpoint"]:
        print(f"  best checkpoint -> {summary['best_checkpoint']}")
        print(
            "  re-extract fine-tuned embeddings with: "
            "PhikonV2Backend(local_checkpoint='<path>')"
        )


if __name__ == "__main__":
    app()
