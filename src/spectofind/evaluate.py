"""
evaluate.py — Per-class accuracy report and confusion matrix for Spectofind.

Usage:
    uv run python -m spectofind.evaluate
    uv run python -m spectofind.evaluate --checkpoint checkpoints/last_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich import box

from spectofind import config as cfg
from spectofind.dataset import SpectrogramDataset, get_class_names
from spectofind.model import load_checkpoint

console = Console()


@torch.no_grad()
def evaluate(checkpoint_path: str | None = None) -> dict:
    """
    Run full evaluation on the validation fold.

    Returns a dict with:
        - overall_accuracy (float)
        - per_class_accuracy (list[float], len = NUM_CLASSES)
        - confusion_matrix (np.ndarray, shape NUM_CLASSESxNUM_CLASSES)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.rule("[bold cyan]Spectofind — Evaluation[/bold cyan]")
    model = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    val_ds = SpectrogramDataset(folds=[cfg.VAL_FOLD], augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    all_preds: list[int] = []
    all_labels: list[int] = []

    console.print(f"[dim]Evaluating on fold {cfg.VAL_FOLD} ({len(val_ds)} samples) …[/dim]")

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=cfg.USE_AMP and device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds,  dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)

    class_names = get_class_names()
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int32)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    per_class_acc = []
    for i in range(cfg.NUM_CLASSES):
        total_i = cm[i].sum()
        per_class_acc.append(cm[i, i] / total_i if total_i > 0 else 0.0)

    overall = (all_preds == all_labels).mean()

    # ── Rich table ────────────────────────────────────────────────────────────
    table = Table(
        title=f"Per-Class Accuracy  (Overall: [bold]{overall*100:.2f}%[/bold])",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Idx",      width=5,  justify="right", style="dim")
    table.add_column("Class",    width=22, style="cyan")
    table.add_column("Accuracy", width=10, justify="right")
    table.add_column("Correct",  width=10, justify="right")
    table.add_column("Total",    width=8,  justify="right", style="dim")

    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.5 else "red")
        total_i = int(cm[i].sum())
        correct_i = int(cm[i, i])
        table.add_row(
            str(i),
            name,
            f"[{color}]{acc*100:.1f}%[/{color}]",
            str(correct_i),
            str(total_i),
        )

    console.print(table)
    console.print(
        f"\n[bold green]Overall accuracy: {overall*100:.2f}%[/bold green]  "
        f"[dim](fold {cfg.VAL_FOLD}, {len(val_ds)} samples)[/dim]"
    )

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    _save_confusion_matrix(cm, class_names, overall)

    return {
        "overall_accuracy": float(overall),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
    }


def _save_confusion_matrix(
    cm: np.ndarray, class_names: list[str], overall_acc: float
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Normalise per row (true class) so colours show accuracy regardless of class size
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm_norm,
        annot=False,       # 50×50 = too dense for text
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="grey",
    )
    ax.set_title(
        f"Spectofind — Confusion Matrix (Overall acc: {overall_acc*100:.2f}%)",
        fontsize=14,
        pad=16,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    fig.tight_layout()
    out = cfg.RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[dim]  Confusion matrix saved → {out}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Spectofind model")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file (default: best_model.pth)"
    )
    args = parser.parse_args()
    evaluate(args.checkpoint)


if __name__ == "__main__":
    main()
