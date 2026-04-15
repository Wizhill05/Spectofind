"""
train.py — Training loop for Spectofind.

Features:
  - Mixed precision (AMP)
  - OneCycleLR warm-up → CosineAnnealingLR fine-tuning
  - SpecAugment (frequency & time masking) applied on-GPU
  - Backbone unfreeze after epoch 5 for full fine-tuning
  - Best-val-accuracy checkpoint saving
  - Resume from last checkpoint with --resume
  - Rich progress display

Usage:
    uv run python -m spectofind.train
    uv run python -m spectofind.train --epochs 60 --resume
    uv run python -m spectofind.train --epochs 40 --batch-size 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking
from rich.console import Console
from rich.table import Table
from rich import box

from spectofind import config as cfg
from spectofind.dataset import SpectrogramDataset, get_class_names
from spectofind.model import SpectroNet, build_model, count_parameters

console = Console()


# ── SpecAugment (on GPU, applied per-batch) ────────────────────────────────

class SpecAugment(nn.Module):
    """Apply frequency and time masking to a batch of log-mel spectrogram images.
    
    Input shape: (B, C, H, W)  — treat H as frequency, W as time.
    """
    def __init__(
        self,
        freq_mask_param: int = cfg.FREQ_MASK_PARAM,
        time_mask_param: int = cfg.TIME_MASK_PARAM,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList([FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)])
        self.time_masks = nn.ModuleList([TimeMasking(time_mask_param) for _ in range(n_time_masks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — apply masks per item in batch along H and W dims
        # FrequencyMasking expects (B, freq, time), so squeeze/expand as needed
        B, C, H, W = x.shape
        # Flatten channel: treat first channel as the spectrogram for masking
        x_mono = x[:, 0, :, :]  # (B, H, W)
        for fm in self.freq_masks:
            x_mono = fm(x_mono)
        for tm in self.time_masks:
            x_mono = tm(x_mono)
        # Broadcast back to all channels
        x = x.clone()
        mask = (x_mono == 0)  # where masking set to 0
        x[:, :, mask[0]] = 0  # rough broadcast — good enough for augmentation
        return x


# ── Training utilities ────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]🖥  GPU: {torch.cuda.get_device_name(0)}[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠  No GPU found — training on CPU (will be slow)[/yellow]")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    spec_augment: SpecAugment | None,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if spec_augment is not None:
            images = spec_augment(images)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Validation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    epochs: int = cfg.NUM_EPOCHS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    use_amp: bool = cfg.USE_AMP,
    unfreeze_epoch: int = 5,
    resume: bool = False,
) -> None:
    console.rule("[bold cyan]Spectofind — Training[/bold cyan]")

    device = get_device()
    torch.backends.cudnn.benchmark = True

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    console.print("[dim]Loading dataset …[/dim]")
    train_ds = SpectrogramDataset(folds=cfg.TRAIN_FOLDS, augment=True)
    val_ds   = SpectrogramDataset(folds=[cfg.VAL_FOLD],  augment=False)
    console.print(
        f"[dim]  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples[/dim]"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(device)
    total_p, trainable_p = count_parameters(model)
    console.print(
        f"[dim]  Model: EfficientNet-B0 | Total params: {total_p:,} | "
        f"Trainable: {trainable_p:,}[/dim]"
    )

    spec_aug = SpecAugment().to(device) if device.type == "cuda" else None

    # ── Resume: load checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    history: list[dict] = []
    already_unfrozen = False

    if resume:
        ckpt_path = cfg.LAST_CKPT
        if not ckpt_path.exists():
            console.print("[red]No checkpoint found at checkpoints/last_model.pth — starting fresh.[/red]")
            resume = False
        else:
            console.print(f"[cyan]Resuming from {ckpt_path} ...[/cyan]")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_val_acc  = ckpt.get("best_val_acc", 0.0)
            history       = ckpt.get("history", [])
            already_unfrozen = ckpt.get("unfrozen", False) or start_epoch > unfreeze_epoch
            console.print(
                f"[dim]  Resuming from epoch {start_epoch} | "
                f"Best val acc so far: {best_val_acc*100:.2f}%[/dim]"
            )

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if already_unfrozen or resume:
        # Already past warm-up: unfreeze everything and use CosineAnnealingLR
        model.unfreeze()
        fine_tune_lr = lr * 0.1
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=fine_tune_lr, weight_decay=cfg.WEIGHT_DECAY
        )
        remaining_epochs = epochs - start_epoch + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(remaining_epochs, 1), eta_min=1e-6
        )
        console.print(
            f"[yellow]Backbone unfrozen | CosineAnnealingLR | lr={fine_tune_lr:.2e} for {remaining_epochs} epochs[/yellow]"
        )
    else:
        # Fresh start: OneCycleLR warm-up for first `unfreeze_epoch` epochs
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.MAX_LR,
            epochs=unfreeze_epoch,
            steps_per_epoch=len(train_loader),
            pct_start=cfg.PCT_START,
            anneal_strategy="cos",
        )

    if resume and "optimizer_state_dict" in (ckpt if resume else {}):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            console.print("[dim]  Optimizer state restored.[/dim]")
        except Exception:
            console.print("[dim]  Optimizer state skipped (architecture change).[/dim]")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # ── Training loop ─────────────────────────────────────────────────────────

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Epoch",    justify="right", style="cyan",  width=7)
    table.add_column("Train L",  justify="right", style="white", width=9)
    table.add_column("Train Acc",justify="right", style="green", width=10)
    table.add_column("Val L",    justify="right", style="white", width=9)
    table.add_column("Val Acc",  justify="right", style="bold green", width=10)
    table.add_column("LR",       justify="right", style="dim",   width=10)
    table.add_column("Time",     justify="right", style="dim",   width=7)

    total_epochs = epochs
    console.print(f"\n[bold]Training epochs {start_epoch} → {total_epochs} ...[/bold]\n")

    for epoch in range(start_epoch, total_epochs + 1):
        # Unfreeze backbone on schedule (only relevant for fresh runs)
        if epoch == unfreeze_epoch and not already_unfrozen:
            already_unfrozen = True
            model.unfreeze()
            fine_tune_lr = lr * 0.1
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=fine_tune_lr, weight_decay=cfg.WEIGHT_DECAY
            )
            remaining_epochs = total_epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining_epochs, 1), eta_min=1e-6
            )
            console.print(
                f"[yellow]Epoch {epoch}: Backbone unfrozen — switching to CosineAnnealingLR[/yellow]"
            )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, spec_aug, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(
            dict(
                epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                val_loss=val_loss, val_acc=val_acc, lr=current_lr,
            )
        )

        table.add_row(
            str(epoch),
            f"{tr_loss:.4f}",
            f"{tr_acc*100:.2f}%",
            f"{val_loss:.4f}",
            f"{val_acc*100:.2f}%",
            f"{current_lr:.2e}",
            f"{elapsed:.0f}s",
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":  val_acc,
                    "num_classes": cfg.NUM_CLASSES,
                    "class_names": get_class_names(),
                },
                cfg.BEST_CKPT,
            )

        # Always save last checkpoint (full state for resume)
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc":         best_val_acc,
                "history":              history,
                "unfrozen":             already_unfrozen,
                "num_classes":          cfg.NUM_CLASSES,
            },
            cfg.LAST_CKPT,
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Training complete! Best val accuracy: {best_val_acc*100:.2f}%[/bold green]"
    )
    console.print(f"[dim]  Checkpoint saved → {cfg.BEST_CKPT}[/dim]")

    # Save training history plot
    _save_history_plot(history)


def _save_history_plot(history: list[dict]) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Spectofind — Training History", fontsize=13, fontweight="bold")

    ax1.plot(epochs, [h["tr_loss"] for h in history], label="Train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["tr_acc"]*100 for h in history], label="Train")
    ax2.plot(epochs, [h["val_acc"]*100 for h in history], label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    out = cfg.RESULTS_DIR / "training_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"[dim]  Training curve saved → {out}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spectofind model")
    parser.add_argument("--epochs",     type=int,   default=cfg.NUM_EPOCHS,  help="Total epochs to train up to")
    parser.add_argument("--batch-size", type=int,   default=cfg.BATCH_SIZE,  help="Batch size")
    parser.add_argument("--lr",         type=float, default=cfg.LEARNING_RATE, help="Base learning rate")
    parser.add_argument("--no-amp",     action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume",     action="store_true", help="Resume from checkpoints/last_model.pth")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=not args.no_amp,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
