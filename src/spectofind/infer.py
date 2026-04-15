"""
infer.py — Single-file and live microphone inference for Spectofind.

Usage:
    # Classify a WAV file
    uv run python -m spectofind.infer --file path/to/audio.wav

    # Live microphone (records 5 s, classifies, repeats)
    uv run python -m spectofind.infer --mic

    # Show top-5 predictions
    uv run python -m spectofind.infer --file audio.wav --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from spectofind import config as cfg
from spectofind.model import load_checkpoint
from spectofind.preprocessing import wav_to_spectrogram_image, array_to_spectrogram_image

console = Console()

# ImageNet normalisation
_TRANSFORM = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Core inference ────────────────────────────────────────────────────────────

def predict_from_image(
    image,   # PIL.Image
    model: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Run model on a PIL image and return top-k (class_name, probability) pairs.
    """
    tensor = _TRANSFORM(image).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.to(memory_format=torch.channels_last)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor)
    
    probs = F.softmax(logits, dim=1).squeeze()
    top_probs, top_indices = probs.topk(top_k)

    return [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]


def _print_predictions(predictions: list[tuple[str, float]], source: str) -> None:
    table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
    table.add_column("Rank",  style="dim",   width=4)
    table.add_column("Class", style="cyan",  width=22)
    table.add_column("Confidence", width=8)
    table.add_column("Bar", width=30)

    for rank, (name, prob) in enumerate(predictions, 1):
        pct = prob * 100
        bar_len = int(pct / 100 * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        color = "bold green" if rank == 1 else ("yellow" if rank == 2 else "dim")
        table.add_row(
            f"#{rank}",
            name.replace("_", " ").title(),
            f"[{color}]{pct:.1f}%[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(
        Panel(table, title=f"[bold]🔊 {source}[/bold]", border_style="cyan", padding=(0, 1))
    )


# ── File inference ────────────────────────────────────────────────────────────

def infer_file(wav_path: Path, model, device, class_names, top_k: int = 3) -> None:
    console.print(f"[dim]Converting {wav_path.name} to spectrogram …[/dim]")
    img = wav_to_spectrogram_image(wav_path)
    preds = predict_from_image(img, model, device, class_names, top_k)
    _print_predictions(preds, wav_path.name)


# ── Microphone inference ──────────────────────────────────────────────────────

def infer_mic(model, device, class_names, top_k: int = 3) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        console.print(
            "[red]sounddevice not installed. Run: uv add sounddevice[/red]"
        )
        sys.exit(1)

    console.print(
        "[bold cyan]🎤 Live microphone inference[/bold cyan] — "
        f"recording {cfg.MIC_DURATION:.0f} s per clip. Press Ctrl+C to stop.\n"
    )

    while True:
        try:
            console.print("[dim]▶ Recording …[/dim]", end="\r")
            audio = sd.rec(
                int(cfg.MIC_DURATION * cfg.SAMPLE_RATE),
                samplerate=cfg.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()
            img = array_to_spectrogram_image(audio, cfg.SAMPLE_RATE)
            preds = predict_from_image(img, model, device, class_names, top_k)
            _print_predictions(preds, "Microphone")
            time.sleep(0.2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped.[/yellow]")
            break


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectofind — Sound inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--file", type=str, help="Path to a WAV audio file")
    grp.add_argument("--mic",  action="store_true", help="Use live microphone input")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold cyan]Spectofind — Inference[/bold cyan]")
    console.print(f"[dim]Device: {device}[/dim]")

    model = load_checkpoint(args.checkpoint, device=device)

    # Load class names from checkpoint if available
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.BEST_CKPT
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names", None)
    if class_names is None:
        from spectofind.dataset import get_class_names
        class_names = get_class_names()

    if args.mic:
        infer_mic(model, device, class_names, top_k=args.top_k)
    else:
        wav = Path(args.file)
        if not wav.exists():
            console.print(f"[red]File not found: {wav}[/red]")
            sys.exit(1)
        infer_file(wav, model, device, class_names, top_k=args.top_k)


if __name__ == "__main__":
    main()
