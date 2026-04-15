"""
dataset.py — ESC-50 download, extraction, and PyTorch Dataset.

Usage (standalone):
    uv run python -m spectofind.dataset
"""

from __future__ import annotations

import csv
import io
import shutil
import zipfile
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from torch.utils.data import Dataset
from torchvision import transforms

from spectofind import config as cfg

console = Console()


# ── Download & Extract ────────────────────────────────────────────────────────

def download_esc50(force: bool = False) -> None:
    """Download and extract the ESC-50 dataset if not already present."""
    # Already fully extracted
    if cfg.AUDIO_DIR.exists() and any(cfg.AUDIO_DIR.glob("*.wav")):
        if not force:
            console.print(
                f"[green]OK ESC-50 already downloaded at[/green] {cfg.ESC50_DIR}"
            )
            return

    # Already downloaded + extracted but not yet renamed
    extracted = cfg.DATA_DIR / "ESC-50-master"
    if extracted.exists() and not cfg.ESC50_DIR.exists() and not force:
        console.print("[cyan]ESC-50-master found — skipping download, just renaming ...[/cyan]")
        _move_extracted(extracted)
        n = len(list(cfg.AUDIO_DIR.glob("*.wav")))
        console.print(f"[green]OK Moved {n} audio files to {cfg.ESC50_DIR}[/green]")
        return

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Downloading ESC-50 (~600 MB) ...[/cyan]")
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("ESC-50", total=None)
        response = requests.get(cfg.ESC50_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        progress.update(task, total=total)
        
        buf = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 256):
            buf.write(chunk)
            progress.advance(task, len(chunk))
    
    console.print("[cyan]Extracting ...[/cyan]")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(cfg.DATA_DIR)

    extracted = cfg.DATA_DIR / "ESC-50-master"
    _move_extracted(extracted)

    n = len(list(cfg.AUDIO_DIR.glob("*.wav")))
    console.print(f"[green]OK Extracted {n} audio files to {cfg.ESC50_DIR}[/green]")


def _move_extracted(extracted: Path) -> None:
    """Move ESC-50-master -> ESC-50 using shutil.move (Windows-safe)."""
    if extracted.exists() and not cfg.ESC50_DIR.exists():
        console.print("[dim]  Renaming ESC-50-master -> ESC-50 ...[/dim]")
        shutil.move(str(extracted), str(cfg.ESC50_DIR))
    elif extracted.exists() and cfg.ESC50_DIR.exists():
        # Partial previous run: merge missing items then clean up
        for item in extracted.iterdir():
            dest = cfg.ESC50_DIR / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(extracted), ignore_errors=True)


def verify_esc50() -> bool:
    """Return True if dataset looks complete (2000 WAV files, meta CSV)."""
    wavs = list(cfg.AUDIO_DIR.glob("*.wav"))
    return len(wavs) == 2000 and cfg.META_CSV.exists()


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_metadata() -> list[dict]:
    """Return list of dicts with keys: filename, fold, target, category."""
    rows = []
    with open(cfg.META_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "filename": row["filename"],
                    "fold":     int(row["fold"]),
                    "target":   int(row["target"]),
                    "category": row["category"],
                }
            )
    return rows


def get_class_names() -> list[str]:
    """Return 50 class names sorted by label index."""
    meta = load_metadata()
    mapping: dict[int, str] = {}
    for row in meta:
        mapping[row["target"]] = row["category"]
    return [mapping[i] for i in range(cfg.NUM_CLASSES)]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Load pre-computed Mel-spectrogram PNG images for a given set of folds.
    
    Pre-compute spectrograms first with:
        uv run python -m spectofind.preprocessing
    """

    # ImageNet normalisation (we fine-tune a pretrained model)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, folds: list[int], augment: bool = False):
        self.augment = augment
        meta = load_metadata()
        self.samples: list[tuple[Path, int]] = []

        for row in meta:
            if row["fold"] in folds:
                img_path = cfg.SPECTROGRAM_DIR / f"{row['filename']}.png"
                if img_path.exists():
                    self.samples.append((img_path, row["target"]))

        if not self.samples:
            raise FileNotFoundError(
                f"No spectrogram PNGs found in {cfg.SPECTROGRAM_DIR}. "
                "Run:  uv run python -m spectofind.preprocessing"
            )

        self.transform = self._build_transform(augment)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        base = [
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ]
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            return transforms.Compose(aug + base)
        return transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold cyan]Spectofind — Dataset Setup[/bold cyan]")
    download_esc50()
    ok = verify_esc50()
    if ok:
        console.print("[green bold]OK ESC-50 dataset is complete (2000 files).[/green bold]")
        names = get_class_names()
        console.print(f"[dim]Classes: {', '.join(names[:10])} … ({len(names)} total)[/dim]")
    else:
        console.print("[red]✗ Dataset appears incomplete. Try with force=True.[/red]")


if __name__ == "__main__":
    main()
