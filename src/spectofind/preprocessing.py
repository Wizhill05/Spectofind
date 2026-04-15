"""
preprocessing.py — Convert ESC-50 WAV files to Mel-spectrogram PNG images.

Pre-computing spectrograms once means each training epoch simply loads PNGs
(via PIL) instead of decoding audio — dramatically faster on every run.

Usage:
    uv run python -m spectofind.preprocessing
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import track

from spectofind import config as cfg
from spectofind.dataset import load_metadata

# Use non-interactive backend — no display required
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


# ── Core conversion ───────────────────────────────────────────────────────────

def audio_to_melspectrogram(
    audio_path: Path,
    sample_rate: int = cfg.SAMPLE_RATE,
    n_mels: int = cfg.N_MELS,
    hop_length: int = cfg.HOP_LENGTH,
    n_fft: int = cfg.N_FFT,
    f_min: float = cfg.F_MIN,
    f_max: float = cfg.F_MAX,
    duration: float = cfg.CLIP_DURATION,
) -> np.ndarray:
    """
    Load an audio file and compute a log-scale Mel-spectrogram.

    Returns
    -------
    np.ndarray  shape (n_mels, time_frames), dtype float32 (log dB scale)
    """
    y, sr = librosa.load(str(audio_path), sr=sample_rate, duration=duration, mono=True)

    # Pad or trim to exact length
    target_len = int(sample_rate * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=f_min,
        fmax=f_max,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)   # log scale in dB
    return mel_db.astype(np.float32)


def save_spectrogram_image(mel_db: np.ndarray, out_path: Path, img_size: int = cfg.IMG_SIZE) -> None:
    """Save a Mel-spectrogram array as a colourized PNG image (viridis colormap)."""
    fig, ax = plt.subplots(figsize=(img_size / 100, img_size / 100), dpi=100)
    librosa.display.specshow(
        mel_db,
        sr=cfg.SAMPLE_RATE,
        hop_length=cfg.HOP_LENGTH,
        fmin=cfg.F_MIN,
        fmax=cfg.F_MAX,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap="viridis",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ── Batch conversion ──────────────────────────────────────────────────────────

def precompute_all(force: bool = False) -> None:
    """Convert every ESC-50 WAV to a PNG spectrogram, skipping existing files."""
    if not cfg.AUDIO_DIR.exists():
        console.print(
            "[red]✗ ESC-50 audio directory not found. "
            "Run:  uv run python -m spectofind.dataset[/red]"
        )
        return

    cfg.SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()

    to_process = []
    for row in meta:
        wav = cfg.AUDIO_DIR / row["filename"]
        png = cfg.SPECTROGRAM_DIR / f"{row['filename']}.png"
        if force or not png.exists():
            to_process.append((wav, png))

    if not to_process:
        console.print(
            f"[green]✓ All {len(meta)} spectrograms already exist. "
            f"Use force=True to recompute.[/green]"
        )
        return

    console.print(
        f"[cyan]🎵 Converting {len(to_process)} audio files → PNG spectrograms …[/cyan]"
    )
    errors = 0
    for wav, png in track(to_process, description="Processing …", console=console):
        try:
            mel = audio_to_melspectrogram(wav)
            save_spectrogram_image(mel, png)
        except Exception as e:
            console.print(f"[red]  Error on {wav.name}: {e}[/red]")
            errors += 1

    ok = len(to_process) - errors
    console.print(
        f"[green bold]✓ Done — {ok} spectrograms saved to {cfg.SPECTROGRAM_DIR}[/green bold]"
    )
    if errors:
        console.print(f"[yellow]  {errors} errors — check audio files[/yellow]")


def wav_to_spectrogram_image(wav_path: Path) -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Convert a single WAV file to a PIL Image for inference."""
    import io
    from PIL import Image

    mel = audio_to_melspectrogram(wav_path)
    buf = io.BytesIO()

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(
        mel,
        sr=cfg.SAMPLE_RATE,
        hop_length=cfg.HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap="viridis",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf).convert("RGB")


def array_to_spectrogram_image(audio_array: np.ndarray, sample_rate: int) -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Convert a raw numpy audio array to a PIL Image (for live mic inference)."""
    import io
    from PIL import Image

    # Resample if needed
    if sample_rate != cfg.SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=cfg.SAMPLE_RATE)

    # Pad / trim
    target_len = int(cfg.SAMPLE_RATE * cfg.CLIP_DURATION)
    if len(audio_array) < target_len:
        audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))
    else:
        audio_array = audio_array[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio_array.astype(np.float32),
        sr=cfg.SAMPLE_RATE,
        n_mels=cfg.N_MELS,
        hop_length=cfg.HOP_LENGTH,
        n_fft=cfg.N_FFT,
        fmin=cfg.F_MIN,
        fmax=cfg.F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(
        mel_db,
        sr=cfg.SAMPLE_RATE,
        hop_length=cfg.HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap="viridis",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold cyan]Spectofind — Preprocessing[/bold cyan]")
    precompute_all()


if __name__ == "__main__":
    main()
