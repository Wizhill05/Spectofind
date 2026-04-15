# 🔊 Spectofind

**Spectrogram-based environmental sound classifier** using transfer learning on EfficientNet-B0.

Converts raw audio into Mel-spectrogram images and applies a pretrained computer vision model to classify 50 classes of environmental sounds — breathing, fan, rain, keyboard, dog bark, and more.

---

## How It Works

```
Audio WAV ──► Mel-Spectrogram (PNG) ──► EfficientNet-B0 ──► 50 class prediction
                  (cached pre-computed)     (ImageNet pretrained)
```

The key insight: **spectrograms are just images**. By converting audio to colour-coded time-frequency images, we can leverage the full power of ImageNet-pretrained CV models — no need to train from scratch.

---

## Dataset: ESC-50

- **2,000 clips** × 5 seconds each, 44.1 kHz mono
- **50 classes** across 5 categories: Animals, Nature, Human sounds, Interior, Exterior
- Download size: ~600 MB
- License: CC BY (free to use)
- Auto-downloaded on first run

---

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python ≥ 3.10.

```bash
# Install dependencies (first time only)
uv sync
```

> **GPU Note:** This project uses CUDA. Install PyTorch with CUDA support:
> ```bash
> uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Workflow

### 1. Download the dataset
```bash
uv run python -m spectofind.dataset
```
Downloads and verifies ESC-50 into `data/ESC-50/`.

### 2. Pre-compute spectrograms
```bash
uv run python -m spectofind.preprocessing
```
Converts all 2,000 WAV files to colour PNG spectrograms in `spectrograms/`.  
This runs **once** — subsequent training epochs load PNGs directly (fast!).

### 3. Train
```bash
uv run python -m spectofind.train
```
Trains EfficientNet-B0 on folds 1–4, validates on fold 5.  
- **Expected time:** ~15–20 minutes on RTX 4060  
- **Expected accuracy:** ~82–87% on ESC-50 fold 5

Options:
```bash
uv run python -m spectofind.train --epochs 40 --batch-size 64
```

### 4. Evaluate
```bash
uv run python -m spectofind.evaluate
```
Prints per-class accuracy table and saves `results/confusion_matrix.png`.

### 5. Classify a sound file
```bash
uv run python -m spectofind.infer --file path/to/audio.wav
uv run python -m spectofind.infer --file path/to/audio.wav --top-k 5
```

### 6. Live microphone inference
```bash
uv run python -m spectofind.infer --mic
```
Records 5-second chunks from your mic and classifies in real time.

---

## Training Optimizations

| Optimization | Benefit |
|---|---|
| Pre-computed spectrogram PNGs | No per-epoch audio decoding |
| Mixed Precision (AMP) | ~2× speed boost, lower VRAM |
| `cudnn.benchmark = True` | Auto-selects fastest convolution kernels |
| `pin_memory` + `num_workers=4` | Fast CPU→GPU data transfer |
| `channels_last` memory format | Faster convolutions on Ampere GPUs (RTX 40xx) |
| Frozen early backbone (first 5 epochs) | Head warms up before full fine-tuning |
| SpecAugment (freq + time masking) | Better generalization, ~3–5% accuracy boost |
| OneCycleLR scheduler | Fast convergence |

---

## Project Structure

```
Spectofind/
├── src/spectofind/
│   ├── config.py          # All hyperparameters
│   ├── dataset.py         # ESC-50 download + PyTorch Dataset
│   ├── preprocessing.py   # Audio → Mel-spectrogram pipeline
│   ├── model.py           # EfficientNet-B0 + classifier head
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Per-class accuracy + confusion matrix
│   └── infer.py           # File + live mic inference
├── data/ESC-50/           # Dataset (auto-downloaded)
├── spectrograms/          # Pre-computed PNG spectrograms
├── checkpoints/           # Saved model weights
└── results/               # training_history.png, confusion_matrix.png
```

---

## ESC-50 Classes

| Category | Classes |
|---|---|
| 🐾 Animals | dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow |
| 🌊 Nature | rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, thunderstorm |
| 👤 Human | crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping |
| 🏠 Interior | door_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking |
| 🏙 Exterior | helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw |
