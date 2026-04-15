"""Central configuration — edit this file to change hyperparameters."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR         = ROOT_DIR / "data"
ESC50_DIR        = DATA_DIR / "ESC-50"
AUDIO_DIR        = ESC50_DIR / "audio"
META_CSV         = ESC50_DIR / "meta" / "esc50.csv"
SPECTROGRAM_DIR  = ROOT_DIR / "spectrograms"
CHECKPOINT_DIR   = ROOT_DIR / "checkpoints"
RESULTS_DIR      = ROOT_DIR / "results"

# Create output dirs if missing
for _d in (SPECTROGRAM_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── ESC-50 Download ───────────────────────────────────────────────────────────
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP = DATA_DIR / "ESC-50-master.zip"

# ── Audio / Spectrogram ───────────────────────────────────────────────────────
SAMPLE_RATE      = 22050   # Hz — librosa default, enough for all env. sounds
CLIP_DURATION    = 5.0     # seconds (all ESC-50 clips are exactly 5 s)
N_MELS           = 128     # mel filter banks
HOP_LENGTH       = 512     # STFT hop
N_FFT            = 2048    # FFT window size
F_MIN            = 20      # Hz lower bound
F_MAX            = 8000    # Hz upper bound (captures most env. sounds well)
IMG_SIZE         = 224     # pixels — EfficientNet-B0 native input size

# ── Dataset split ─────────────────────────────────────────────────────────────
# ESC-50 has 5 pre-defined folds. We train on folds 1-4 and validate on fold 5.
TRAIN_FOLDS  = [1, 2, 3, 4]
VAL_FOLD     = 5

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES      = 50
MODEL_NAME       = "efficientnet_b0"   # timm model name
PRETRAINED       = True

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 64
NUM_EPOCHS       = 30
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-4
NUM_WORKERS      = 4
PIN_MEMORY       = True
# Mixed precision — set False if you hit NaN issues (very unlikely on 4060)
USE_AMP          = True
# OneCycleLR config
MAX_LR           = 1e-3
PCT_START        = 0.3    # fraction of cycle spent in warm-up

# ── Checkpointing ─────────────────────────────────────────────────────────────
BEST_CKPT        = CHECKPOINT_DIR / "best_model.pth"
LAST_CKPT        = CHECKPOINT_DIR / "last_model.pth"

# ── SpecAugment (on-the-fly augmentation applied to spectrogram images) ───────
FREQ_MASK_PARAM  = 27    # max consecutive mel bands to mask
TIME_MASK_PARAM  = 50    # max consecutive time steps to mask

# ── Inference ─────────────────────────────────────────────────────────────────
MIC_DURATION     = 5.0   # seconds to record from mic per inference call
MIC_CHUNK_HOP    = 2.5   # seconds between sliding inference windows (streaming)
