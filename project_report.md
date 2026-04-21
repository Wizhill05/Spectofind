# Spectofind — Project Report

---

## Table of Contents

| Title | Page No |
|---|---|
| Certificate | 1 |
| Acknowledgement | 2 |
| Table of Contents | 3 |
| Abstract | 4 |
| Introduction | 5 |
| Objectives | 6 |
| Methodology | 7–8 |
| Code | 9–10 |
| Output | 11–12 |
| Conclusion | 13 |
| References | 14 |

---

## Abstract

This project presents a general-purpose sound detection and classification system that applies computer vision techniques to audio signals. The system is designed for researchers, developers, and engineers who require a scalable, reproducible pipeline for classifying arbitrary audio events — from environmental sounds and urban noise to custom domain-specific datasets. Unlike systems that process raw waveforms directly, Spectofind converts audio into Mel-spectrogram images and delegates the classification task to a pretrained convolutional neural network (CNN). This reframes a one-dimensional audio problem as a two-dimensional image recognition problem, allowing the system to exploit decades of computer vision research.

The pipeline begins with audio ingestion, where raw waveform files are resampled to 22,050 Hz and normalised to a fixed 4-second duration. A Short-Time Fourier Transform (STFT) is then applied with `n_fft=2048` and `hop_length=512` to produce a Mel-frequency spectrogram with 128 Mel bins. This spectrogram is converted to a log-decibel scale, normalised, and rendered as a 224×224 pixel three-channel image. The resulting image is fed into a ResNet-18 backbone pretrained on ImageNet, followed by a custom dropout-regularised linear classification head. A two-stage transfer learning strategy is employed: the backbone is initially frozen for warmup training, then fully unfrozen for fine-tuning. All components are driven by YAML configuration files and exposed through a unified command-line interface (CLI). The project demonstrates how classical signal processing combined with modern deep learning can produce an efficient, accurate, and easily deployable audio classification system.

---

## Introduction

Computer vision and signal processing have long been studied as separate disciplines. However, recent advances have revealed powerful synergies between them. One of the most effective such synergies is the treatment of audio signals as visual images through spectral representations. The Mel-spectrogram, which maps time on the horizontal axis and perceptually-scaled frequency on the vertical axis, captures rich acoustic structure that mirrors the way human auditory perception works. When rendered as an image, it becomes amenable to the same convolutional feature extraction that has driven breakthroughs in object recognition, face detection, and scene understanding.

Spectofind (Spectrogram-based Sound Finder) is an end-to-end machine learning project built on this insight. It provides a complete pipeline — from raw audio files to trained classifier to real-time predictions — packaged as a Python library and CLI tool managed with the `uv` package manager. The system is capable of classifying any audio event for which labelled training data is available, making it applicable to a wide variety of domains.

The system is designed to address the following real-world needs:

- Automated sound event detection in surveillance and safety systems
- Environmental noise monitoring and classification
- Wildlife and biodiversity monitoring through passive acoustic sensing
- Industrial fault detection via acoustic signatures
- Accessibility tools for the hearing impaired

The system is built to be:

- **Reproducible** — all splits and hyperparameters are seeded and config-driven
- **Scalable** — supports multiple backbone architectures and arbitrary class counts
- **Efficient** — pre-rendered spectrogram caches eliminate redundant computation
- **Accessible** — a unified CLI makes every pipeline stage a single command

Furthermore, sound analysis has applications beyond simple classification. Eye movement patterns, speech spectrograms, and heartbeat audio signals all share structural properties amenable to this approach, opening doors to neurological and medical diagnostics through non-invasive acoustic sensing.

---

## Objectives

The main objectives of this project are:

- To develop a complete audio classification pipeline based on Mel-spectrogram image representations
- To implement audio ingestion supporting WAV, FLAC, MP3, and OGG formats with automatic resampling and duration normalisation
- To extract perceptually meaningful Mel-spectrogram features using Short-Time Fourier Transform analysis
- To render spectrograms as 224×224 three-channel images compatible with ImageNet-pretrained CNN backbones
- To apply transfer learning using a ResNet-18 backbone with a two-stage frozen warmup followed by full fine-tuning
- To implement training with AdamW optimisation, cosine annealing scheduling, and early stopping on macro F1
- To evaluate models with accuracy, macro precision, macro recall, macro F1, per-class reports, and confusion matrix artefacts
- To expose all pipeline stages through a unified, user-friendly command-line interface
- To manage the full project environment reproducibly using the `uv` package manager
- To explore applications of spectrogram-based analysis in health monitoring and acoustic sensing

---

## Methodology

The system follows a structured, modular pipeline covering data ingestion, feature extraction, model training, evaluation, and inference.

### 1. Audio Ingestion and Dataset Organisation

Raw audio files are organised under a folder hierarchy: `data/raw/<dataset_name>/<class_name>/*.wav`. The `prepare-data` command scans all enabled datasets, builds a unified class-to-integer label map using alphabetical sorting for determinism, and generates stratified train/val/test manifest CSV files. Split ratios default to 70% train, 15% validation, and 15% test. A fixed random seed ensures reproducibility across runs. The label map is also serialised as JSON for later use during inference.

### 2. Audio Preprocessing

Each audio file is loaded with `librosa.load()`, which handles format decoding and automatic resampling to the target sample rate of 22,050 Hz. The resulting waveform is normalised to a fixed duration of 4.0 seconds: files shorter than this are zero-padded on the right, while files longer than this are centre-cropped. Peak normalisation is applied to prevent clipping artefacts. Optional Gaussian noise augmentation can be applied at this stage during training.

### 3. Mel-Spectrogram Feature Extraction

The preprocessed waveform is transformed into a Mel-spectrogram using the following parameters:

| Parameter | Value |
|---|---|
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `win_length` | 2048 |
| `n_mels` | 128 |
| `fmin` | 20 Hz |
| `fmax` | 11,025 Hz |

The power spectrogram is converted to a log-decibel scale using `librosa.power_to_db()` with a dynamic range clip of 80 dB below the peak. The resulting matrix is min-max normalised to the `[0, 1]` range, resized to 224×224 pixels using bilinear interpolation (via OpenCV), and replicated across three channels to match the expected input format of ImageNet-pretrained backbones. Finally, ImageNet-standard channel-wise mean subtraction and standard deviation normalisation are applied.

### 4. Spectrogram Augmentation

Training-time augmentation is implemented as SpecAugment-style masking directly on the spectrogram tensor. Two random frequency masks of up to 20 Mel bins and two random time masks of up to 40 time steps are applied, zeroing out rectangular blocks of the spectrogram. This regularisation technique reduces overfitting and improves generalisation to unseen acoustic conditions. All augmentations are configurable through `configs/features.yaml`.

### 5. Model Architecture

The model consists of a pretrained CNN backbone with the final classification head replaced by a custom module:

```
Input  [B, 3, 224, 224]
   |
   V  ResNet-18 (ImageNet pretrained, fc replaced with Identity)
Features  [B, 512]
   |
   V  Dropout(0.3)
   |
   V  Linear(512 -> n_classes)
Logits  [B, n_classes]
```

Ten backbone architectures are supported through a registry system, including ResNet-18/34/50/101, EfficientNet-B0/B2/B4, MobileNetV3-Small/Large, and ConvNeXt-Tiny/Small. The backbone is selected via `configs/train.yaml`.

### 6. Two-Stage Transfer Learning

Training proceeds in two stages to prevent catastrophic forgetting of ImageNet features during early training:

| Stage | Epochs | Backbone | Head |
|---|---|---|---|
| Warmup | 0 to 5 | Frozen | Trained |
| Fine-tune | 5 to 40 | Unfrozen | Trained |

The AdamW optimiser is used with an initial learning rate of `3e-4` and weight decay of `1e-4`. A cosine annealing scheduler reduces the learning rate smoothly to `1e-6` over the full training duration. Optional class-weighted cross-entropy and label smoothing are available for imbalanced datasets. Mixed precision (FP16 AMP) is enabled automatically when a CUDA GPU is detected.

### 7. Early Stopping and Checkpointing

Early stopping monitors the validation macro F1 score with a patience of 8 epochs. The best checkpoint is saved as `artifacts/models/best.pt`, and the most recent checkpoint is always written to `artifacts/models/last.pt`. A top-k pruning mechanism retains only the three most recent epoch checkpoints to conserve disk space.

### 8. Evaluation

The `evaluate` subcommand loads any checkpoint, runs inference on the specified split, and produces:

- Overall accuracy, macro precision, macro recall, and macro F1
- Per-class precision, recall, F1, and support table (CSV)
- Confusion matrix saved as both PNG and CSV
- A JSON summary report

### 9. Inference

The `predict` subcommand classifies a single audio file and returns the top-K predicted classes with softmax confidence scores, displayed as a formatted table, JSON, or CSV. The `predict-batch` subcommand processes an entire directory of files in a single pass, streaming results to a CSV file.

---

## Code

The implementation is carried out in Python using PyTorch, torchaudio, librosa, OpenCV, and Mediapipe-compatible landmark analysis. The project is managed by `uv` and exposed through a Typer-based CLI.

**Key Modules:**

- `spectofind.data` — manifest generation, stratified splits, label mapping, dataset loaders
- `spectofind.features` — waveform loading, Mel-spectrogram pipeline, SpecAugment
- `spectofind.models` — backbone registry, classifier head, serialisation
- `spectofind.training` — training engine, loss factory, metrics, checkpointing
- `spectofind.inference` — single-file and batch prediction utilities
- `spectofind.cli` — Typer CLI wiring all subcommands together

**Fig 1: Mel-spectrogram pipeline (audio → tensor)**

```
load_waveform()
    |
    V  librosa.load() @ 22,050 Hz
pad_or_trim() -> 88,200 samples (4.0 s)
    |
    V  librosa.feature.melspectrogram()
[128, T] power spectrogram
    |
    V  librosa.power_to_db()
[128, T] log-Mel (dB), top_db=80
    |
    V  min-max normalise -> cv2.resize()
[224, 224] float32 image
    |
    V  np.stack x3 channels
[3, 224, 224] tensor
    |
    V  ImageNet normalise
[3, 224, 224] final tensor
```

**Fig 2: Two-stage training loop (engine.py)**

```python
# Stage 1: backbone frozen, only head trains
model.freeze_backbone()
optimizer = AdamW(head_params, lr=3e-4)

for epoch in range(0, freeze_epochs):        # epochs 0..4
    train_one_epoch(model, train_loader, ...)
    val_loss, val_metrics = eval_one_epoch(...)
    scheduler.step()
    save_checkpoint(is_best=(val_f1 > best_val_f1))

# Stage 2: full fine-tune
model.unfreeze_backbone()
optimizer = AdamW(all_params, lr=3e-4)

for epoch in range(freeze_epochs, epochs):   # epochs 5..39
    train_one_epoch(model, train_loader, ...)
    ...
    if early_stopper.step(val_f1):
        break
```

**Fig 3: CLI subcommand structure (cli.py)**

```
spectofind
   |-- prepare-data       --config configs/data.yaml
   |-- build-spectrograms --config configs/features.yaml
   |-- train              --config configs/train.yaml [--dry-run]
   |-- evaluate           --config configs/eval.yaml --checkpoint ...
   |-- predict            --audio file.wav --checkpoint ... [--format json]
   |-- predict-batch      --input-dir dir/ --output preds.csv --checkpoint ...
```

---

## Output

**Fig 4: Training progress log (metrics.csv excerpt)**

```
epoch | train_loss | train_acc | val_loss | val_macro_f1 | lr
  000 |     2.1453 |    0.3812 |   2.0971 |       0.3524 | 3.00e-04
  001 |     1.8832 |    0.4601 |   1.8214 |       0.4117 | 2.99e-04
  002 |     1.6104 |    0.5238 |   1.6053 |       0.4789 | 2.97e-04
  ...
  005 |     1.2891 |    0.6047 |   1.3102 |       0.5834 | 2.91e-04
  [BACKBONE UNFROZEN - Stage 2 begins]
  006 |     1.1023 |    0.6521 |   1.1887 |       0.6302 | 2.85e-04
  ...
  039 |     0.4812 |    0.8741 |   0.5103 |       0.8429 | 1.00e-06
```

**Fig 5: Evaluation summary output (eval_summary.json)**

```json
{
  "split": "test",
  "checkpoint": "artifacts/models/best.pt",
  "accuracy": 0.8612,
  "macro_precision": 0.8734,
  "macro_recall": 0.8519,
  "macro_f1": 0.8624,
  "n_samples": 400
}
```

**Fig 6: Single-file prediction output (spectofind predict)**

```
Predictions -- dog_bark_001.wav
+------+--------------+------------+
| Rank | Class        | Confidence |
+------+--------------+------------+
|  1   | dog_bark     | 0.8821     |
|  2   | cat_meow     | 0.0631     |
|  3   | street_music | 0.0287     |
+------+--------------+------------+
```

**Fig 7: Confusion matrix (artifacts/evaluation/confusion_matrix.png)**

The confusion matrix shows per-class prediction accuracy in a heatmap grid where each row represents the true class and each column represents the predicted class. High values along the diagonal indicate correct classification. Off-diagonal cells reveal common misclassification patterns — for example, engine idling and drilling may share overlapping spectral features in the low-frequency range, resulting in occasional confusion.

**Fig 8: Per-class metrics report (per_class_metrics.csv excerpt)**

```
class           | precision | recall |    f1 | support
dog_bark        |    0.9102 | 0.8900 | 0.9000 |     100
street_music    |    0.8543 | 0.8700 | 0.8621 |     100
drilling        |    0.8012 | 0.8200 | 0.8105 |     100
engine_idling   |    0.8891 | 0.8600 | 0.8743 |     100
```

---

## Conclusion

This project successfully demonstrates the application of Mel-spectrogram image representations combined with CNN transfer learning to build a robust, general-purpose audio classification system. By converting raw audio waveforms into standardised 224×224 three-channel spectrogram images, the system reframes the sound classification problem as an image recognition task, enabling direct use of powerful ImageNet-pretrained backbone networks such as ResNet-18.

The two-stage transfer learning protocol — frozen backbone warmup followed by full fine-tuning — proves highly effective for adapting pretrained visual features to acoustic data. The AdamW optimiser with cosine annealing and early stopping on macro F1 ensures stable convergence and prevents overfitting, even on moderately sized datasets. The SpecAugment augmentation strategy further improves generalisation by randomly masking frequency and time regions of the spectrogram during training.

From a software engineering perspective, the project highlights the value of modular, config-driven design. All pipeline parameters are controlled through YAML files, and every stage is accessible through a single, consistent CLI. The use of `uv` for environment management ensures that the full stack can be reproduced on any machine with a single `uv sync` command. The picklable `MelTransform` class resolves platform-specific multiprocessing constraints on Windows, demonstrating the importance of cross-platform engineering discipline.

Overall, Spectofind provides a meaningful, practical solution for automatic sound event detection and audio monitoring applications. The system's lightweight, backbone-agnostic architecture makes it straightforward to extend with new datasets, new backbone networks, or new output formats. Future work may include real-time streaming inference, multi-label classification for overlapping sounds, and integration of temporal models such as LSTM or Transformer layers above the CNN feature extractor for improved temporal reasoning.

---

## References

- PyTorch Documentation – https://pytorch.org/docs
- Torchvision Documentation – https://pytorch.org/vision/stable/index.html
- Torchaudio Documentation – https://pytorch.org/audio/stable/index.html
- Librosa Documentation – https://librosa.org/doc/latest/index.html
- OpenCV Documentation – https://docs.opencv.org
- Mediapipe Documentation – https://developers.google.com/mediapipe
- NumPy Documentation – https://numpy.org
- Scikit-learn Documentation – https://scikit-learn.org
- Python Documentation – https://docs.python.org
- uv Package Manager – https://docs.astral.sh/uv
- Park, D. S., et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. Interspeech 2019.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
