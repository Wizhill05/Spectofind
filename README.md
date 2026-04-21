<div align="center">

# 🔊 Spectofind

### Advanced Environmental Sound Classification Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/mit)
[![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-React%20%2B%20Vite-61dafb.svg)](https://vitejs.dev/)

<p align="center">
  <strong>Identify environmental sounds in real-time with machine learning</strong>
</p>

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

**Spectofind** is a high-performance environmental sound classifier that uses transfer learning on an EfficientNet-B0 backbone. It converts raw audio into Mel-spectrogram images and applies computer vision techniques to classify 50 different classes of environmental sounds — from breathing and rain to dog barks and sirens.

### 🎯 Use Cases

- **Real-Time Detection**: Monitor microphone inputs and instantly identify surrounding environmental sounds.
- **Audio Analysis**: Batch classify saved `.wav` files.
- **Model Training**: A highly optimized pipeline for fast CNN training on the ESC-50 dataset.
- **Visual Analytics**: Interactive web dashboard to monitor training health, accuracy, and live classification.

## 💡 Why I Made It

Analyzing raw audio waveforms is notoriously difficult and computationally expensive. However, when you convert audio into colour-coded time-frequency images (spectrograms), you can leverage the immense power of ImageNet-pretrained computer vision models.

Spectofind was built to demonstrate how transforming audio into the visual domain allows models like EfficientNet to excel at sound classification without needing to train from scratch. The newly added brutalist web UI takes this a step further, providing a professional, real-time visual interface for model interactions and live telemetry.

## ✨ Features

<table>
<tr>
<td>

### 🖥️ Core Engine

- 🚀 **Pre-computed Spectrograms** for blazing fast training
- 🎛️ **Mixed Precision (AMP)** for low VRAM usage
- 🎯 **50 Sound Classes** support out-of-the-box
- 🎤 **Live Mic Inference** via CLI or Web UI
- 📈 **SpecAugment** for robust model generalization

</td>
<td>

### 🌟 Web Dashboard (New!)

- 📱 **Mobile Responsive** brutalist, sharp monochrome design
- 📊 **Real-time Telemetry** for model evaluation
- 🔊 **Interactive Class Matrix** with audio sample playback
- 🎙️ **Recording Studio** with live spectrogram visualization
- ⚡ **WebSocket Streaming** for low-latency classification

</td>
</tr>
</table>

## 🚀 Installation

### Prerequisites

```bash
# Requires uv package manager and Python >= 3.10
# Install uv if you haven't already:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/spectofind.git
cd spectofind

# Install Python backend dependencies
uv sync

# Install PyTorch with CUDA support (Highly Recommended)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Frontend dependencies
cd ui/frontend
npm install
```

## 📖 Usage

### 🎨 Web Dashboard Mode (Recommended)

Launch the modern graphical interface. This requires running both the backend API and the frontend dev server.

**Terminal 1 (Backend):**

```bash
uv run uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**

```bash
cd ui/frontend
npm run dev
```

Open `http://localhost:5173/` in your browser.

<details>
<summary><strong>GUI Features</strong></summary>

- **Dashboard**: View model parameters, training loss/accuracy charts, and a full class accuracy matrix.
- **Audio Previews**: Click "PLAY SAMPLE" in the Class Matrix to hear dataset examples.
- **Recording Studio**: Stream your microphone to the backend and watch real-time inference telemetry and live spectrograms.
</details>

### 💻 CLI Pipeline

The CLI provides the complete pipeline for dataset management, training, and inference.

#### 1. Download & Prep

```bash
uv run python -m spectofind.dataset
uv run python -m spectofind.preprocessing
```

#### 2. Train & Evaluate

```bash
# Train the model (~15 mins on RTX 4060)
uv run python -m spectofind.train --epochs 30 --batch-size 64

# Evaluate per-class accuracy
uv run python -m spectofind.evaluate
```

#### 3. Inference

```bash
# Classify an audio file
uv run python -m spectofind.infer --file path/to/audio.wav

# Live microphone classification
uv run python -m spectofind.infer --mic
```

## 🏗️ Architecture

```
Spectofind/
├── src/spectofind/        # Core ML Engine
│   ├── config.py          # Global hyperparameters
│   ├── dataset.py         # ESC-50 downloader
│   ├── preprocessing.py   # Audio to Mel-spectrogram
│   └── train.py           # Training loop
├── ui/
│   ├── backend/           # FastAPI & WebSocket server
│   └── frontend/          # React + Vite Brutalist UI
├── data/                  # Auto-downloaded ESC-50
├── spectrograms/          # Pre-computed image cache
└── checkpoints/           # Saved model weights
```

### 🔧 How It Works

1. **Audio Transformation**: `librosa` converts 5s audio chunks into 128-band Mel-spectrogram PNGs.
2. **Transfer Learning**: An EfficientNet-B0 backbone (pretrained on ImageNet) processes the spectrograms.
3. **WebSockets**: The React frontend records audio chunks and streams them via WebSockets to the FastAPI backend.
4. **Real-time Processing**: The backend caches the model in memory, processes incoming streams, and returns predictions and confidence scores instantly.

## 🛠️ Advanced Configuration

Edit `src/spectofind/config.py` to modify system behavior:

```python
BATCH_SIZE = 64
NUM_EPOCHS = 30
SAMPLE_RATE = 22050
IMG_SIZE = 224
MIC_DURATION = 5.0 # Seconds per mic inference chunk
```

## 🐛 Troubleshooting

<details>
<summary><strong>CUDA / PyTorch Issues</strong></summary>

If training is slow, ensure PyTorch is utilizing your GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

</details>

<details>
<summary><strong>Microphone Not Detected (Web UI)</strong></summary>

Ensure you have granted microphone permissions to your browser. Browsers may restrict microphone access on `http` unless running on `localhost`.

</details>

## ⚠️ Known Limitations

- **Microphone Environment Noise**: Live inference is highly sensitive to background static. Ensure a clean audio input.
- **Dataset Bias**: The model is tuned heavily to the ESC-50 dataset. Unseen environmental categories might trigger false positives.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch, FastAPI, React, and Vite.
- Dataset provided by the [ESC-50 project](https://github.com/karoldvl/ESC-50).

## 📞 Support

- 🐛 Report bugs by opening an issue.
- 💡 Request features via GitHub issues.

---
