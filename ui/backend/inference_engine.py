"""
inference_engine.py — Wraps the trained model for inference.

Used by the FastAPI backend. Loaded once at server startup and
shared across all requests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from spectofind import config as cfg
from spectofind.dataset import get_class_names
from spectofind.model import SpectroNet
from spectofind.preprocessing import (
    array_to_spectrogram_image,
    wav_to_spectrogram_image,
)


class InferenceEngine:
    """Thread-safe model wrapper for spectrogram-based audio classification."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = Path(checkpoint_path) if checkpoint_path else cfg.BEST_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.class_names: list[str] = ckpt.get("class_names", None) or get_class_names()
        num_classes = ckpt.get("num_classes", cfg.NUM_CLASSES)

        self.model = SpectroNet(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        if self.device.type != "cpu":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f"[InferenceEngine] Loaded on {self.device} "
              f"| {num_classes} classes | from {ckpt_path.name}")

    @torch.no_grad()
    def predict_from_file(self, wav_path: Path, top_k: int = 5) -> list[dict]:
        """Classify a WAV file. Returns list of {class_name, confidence}."""
        img = wav_to_spectrogram_image(wav_path)
        return self._predict(img, top_k)

    @torch.no_grad()
    def predict_from_array(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5
    ) -> list[dict]:
        """Classify raw PCM float32 audio. Returns list of {class_name, confidence}."""
        img = array_to_spectrogram_image(audio, sample_rate)
        return self._predict(img, top_k)

    def _predict(self, img, top_k: int) -> list[dict]:
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self.device.type != "cpu":
            tensor = tensor.to(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1).squeeze()
        top_probs, top_indices = probs.topk(min(top_k, len(probs)))

        return [
            {
                "class_name": self.class_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

    # ── Dashboard helpers ─────────────────────────────────────────────────────

    @staticmethod
    def get_checkpoint_info() -> dict:
        """Read checkpoint metadata for the dashboard (no model loading)."""
        info = {
            "model_name": "EfficientNet-B0",
            "num_classes": cfg.NUM_CLASSES,
            "total_params": 0,
            "best_epoch": 0,
            "best_val_acc": 0.0,
            "total_epochs": 0,
            "history": [],
            "class_names": [],
        }

        if cfg.BEST_CKPT.exists():
            ckpt = torch.load(cfg.BEST_CKPT, map_location="cpu", weights_only=False)
            info["best_epoch"] = ckpt.get("epoch", 0)
            info["best_val_acc"] = round(ckpt.get("val_acc", 0.0), 4)
            info["class_names"] = ckpt.get("class_names", [])

        if cfg.LAST_CKPT.exists():
            ckpt = torch.load(cfg.LAST_CKPT, map_location="cpu", weights_only=False)
            info["total_epochs"] = ckpt.get("epoch", 0)
            info["history"] = ckpt.get("history", [])

        # Count params from model architecture (no checkpoint needed)
        from spectofind.model import count_parameters
        m = SpectroNet(pretrained=False)
        total, _ = count_parameters(m)
        info["total_params"] = total

        return info
