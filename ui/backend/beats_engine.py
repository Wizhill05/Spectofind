"""
beats_engine.py — BEATs (Microsoft) inference wrapper.

Auto-downloads the fine-tuned BEATs_iter3+ (AS2M) checkpoint on first use.
Provides the same predict_from_array() interface as InferenceEngine.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchaudio

from spectofind import config as cfg
from spectofind.beats.BEATs import BEATs, BEATsConfig

# Fine-tuned BEATs_iter3+ (AS2M) — best publicly available checkpoint
BEATS_CHECKPOINT_URL = (
    "https://huggingface.co/THUdyh/Ola_speech_encoders/resolve/main/"
    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
BEATS_CHECKPOINT_NAME = "beats_iter3_plus_as2m_finetuned.pt"
BEATS_TARGET_SR = 16000  # BEATs expects 16 kHz mono audio

AUDIOSET_ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
AUDIOSET_ONTOLOGY_NAME = "ontology.json"


class BeatsEngine:
    """Wraps the BEATs model for real-time audio classification."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = cfg.CHECKPOINT_DIR / BEATS_CHECKPOINT_NAME

        # Auto-download if missing
        if not ckpt_path.exists():
            print(f"[BEATs] Downloading checkpoint to {ckpt_path} ...")
            cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(BEATS_CHECKPOINT_URL, str(ckpt_path))
            print(f"[BEATs] Download complete ({ckpt_path.stat().st_size / 1e6:.0f} MB)")

        # Load model
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model_cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(model_cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

        # Label dict: {int_idx: "AudioSet label id (e.g., /m/09x0r)"}
        self.label_dict: dict[int, str] = checkpoint.get("label_dict", {})

        # Load AudioSet ontology for human-readable names
        ontology_path = cfg.CHECKPOINT_DIR / AUDIOSET_ONTOLOGY_NAME
        if not ontology_path.exists():
            print(f"[BEATs] Downloading AudioSet ontology to {ontology_path} ...")
            urllib.request.urlretrieve(AUDIOSET_ONTOLOGY_URL, str(ontology_path))

        with open(ontology_path, "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
            self.id_to_name: dict[str, str] = {item["id"]: item["name"] for item in ontology_data}

        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[BEATs] Loaded on {self.device} "
            f"| {len(self.label_dict)} classes "
            f"| {total_params / 1e6:.1f}M params"
        )

    @torch.no_grad()
    def predict_from_array(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5
    ) -> list[dict]:
        """Classify raw PCM float32 audio. Returns list of {class_name, confidence}."""
        # Convert to torch tensor
        waveform = torch.from_numpy(audio).float()

        # Ensure mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)

        # Resample to 16kHz if needed
        if sample_rate != BEATS_TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=BEATS_TARGET_SR
            )
            waveform = resampler(waveform)

        # BEATs expects (batch, samples)
        waveform = waveform.unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=self.device)

        # Run inference
        probs, _ = self.model.extract_features(waveform, padding_mask=padding_mask)

        # Get top-k predictions
        top_probs, top_indices = probs.squeeze(0).topk(min(top_k, probs.size(-1)))

        results = []
        for idx, prob in zip(top_indices, top_probs):
            label_id = self.label_dict.get(idx.item(), f"class_{idx.item()}")
            label_name = self.id_to_name.get(label_id, label_id)
            results.append({
                "class_name": label_name,
                "confidence": round(prob.item(), 4),
            })

        return results
