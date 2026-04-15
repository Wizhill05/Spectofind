"""
model.py — EfficientNet-B0 classifier head for spectrogram classification.

We load ImageNet-pretrained weights via `timm`, replace the final classifier
with a new linear head for 50 classes, and optionally freeze the backbone
for the first few epochs (feature extraction → then fine-tune).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from spectofind import config as cfg


class SpectroNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for Mel-spectrogram classification.

    Architecture:
        EfficientNet-B0 backbone (1280-d features)
        → Dropout(0.3)
        → Linear(1280, num_classes)

    The backbone is initially partially frozen; call `unfreeze()` to enable
    full fine-tuning after the head has warmed up.
    """

    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        pretrained: bool = cfg.PRETRAINED,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,        # strip original head → returns features
            drop_rate=drop_rate,
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes),
        )

        # Freeze all backbone layers except the last block + stem — lets the
        # new head warm up without destroying pretrained features.
        self._freeze_backbone()

    # ── Freeze / Unfreeze ─────────────────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Freeze early backbone layers (keep last 2 blocks trainable)."""
        blocks = list(self.backbone.blocks.children()) if hasattr(self.backbone, "blocks") else []
        freeze_up_to = max(0, len(blocks) - 2)
        frozen = 0
        for i, block in enumerate(blocks):
            if i < freeze_up_to:
                for p in block.parameters():
                    p.requires_grad = False
                    frozen += 1
        # Always keep the head trainable
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) → (B, C)
        features = self.backbone(x)      # (B, feature_dim)
        return self.head(features)       # (B, num_classes)


# ── Convenience ──────────────────────────────────────────────────────────────

def build_model(device: torch.device | str = "cpu") -> SpectroNet:
    """Build and return model on the specified device."""
    model = SpectroNet()
    model = model.to(device)
    # Use channels_last memory format for Ampere GPUs (RTX 40xx) — faster convolutions
    if str(device) != "cpu":
        model = model.to(memory_format=torch.channels_last)
    return model


def load_checkpoint(path: str | None = None, device: torch.device | str = "cpu") -> SpectroNet:
    """Load a saved checkpoint into a SpectroNet model."""
    from pathlib import Path

    ckpt_path = Path(path) if path else cfg.BEST_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpectroNet(num_classes=checkpoint.get("num_classes", cfg.NUM_CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    if str(device) != "cpu":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
