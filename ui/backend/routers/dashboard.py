"""
dashboard.py — REST endpoints for the training dashboard tab.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import FileResponse

from spectofind import config as cfg

router = APIRouter()

# Cache evaluation results in memory (computed once, ~1-2 s on GPU)
_evaluation_cache: dict | None = None


@router.get("/model-info")
async def model_info():
    """Return model metadata and training history from checkpoints."""
    from ui.backend.inference_engine import InferenceEngine

    info = await asyncio.to_thread(InferenceEngine.get_checkpoint_info)
    return info


@router.get("/evaluation")
async def evaluation():
    """Run evaluation on validation fold and return per-class accuracy."""
    global _evaluation_cache
    if _evaluation_cache is not None:
        return _evaluation_cache

    result = await asyncio.to_thread(_compute_evaluation)
    _evaluation_cache = result
    return result


@router.get("/confusion-matrix")
async def confusion_matrix_image():
    """Serve the confusion matrix PNG."""
    path = cfg.RESULTS_DIR / "confusion_matrix.png"
    if not path.exists():
        return {"error": "Run evaluation first: uv run python -m spectofind.evaluate"}
    return FileResponse(str(path), media_type="image/png")


@router.get("/training-history-image")
async def training_history_image():
    """Serve the training history plot PNG."""
    path = cfg.RESULTS_DIR / "training_history.png"
    if not path.exists():
        return {"error": "No training history image found."}
    return FileResponse(str(path), media_type="image/png")


@router.get("/audio-sample/{class_idx}")
async def audio_sample(class_idx: int):
    """Serve a sample .wav file for the given class index."""
    from spectofind.dataset import load_metadata
    meta = load_metadata()
    for row in meta:
        if row["target"] == class_idx:
            audio_path = cfg.AUDIO_DIR / row["filename"]
            if audio_path.exists():
                return FileResponse(str(audio_path), media_type="audio/wav")
    return {"error": "Audio sample not found"}


# ── Internal ──────────────────────────────────────────────────────────────────

def _compute_evaluation() -> dict:
    """Compute per-class accuracy on the validation fold."""
    from torch.utils.data import DataLoader
    from spectofind.dataset import SpectrogramDataset, get_class_names
    from spectofind.model import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(device=device)
    model.eval()

    val_ds = SpectrogramDataset(folds=[cfg.VAL_FOLD], augment=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = get_class_names()

    per_class = []
    for i in range(cfg.NUM_CLASSES):
        mask = all_labels == i
        total = int(mask.sum())
        correct = int((all_preds[mask] == i).sum()) if total > 0 else 0
        per_class.append({
            "idx": i,
            "name": class_names[i],
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        })

    return {
        "overall_accuracy": round(float((all_preds == all_labels).mean()), 4),
        "per_class": per_class,
    }
