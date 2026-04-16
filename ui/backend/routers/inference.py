"""
inference.py — REST file upload + WebSocket live streaming inference + model switching.

WebSocket protocol:
  1. Client connects to /api/infer/stream
  2. Client sends JSON: {"sampleRate": 44100}
  3. Client streams binary Float32 PCM audio buffers (~3-5 sec each)
  4. Server responds with JSON: {"predictions": [...], "timestamp": ...}
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()


class ModelSwitch(BaseModel):
    model: str  # "beats" or "custom"


@router.post("/switch-model")
async def switch_model(request: Request, body: ModelSwitch):
    """Switch the active inference model."""
    model_name = body.model.lower()

    if model_name == "beats":
        if request.app.state.beats_engine is None:
            return {"error": "BEATs model not loaded", "active_model": request.app.state.active_model}
        request.app.state.engine = request.app.state.beats_engine
        request.app.state.active_model = "beats"
    elif model_name == "custom":
        if request.app.state.custom_engine is None:
            return {"error": "Custom model not loaded", "active_model": request.app.state.active_model}
        request.app.state.engine = request.app.state.custom_engine
        request.app.state.active_model = "custom"
    else:
        return {"error": f"Unknown model: {model_name}", "active_model": request.app.state.active_model}

    print(f"[Spectofind] Switched active model to: {request.app.state.active_model}")
    return {"active_model": request.app.state.active_model}


@router.get("/active-model")
async def get_active_model(request: Request):
    """Get the currently active inference model."""
    return {
        "active_model": request.app.state.active_model,
        "beats_available": request.app.state.beats_engine is not None,
        "custom_available": request.app.state.custom_engine is not None,
    }


@router.post("/file")
async def infer_file(request: Request, file: UploadFile = File(...)):
    """Upload a WAV file and get classification predictions."""
    engine = request.app.state.engine

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        predictions = await asyncio.to_thread(
            engine.predict_from_file, tmp_path, 5
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"predictions": predictions}


@router.websocket("/stream")
async def stream_inference(ws: WebSocket):
    """
    Live audio streaming inference.

    The client sends:
      1. A JSON config message: {"sampleRate": 44100}
      2. Binary Float32 PCM audio buffers

    The server responds with prediction JSON after each audio buffer.
    """
    await ws.accept()
    sample_rate = 44100  # default, client will override

    try:
        # First message: config
        config_raw = await ws.receive_text()
        config = json.loads(config_raw)
        sample_rate = int(config.get("sampleRate", 44100))
        print(f"[WS] Client connected — sample rate: {sample_rate} Hz")

        # Subsequent messages: binary PCM audio
        while True:
            data = await ws.receive_bytes()

            # Convert raw bytes to float32 numpy array
            n_samples = len(data) // 4  # float32 = 4 bytes
            if n_samples < sample_rate:  # less than 1 second — skip
                continue

            audio = np.frombuffer(data, dtype=np.float32).copy()

            # Get the current active engine (may have been switched mid-stream)
            engine = ws.app.state.engine

            # Run inference in thread pool (blocking CUDA call)
            predictions = await asyncio.to_thread(
                engine.predict_from_array, audio, sample_rate, 5
            )

            await ws.send_json({
                "predictions": predictions,
                "timestamp": time.time(),
                "audio_duration": round(n_samples / sample_rate, 2),
                "model": ws.app.state.active_model,
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await ws.close(code=1011, reason=str(e))
        except Exception:
            pass
