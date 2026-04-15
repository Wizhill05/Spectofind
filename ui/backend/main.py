"""
Spectofind — FastAPI backend.

Serves the trained model for inference (REST + WebSocket) and
provides dashboard data (training history, per-class accuracy).

Start:
    uv run uvicorn ui.backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from spectofind import config as cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release on shutdown."""
    from ui.backend.inference_engine import InferenceEngine

    print("[Spectofind] Loading model from", cfg.BEST_CKPT)
    app.state.engine = InferenceEngine(cfg.BEST_CKPT)
    print("[Spectofind] Model loaded — ready to serve")
    yield
    # Cleanup (nothing to do)


app = FastAPI(
    title="Spectofind API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve results/ images (confusion_matrix.png, training_history.png)
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(cfg.RESULTS_DIR)), name="results")

# Include routers
from ui.backend.routers.dashboard import router as dashboard_router   # noqa: E402
from ui.backend.routers.inference import router as inference_router   # noqa: E402

app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(inference_router, prefix="/api/infer", tags=["inference"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "engine")}
