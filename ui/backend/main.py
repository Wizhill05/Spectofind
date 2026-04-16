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
    """Load both models at startup, release on shutdown."""
    # ── Custom EfficientNet model ─────────────────────────────────────────
    try:
        from ui.backend.inference_engine import InferenceEngine
        print("[Spectofind] Loading custom model from", cfg.BEST_CKPT)
        app.state.custom_engine = InferenceEngine(cfg.BEST_CKPT)
        print("[Spectofind] Custom model loaded")
    except FileNotFoundError:
        print("[Spectofind] Custom model checkpoint not found — skipping")
        app.state.custom_engine = None

    # ── BEATs model ──────────────────────────────────────────────────────
    try:
        from ui.backend.beats_engine import BeatsEngine
        print("[Spectofind] Loading BEATs model ...")
        app.state.beats_engine = BeatsEngine()
        print("[Spectofind] BEATs model loaded")
    except Exception as e:
        print(f"[Spectofind] BEATs model failed to load: {e}")
        app.state.beats_engine = None

    # Default active model: BEATs if available, else custom
    if app.state.beats_engine is not None:
        app.state.active_model = "beats"
        app.state.engine = app.state.beats_engine
    elif app.state.custom_engine is not None:
        app.state.active_model = "custom"
        app.state.engine = app.state.custom_engine
    else:
        app.state.active_model = "none"
        app.state.engine = None

    print(f"[Spectofind] Active model: {app.state.active_model}")
    yield
    # Cleanup (nothing to do)


app = FastAPI(
    title="Spectofind API",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow development access from any network device
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {
        "status": "ok",
        "active_model": getattr(app.state, "active_model", "none"),
        "beats_loaded": getattr(app.state, "beats_engine", None) is not None,
        "custom_loaded": getattr(app.state, "custom_engine", None) is not None,
    }
