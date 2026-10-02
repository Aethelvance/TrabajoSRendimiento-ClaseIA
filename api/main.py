"""API inferencia MLP rendimiento: POST /predict, GET /health, GET /metrics."""
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import MLPRegressor
from .schemas import HealthOut, MetricsOut, PredictIn, PredictOut

ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL = ROOT / "models" / "best_model.pt"
SCALER_FILE = ROOT / "models" / "scaler.pt"

# Constantes de training.evaluate en test (90 filas). No se recalculan.
METRICS = {"mae": 4.33, "rmse": 5.22, "r2": 0.9884, "params": 641, "arch": "2-32-16-1"}

_state = {"model": None, "scaler": None, "loaded": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if BEST_MODEL.exists() and SCALER_FILE.exists():
        model = MLPRegressor()
        model.load_state_dict(torch.load(BEST_MODEL, map_location="cpu"))
        model.eval()
        _state["model"] = model
        _state["scaler"] = torch.load(SCALER_FILE, map_location="cpu")
        _state["loaded"] = True
    yield


app = FastAPI(title="MLP Rendimiento", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8082"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    if not _state["loaded"]:
        raise HTTPException(503, "modelo no cargado")
    s = _state["scaler"]
    with torch.no_grad():
        x = torch.tensor([[body.deporte, body.sueno]])
        xn = (x - s["x_mean"]) / s["x_std"]
        pred_norm = _state["model"](xn)
        y = pred_norm * s["y_std"] + s["y_mean"]
    return PredictOut(rendimiento=round(float(y.item()), 1))


@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(status="ok", model_loaded=_state["loaded"])


@app.get("/metrics", response_model=MetricsOut)
def metrics():
    return MetricsOut(**METRICS)
