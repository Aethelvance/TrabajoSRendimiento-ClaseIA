"""Esquemas request/response (diagrama 05)."""
from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    deporte: float = Field(ge=0.0, le=5.0, description="Horas de deporte 0-5")
    sueno: float = Field(ge=3.0, le=9.0, description="Horas de sueno 3-9")


class PredictOut(BaseModel):
    rendimiento: float


class HealthOut(BaseModel):
    status: str
    model_loaded: bool


class MetricsOut(BaseModel):
    mae: float
    rmse: float
    r2: float
    params: int
    arch: str
