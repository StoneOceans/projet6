from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Variables utilisées par le modèle de scoring.",
        min_length=1,
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    decision: str
    inference_time_ms: float
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str