from time import perf_counter
from uuid import uuid4


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.database import (
    initialise_database,
    read_recent_predictions,
    save_prediction,
)
from app.model_loader import model
from app.prediction import MODEL_VERSION, predict_client
from app.model_loader import best_threshold, feature_columns, model
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="API de scoring Prêt à Dépenser",
    description=(
        "API permettant de calculer une probabilité de défaut "
        "et de retourner une décision de crédit."
    ),
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    initialise_database()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "API de scoring opérationnelle",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
    )


@app.get("/example")
def prediction_example() -> dict:
    return {
        "features": {
            column: 0
            for column in feature_columns
        }
    }
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    request_id = str(uuid4())
    request_started_at = perf_counter()

    try:
        result = predict_client(payload.features)

        response_time_ms = (
            perf_counter() - request_started_at
        ) * 1000

        save_prediction(
            request_id=request_id,
            input_data=payload.features,
            prediction=result["prediction"],
            probability=result["probability"],
            threshold=result["threshold"],
            decision=result["decision"],
            inference_time_ms=result["inference_time_ms"],
            response_time_ms=round(response_time_ms, 3),
            status_code=200,
            error_message=None,
            model_version=result["model_version"],
        )

        return PredictionResponse(
            **result,
            request_id=request_id,
        )

    except (ValueError, TypeError) as error:
        response_time_ms = (
            perf_counter() - request_started_at
        ) * 1000

        save_prediction(
            request_id=request_id,
            input_data=payload.features,
            prediction=None,
            probability=None,
            threshold=best_threshold,
            decision=None,
            inference_time_ms=None,
            response_time_ms=round(response_time_ms, 3),
            status_code=422,
            error_message=str(error),
            model_version=MODEL_VERSION,
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        response_time_ms = (
            perf_counter() - request_started_at
        ) * 1000

        save_prediction(
            request_id=request_id,
            input_data=payload.features,
            prediction=None,
            probability=None,
            threshold=best_threshold,
            decision=None,
            inference_time_ms=None,
            response_time_ms=round(response_time_ms, 3),
            status_code=500,
            error_message=str(error),
            model_version=MODEL_VERSION,
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction.",
        ) from error

@app.get("/monitoring/predictions")
def monitoring_predictions(
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    predictions = read_recent_predictions(limit=limit)

    readable_predictions = []

    for row in predictions:
        readable_predictions.append(
            {
                "id": row["id"],
                "request_id": row["request_id"],
                "timestamp": row["timestamp"],
                "prediction": row["prediction"],
                "probability": row["probability"],
                "threshold": row["threshold"],
                "decision": row["decision"],
                "inference_time_ms": row["inference_time_ms"],
                "response_time_ms": row["response_time_ms"],
                "status_code": row["status_code"],
                "error_message": row["error_message"],
                "model_version": row["model_version"],
            }
        )

    return {
        "count": len(readable_predictions),
        "predictions": readable_predictions,
    }