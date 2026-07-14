from time import perf_counter
from typing import Any

import pandas as pd

from app.model_loader import (
    best_threshold,
    feature_columns,
    model,
)

MODEL_VERSION = "1.0.0"


def validate_features(features: dict[str, Any]) -> None:
    if not features:
        raise ValueError("Aucune variable n'a été transmise.")

    missing_features = [
        column
        for column in feature_columns
        if column not in features
    ]

    if missing_features:
        preview = ", ".join(missing_features[:10])

        raise ValueError(
            f"Variables obligatoires manquantes : {preview}"
        )


def prepare_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    ordered_data = {
        column: features.get(column)
        for column in feature_columns
    }

    return pd.DataFrame(
        [ordered_data],
        columns=feature_columns,
    )


def predict_client(features: dict[str, Any]) -> dict[str, Any]:
    validate_features(features)

    client_dataframe = prepare_dataframe(features)

    started_at = perf_counter()

    probability = float(
        model.predict_proba(client_dataframe)[:, 1][0]
    )

    inference_time_ms = (
        perf_counter() - started_at
    ) * 1000

    prediction = int(
        probability >= best_threshold
    )

    decision = (
        "Crédit refusé"
        if prediction == 1
        else "Crédit accordé"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 6),
        "threshold": round(best_threshold, 6),
        "decision": decision,
        "inference_time_ms": round(inference_time_ms, 3),
        "model_version": MODEL_VERSION,
    }
