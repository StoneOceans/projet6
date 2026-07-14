import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_lightgbm.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.json"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de seuil introuvable : {THRESHOLD_PATH}"
        )

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in (
            "best_threshold",
            "threshold",
            "seuil",
        ):
            if key in data:
                return float(data[key])

    if isinstance(data, (int, float)):
        return float(data)

    raise ValueError(
        "Le fichier best_threshold.json ne contient pas "
        "un seuil exploitable."
    )


def load_feature_columns() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de colonnes introuvable : {FEATURES_PATH}"
        )

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [str(column) for column in data]

    if isinstance(data, dict):
        for key in (
            "feature_columns",
            "features",
            "columns",
        ):
            if key in data and isinstance(data[key], list):
                return [str(column) for column in data[key]]

    raise ValueError(
        "Le fichier feature_columns.json ne contient pas "
        "une liste de colonnes exploitable."
    )


model = load_model()
best_threshold = load_threshold()
feature_columns = load_feature_columns()
