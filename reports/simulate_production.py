import json
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"

POSSIBLE_DATA_PATHS = [
    BASE_DIR / "data" / "processed" / "home_credit_test_processed.csv",
    BASE_DIR / "data" / "processed" / "home_credit_train_processed.csv",
]

API_URL = "http://127.0.0.1:8000/predict"
NUMBER_OF_CLIENTS = 50


def find_compatible_file(
    feature_columns: list[str],
) -> Path:
    for path in POSSIBLE_DATA_PATHS:
        if not path.exists():
            continue

        columns = pd.read_csv(
            path,
            nrows=0,
        ).columns.tolist()

        missing_columns = [
            column
            for column in feature_columns
            if column not in columns
        ]

        if not missing_columns:
            return path

    raise FileNotFoundError(
        "Aucun fichier contenant les 300 variables "
        "du modèle n'a été trouvé."
    )


def convert_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def main() -> None:
    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    data_path = find_compatible_file(
        feature_columns
    )

    print("Fichier utilisé :", data_path)

    data = pd.read_csv(
        data_path,
        usecols=feature_columns,
        nrows=NUMBER_OF_CLIENTS,
    )

    successful_requests = 0
    failed_requests = 0

    for index, row in data.iterrows():
        payload = {
            "features": {
                column: convert_value(row[column])
                for column in feature_columns
            }
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()

                successful_requests += 1

                print(
                    f"Client {index + 1}: "
                    f"probabilité={result['probability']} | "
                    f"décision={result['decision']}"
                )
            else:
                failed_requests += 1

                print(
                    f"Client {index + 1}: "
                    f"erreur HTTP {response.status_code} | "
                    f"{response.text[:200]}"
                )

        except requests.RequestException as error:
            failed_requests += 1

            print(
                f"Client {index + 1}: "
                f"API inaccessible | {error}"
            )

    print()
    print("Simulation terminée")
    print("Succès :", successful_requests)
    print("Échecs :", failed_requests)


if __name__ == "__main__":
    main()