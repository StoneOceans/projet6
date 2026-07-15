import json
import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "production_data" / "predictions.db"
OUTPUT_PATH = BASE_DIR / "production_data" / "predictions_export.csv"


def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Base SQLite introuvable : {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        production_data = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id",
            connection,
        )

    if production_data.empty:
        print("La base ne contient encore aucune prédiction.")
        return

    if "input_data" in production_data.columns:
        parsed_inputs = production_data["input_data"].apply(json.loads)
        input_features = pd.json_normalize(parsed_inputs)

        production_data = pd.concat(
            [
                production_data.drop(columns=["input_data"]),
                input_features,
            ],
            axis=1,
        )

    production_data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Export créé : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(production_data)}")


if __name__ == "__main__":
    main()