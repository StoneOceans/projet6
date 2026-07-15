import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    feature_columns = json.load(file)
    
possible_files = [
    BASE_DIR / "data" / "processed" / "home_credit_train_processed.csv",
]

data_path = next(
    (
        path
        for path in possible_files
        if path.exists()
    ),
    None,
)

if data_path is None:
    raise FileNotFoundError(
        "Aucun fichier X_test.csv, X_test_imputed.csv "
        "ou reference_data.csv trouvé dans data/."
    )

data = pd.read_csv(data_path)

missing_columns = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"{len(missing_columns)} colonnes du modèle "
        "sont absentes du fichier de données."
    )

client = data.loc[0, feature_columns]

payload = {
    "features": {
        column: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for column, value in client.items()
    }
}

output_path = BASE_DIR / "production_data" / "example_payload.json"

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        payload,
        file,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.item()
        if hasattr(value, "item")
        else str(value),
    )

print(f"Payload créé : {output_path}")
