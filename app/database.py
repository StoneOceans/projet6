import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "production_data"
DATABASE_PATH = DATABASE_DIR / "predictions.db"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialise_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction INTEGER,
                probability REAL,
                threshold REAL,
                decision TEXT,
                inference_time_ms REAL,
                response_time_ms REAL,
                status_code INTEGER NOT NULL,
                error_message TEXT,
                model_version TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction(
    *,
    request_id: str,
    input_data: dict[str, Any],
    prediction: int | None,
    probability: float | None,
    threshold: float,
    decision: str | None,
    inference_time_ms: float | None,
    response_time_ms: float,
    status_code: int,
    error_message: str | None,
    model_version: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                request_id,
                timestamp,
                input_data,
                prediction,
                probability,
                threshold,
                decision,
                inference_time_ms,
                response_time_ms,
                status_code,
                error_message,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data, ensure_ascii=False, default=str),
                prediction,
                probability,
                threshold,
                decision,
                inference_time_ms,
                response_time_ms,
                status_code,
                error_message,
                model_version,
            ),
        )
        connection.commit()


def read_recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
