from uuid import uuid4

from app.database import (
    initialise_database,
    read_recent_predictions,
    save_prediction,
)


def test_database_initialisation():
    initialise_database()

    rows = read_recent_predictions(limit=10)

    assert isinstance(rows, list)


def test_save_successful_prediction():
    request_id = f"pytest-success-{uuid4()}"

    save_prediction(
        request_id=request_id,
        input_data={
            "EXT_SOURCE_3": 0.75,
            "EXT_SOURCE_2": 0.65,
        },
        prediction=0,
        probability=0.20,
        threshold=0.48,
        decision="Crédit accordé",
        inference_time_ms=3.5,
        response_time_ms=5.0,
        status_code=200,
        error_message=None,
        model_version="1.0.0",
    )

    rows = read_recent_predictions(limit=1000)

    matching_rows = [
        row
        for row in rows
        if row["request_id"] == request_id
    ]

    assert len(matching_rows) == 1

    stored_prediction = matching_rows[0]

    assert stored_prediction["prediction"] == 0
    assert stored_prediction["probability"] == 0.20
    assert stored_prediction["threshold"] == 0.48
    assert stored_prediction["status_code"] == 200
    assert stored_prediction["error_message"] is None


def test_save_failed_prediction():
    request_id = f"pytest-error-{uuid4()}"

    save_prediction(
        request_id=request_id,
        input_data={},
        prediction=None,
        probability=None,
        threshold=0.48,
        decision=None,
        inference_time_ms=None,
        response_time_ms=1.0,
        status_code=422,
        error_message="Variables obligatoires manquantes",
        model_version="1.0.0",
    )

    rows = read_recent_predictions(limit=1000)

    matching_rows = [
        row
        for row in rows
        if row["request_id"] == request_id
    ]

    assert len(matching_rows) == 1

    stored_prediction = matching_rows[0]

    assert stored_prediction["prediction"] is None
    assert stored_prediction["status_code"] == 422
    assert stored_prediction["error_message"] is not None
