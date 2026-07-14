import pytest

from app.model_loader import best_threshold, feature_columns, model
from app.prediction import prepare_dataframe, validate_features


def test_model_is_loaded():
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_threshold_is_valid():
    assert isinstance(best_threshold, float)
    assert 0 < best_threshold < 1
    assert best_threshold == pytest.approx(0.48)


def test_feature_columns_are_loaded():
    assert isinstance(feature_columns, list)
    assert len(feature_columns) == 300
    assert all(
        isinstance(column, str)
        for column in feature_columns
    )


def test_empty_features_raise_error():
    with pytest.raises(
        ValueError,
        match="Aucune variable",
    ):
        validate_features({})


def test_missing_features_raise_error():
    with pytest.raises(
        ValueError,
        match="Variables obligatoires manquantes",
    ):
        validate_features(
            {
                feature_columns[0]: 0.5,
            }
        )


def test_prepare_dataframe_keeps_column_order():
    features = {
        column: 0
        for column in feature_columns
    }

    dataframe = prepare_dataframe(features)

    assert dataframe.shape == (1, 300)
    assert dataframe.columns.tolist() == feature_columns
