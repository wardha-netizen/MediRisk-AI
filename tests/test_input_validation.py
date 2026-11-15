
import numpy as np
import pandas as pd
import pytest

from src.prediction import validate_input_data


FEATURES = ["age", "bmi", "gender"]


def test_valid_input_is_accepted():
    input_data = {
        "age": 45,
        "bmi": 25.5,
        "gender": "Male"
    }

    result = validate_input_data(input_data, FEATURES)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == FEATURES
    assert result.iloc[0]["age"] == 45


def test_feature_order_is_preserved():
    input_data = {
        "gender": "Female",
        "bmi": 30.0,
        "age": 50
    }

    result = validate_input_data(input_data, FEATURES)

    assert list(result.columns) == FEATURES


def test_missing_feature_is_rejected():
    input_data = {
        "age": 45,
        "bmi": 25.5
    }

    with pytest.raises(ValueError, match="Missing input features"):
        validate_input_data(input_data, FEATURES)


@pytest.mark.parametrize("invalid_value", [None, np.nan, pd.NA])
def test_missing_values_are_rejected(invalid_value):
    input_data = {
        "age": 45,
        "bmi": invalid_value,
        "gender": "Male"
    }

    with pytest.raises(ValueError, match="Missing value"):
        validate_input_data(input_data, FEATURES)


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_infinite_values_are_rejected(invalid_value):
    input_data = {
        "age": 45,
        "bmi": invalid_value,
        "gender": "Male"
    }

    with pytest.raises(ValueError, match="infinite values"):
        validate_input_data(input_data, FEATURES)


def test_non_dictionary_input_is_rejected():
    with pytest.raises(TypeError, match="dictionary"):
        validate_input_data([45, 25.5, "Male"], FEATURES)


def test_multiple_values_for_one_feature_are_rejected():
    input_data = {
        "age": 45,
        "bmi": [25.5, 26.0],
        "gender": "Male"
    }

    with pytest.raises(ValueError, match="single value"):
        validate_input_data(input_data, FEATURES)
