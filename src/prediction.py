import math
import numbers
from collections.abc import Mapping

import joblib
import pandas as pd

from pathlib import Path


# Main project folder
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"


# Available disease models
DISEASES = {
    "diabetes": "Diabetes",
    "heart": "Heart Disease",
    "kidney": "Chronic Kidney Disease",
    "liver": "Liver Disease",
    "stroke": "Stroke"
}


def load_model(disease_name):
    """Load a trained model and its preprocessing pipeline."""

    if disease_name not in DISEASES:
        raise ValueError(f"Unknown disease: {disease_name}")

    model_path = MODEL_FOLDER / f"{disease_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model_data = joblib.load(model_path)

    required_keys = {"model", "preprocessor", "feature_columns"}

    if not required_keys.issubset(model_data):
        raise ValueError(
            f"Invalid model file: {model_path.name}"
        )

    return model_data


def validate_input_data(input_data, feature_columns):
    """Validate user inputs before sending them to the model."""

    if not isinstance(input_data, Mapping):
        raise TypeError("Input data must be provided as a dictionary.")

    missing_features = [
        column for column in feature_columns
        if column not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    for column in feature_columns:
        value = input_data[column]

        if not pd.api.types.is_scalar(value):
            raise ValueError(
                f"Invalid value for '{column}': expected a single value."
            )

        if pd.isna(value):
            raise ValueError(
                f"Missing value for '{column}'. Please provide a valid value."
            )

        if isinstance(value, numbers.Real) and not math.isfinite(value):
            raise ValueError(
                f"Invalid numeric value for '{column}': "
                "infinite values are not allowed."
            )

    return pd.DataFrame([
        {column: input_data[column] for column in feature_columns}
    ])


def predict_disease(disease_name, input_data):
    """Generate a prediction using the saved model and preprocessor."""

    model_data = load_model(disease_name)

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    feature_columns = model_data["feature_columns"]

        # Validate user inputs and convert them into a DataFrame
    input_df = validate_input_data(input_data, feature_columns)

    # Apply the original preprocessing pipeline
    processed_input = preprocessor.transform(input_df)

    # Generate prediction
    prediction = int(model.predict(processed_input)[0])

    # Get estimated probability of the positive class
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_input)[0]

        positive_indices = [
            index
            for index, label in enumerate(model.classes_)
            if label == 1
        ]

        if positive_indices:
            probability = float(
                probabilities[positive_indices[0]]
            )

    return {
        "disease": DISEASES[disease_name],
        "prediction": prediction,
        "probability": probability,
        "result": (
            "Disease Risk Detected"
            if prediction == 1
            else "No Disease Risk Detected"
        )
    }
