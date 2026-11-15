
import numpy as np
import pandas as pd

from src.preprocessing import prepare_disease_dataset


def create_sample_heart_data():
    records = []

    for i in range(20):
        records.append({
            "age": 35 + i,
            "sex": i % 2,
            "cp": (i % 4) + 1,
            "trestbps": 110 + i,
            "chol": 180 + i,
            "fbs": i % 2,
            "restecg": i % 3,
            "thalach": 130 + i,
            "exang": i % 2,
            "oldpeak": float(i % 4),
            "slope": (i % 3) + 1,
            "ca": i % 4,
            "thal": 3,
            "num": 0 if i < 10 else 1,
        })

    df = pd.DataFrame(records)

    # Add a missing target to check its handling
    df.loc[0, "num"] = np.nan

    return df


def test_missing_target_is_removed():
    df = create_sample_heart_data()

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_disease_dataset("heart", df)
    )

    total_records = len(y_train) + len(y_test)

    assert total_records == len(df) - 1

    assert not y_train.isna().any()
    assert not y_test.isna().any()


def test_target_classes_are_binary():
    df = create_sample_heart_data()

    _, _, y_train, y_test, _ = (
        prepare_disease_dataset("heart", df)
    )

    labels = set(y_train) | set(y_test)

    assert labels == {0, 1}


def test_preprocessing_dimensions_match():
    df = create_sample_heart_data()

    X_train, X_test, _, _, preprocessor = (
        prepare_disease_dataset("heart", df)
    )

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
