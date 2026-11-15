
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data_loader import load_kidney
from src.preprocessing import KIDNEY_NUMERICAL


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "kidney_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_kidney():
    df = load_kidney().copy()

    for column in KIDNEY_NUMERICAL:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["class"] = (
        df["class"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"ckd": 1, "notckd": 0})
    )

    df = df.dropna(subset=["class"])

    return df


def main():
    print("=" * 60)
    print("MEDIRISK AI - KIDNEY RELIABILITY AUDIT")
    print("=" * 60)

    df = prepare_kidney()

    X = df.drop(columns=["class"])
    y = df["class"].astype(int)

    print("\nTotal records:", len(df))
    print("Duplicate rows:", df.duplicated().sum())

    print("\nClass distribution:")
    print(y.value_counts())

    # Check duplicates across train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_hashes = pd.util.hash_pandas_object(
        X_train.fillna("MISSING").astype(str),
        index=False
    )

    test_hashes = pd.util.hash_pandas_object(
        X_test.fillna("MISSING").astype(str),
        index=False
    )

    overlapping = set(train_hashes) & set(test_hashes)

    print(
        "\nExact duplicate feature rows shared "
        "between train/test:",
        len(overlapping)
    )

    numerical = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical
        )
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    results = []

    for split_name, actual, predicted in [
        ("Training", y_train, train_predictions),
        ("Testing", y_test, test_predictions)
    ]:
        results.append({
            "Split": split_name,
            "Accuracy": accuracy_score(
                actual, predicted
            ),
            "Precision": precision_score(
                actual, predicted,
                zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted,
                zero_division=0
            ),
            "F1 Score": f1_score(
                actual, predicted,
                zero_division=0
            )
        })

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT_DIR / "kidney_train_test_comparison.csv",
        index=False
    )

    print("\nTRAINING VS TESTING")
    print(report.to_string(index=False))

    train_accuracy = report.iloc[0]["Accuracy"]
    test_accuracy = report.iloc[1]["Accuracy"]

    print(
        "\nAccuracy gap:",
        round(train_accuracy - test_accuracy, 4)
    )

    if train_accuracy - test_accuracy > 0.10:
        print(
            "Warning: Large train/test accuracy gap. "
            "Investigate possible overfitting."
        )

    if len(overlapping) > 0:
        print(
            "Warning: Identical feature rows appear "
            "in both training and testing sets."
        )

    print("\nAudit complete.")
    print("Original Kidney model was not overwritten.")


if __name__ == "__main__":
    main()
