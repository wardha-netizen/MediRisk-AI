
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_loader import load_liver


BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = BASE_DIR / "results" / "tuning"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("MEDIRISK AI - LIVER MODEL TUNING")
    print("=" * 60)

    # Load the Liver Disease dataset
    df = load_liver().copy()

    target = "Dataset"

    # Convert labels to numerical format
    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Remove unknown outcomes
    df = df.dropna(subset=[target])

    # Validate target labels before conversion
    if not df[target].isin([1, 2]).all():
        raise ValueError(
            "Unexpected Liver Disease target labels."
        )

    # Convert to binary classes
    df[target] = df[target].map({
        1: 1,
        2: 0
    }).astype(int)

    X = df.drop(columns=[target])
    y = df[target]

    # Identify numerical and categorical features
    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Complete machine learning pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ])

    # Hyperparameter search space
    parameters = {
        "classifier__max_depth": [
            3, 5, 7, 10, None
        ],
        "classifier__min_samples_leaf": [
            2, 5, 10, 20
        ],
        "classifier__class_weight": [
            None, "balanced"
        ],
        "classifier__criterion": [
            "gini", "entropy"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    print("\nTesting Liver model configurations...")

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "\nBest cross-validation F1:",
        round(search.best_score_, 4)
    )

    # Final evaluation on held-out data
    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    results = {
        "Disease": "Liver",
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            zero_division=0
        )
    }

    print("\nTUNED LIVER MODEL RESULTS")

    for metric, value in results.items():
        if metric != "Disease":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    # Save experiment results
    pd.DataFrame([results]).to_csv(
        RESULT_DIR / "liver_tuned_metrics.csv",
        index=False
    )

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]
    ].to_csv(
        RESULT_DIR / "liver_cv_results.csv",
        index=False
    )

    # Store the experimental model separately
    joblib.dump(
        best_model,
        RESULT_DIR / "liver_tuned_pipeline.joblib"
    )

    print("\nLiver tuning experiment completed.")
    print("The original Liver model has not been changed.")


if __name__ == "__main__":
    main()
