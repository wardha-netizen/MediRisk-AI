
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

from src.data_loader import load_stroke


BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = BASE_DIR / "results" / "tuning"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("MEDIRISK AI - STROKE MODEL TUNING")
    print("=" * 60)

    # 1. Load dataset
    df = load_stroke().copy()

    df = df.drop(columns=["id"], errors="ignore")

    # 2. Prepare target
    df["stroke"] = pd.to_numeric(
        df["stroke"], errors="coerce"
    )

    df = df.dropna(subset=["stroke"])

    if not df["stroke"].isin([0, 1]).all():
        raise ValueError(
            "Stroke target contains unexpected labels."
        )

    X = df.drop(columns=["stroke"])
    y = df["stroke"].astype(int)

    # 3. Identify features
    numerical = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # 4. Preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

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

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 6. Complete machine learning pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ])

    # 7. Hyperparameter search
    parameters = {
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__min_samples_leaf": [5, 10, 20],
        "classifier__class_weight": [
            None,
            "balanced"
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

    print("\nTraining different configurations...")

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "\nBest cross-validation F1:",
        round(search.best_score_, 4)
    )

    # 8. Evaluate on held-out test set
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    results = {
        "Disease": "Stroke",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    print("\nTUNED MODEL RESULTS")

    for metric, value in results.items():
        if metric != "Disease":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test, predictions,
            labels=[0, 1]
        )
    )

    # 9. Save experimental results
    pd.DataFrame([results]).to_csv(
        RESULT_DIR / "stroke_tuned_metrics.csv",
        index=False
    )

    cv_results = pd.DataFrame(search.cv_results_)

    cv_results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]
    ].to_csv(
        RESULT_DIR / "stroke_cv_results.csv",
        index=False
    )

    # Save separately from the production model
    joblib.dump(
        best_model,
        RESULT_DIR / "stroke_tuned_pipeline.joblib"
    )

    print("\nExperimental model and results saved.")
    print("Original production model was not overwritten.")


if __name__ == "__main__":
    main()
