
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_loader import load_all_datasets
from src.preprocessing import prepare_disease_dataset


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "results" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_result(actual, predicted):
    actual = int(actual)
    predicted = int(predicted)

    if actual == 1 and predicted == 1:
        return "True Positive"

    if actual == 0 and predicted == 0:
        return "True Negative"

    if actual == 0 and predicted == 1:
        return "False Positive"

    return "False Negative"


def export_predictions():
    datasets = load_all_datasets()

    summary = []
    exported = 0

    for disease, df in datasets.items():
        print(f"\nProcessing {disease}...")

        model_path = (
            MODELS_DIR / f"{disease}_model.joblib"
        )

        if not model_path.exists():
            print(f"Model missing: {model_path}")
            continue

        try:
            saved = joblib.load(model_path)

            model = saved["model"]

            (
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            ) = prepare_disease_dataset(
                disease,
                df
            )

            # This reproduces the existing evaluation
            # workflow. Refit models/preprocessors must
            # correspond to the saved training version.
            predictions = model.predict(X_test)

            probabilities = (
                model.predict_proba(X_test)
                if hasattr(model, "predict_proba")
                else None
            )

            classes = list(model.classes_)

            actual = pd.Series(
                y_test
            ).reset_index(drop=True).astype(int)

            predicted = pd.Series(
                predictions
            ).astype(int)

            result_types = [
                classify_result(a, p)
                for a, p in zip(actual, predicted)
            ]

            results = pd.DataFrame({
                "Test_Record": range(
                    1, len(actual) + 1
                ),
                "Actual": actual,
                "Predicted": predicted,
                "Result": result_types,
            })

            if probabilities is not None and 1 in classes:
                positive_index = classes.index(1)

                results["Positive_Probability"] = (
                    probabilities[:, positive_index]
                )

            output_path = (
                OUTPUT_DIR / f"{disease}_predictions.csv"
            )

            results.to_csv(
                output_path,
                index=False
            )

            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).ravel()

            summary.append({
                "Disease": disease.capitalize(),
                "Test Records": len(results),
                "True Positive": int(tp),
                "True Negative": int(tn),
                "False Positive": int(fp),
                "False Negative": int(fn),
            })

            exported += 1

            print(
                f"Saved {len(results)} records "
                f"to {output_path}"
            )

        except Exception as error:
            print(f"Failed for {disease}: {error}")

    if summary:
        summary_df = pd.DataFrame(summary)

        summary_df.to_csv(
            OUTPUT_DIR / "summary.csv",
            index=False
        )

        print("\nPREDICTION SUMMARY")
        print(summary_df.to_string(index=False))

    print(
        f"\nCompleted: {exported}/{len(datasets)} "
        "disease models."
    )

    if exported != len(datasets):
        raise RuntimeError(
            "Some prediction exports failed. "
            "Review the messages above."
        )


if __name__ == "__main__":
    export_predictions()
