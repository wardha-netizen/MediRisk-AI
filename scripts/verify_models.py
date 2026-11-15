
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data_loader import load_all_datasets
from src.preprocessing import (
    prepare_disease_dataset,
    KIDNEY_NUMERICAL,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULT_DIR / "verification"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_array(data):
    if issparse(data):
        return data.toarray()
    return np.asarray(data)


def compare_score(name, actual, expected, tolerance=0.0001):
    if np.isclose(actual, expected, atol=tolerance):
        print(f"  PASS: {name} matches saved result")
        return True

    print(
        f"  FAIL: {name} differs "
        f"(calculated={actual:.4f}, saved={expected:.4f})"
    )
    return False


def verify_disease(name, df, saved_metrics):
    print("\n" + "=" * 60)
    print(f"VERIFYING: {name.upper()}")
    print("=" * 60)

    model_path = MODEL_DIR / f"{name}_model.joblib"

    if not model_path.exists():
        print("FAIL: Trained model not found")
        return {"Disease": name, "Status": "FAIL: Missing model"}

    try:
        saved = joblib.load(model_path)

        model = saved["model"]
        saved_preprocessor = saved["preprocessor"]
        feature_columns = list(saved["feature_columns"])

        (
            X_train,
            X_test,
            y_train,
            y_test,
            new_preprocessor,
        ) = prepare_disease_dataset(name, df)

        raw_test = df.loc[
            y_test.index, feature_columns
        ].copy()

        if name == "kidney":
            for column in KIDNEY_NUMERICAL:
                if column in raw_test.columns:
                    raw_test[column] = pd.to_numeric(
                        raw_test[column],
                        errors="coerce"
                    )

        saved_X_test = saved_preprocessor.transform(
            raw_test
        )

        original_array = dense_array(X_test)
        saved_array = dense_array(saved_X_test)

        if original_array.shape != saved_array.shape:
            print("FAIL: Preprocessing dimensions differ")
            return {
                "Disease": name,
                "Status": "FAIL: Preprocessing mismatch"
            }

        preprocessing_matches = np.allclose(
            original_array,
            saved_array,
            rtol=1e-6,
            atol=1e-8
        )

        print(
            "Preprocessing consistent:",
            preprocessing_matches
        )

        if not preprocessing_matches:
            print(
                "WARNING: Saved and reconstructed "
                "preprocessing outputs differ."
            )
            return {
                "Disease": name,
                "Status": "FAIL: Preprocessing mismatch"
            }

        predictions = model.predict(saved_X_test)

        actual = np.asarray(y_test, dtype=int)
        predicted = np.asarray(predictions, dtype=int)

        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(
            actual, predicted, zero_division=0
        )
        recall = recall_score(
            actual, predicted, zero_division=0
        )
        f1 = f1_score(
            actual, predicted, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        print("\nConfusion matrix:")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        checks = []

        matching_rows = saved_metrics[
            saved_metrics["Disease"]
            .astype(str)
            .str.lower() == name.lower()
        ]

        if len(matching_rows) == 1:
            row = matching_rows.iloc[0]

            metric_map = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

            for column, calculated in metric_map.items():
                if column in row.index:
                    expected = float(row[column])
                    checks.append(
                        compare_score(
                            column, calculated, expected
                        )
                    )
        else:
            print("WARNING: Saved metrics row unavailable")
            checks.append(False)

        predictions_file = (
            RESULT_DIR / "predictions"
            / f"{name}_predictions.csv"
        )

        if predictions_file.exists():
            exported = pd.read_csv(predictions_file)

            expected_labels = exported[
                ["Actual", "Predicted"]
            ].to_numpy(dtype=int)

            current_labels = np.column_stack(
                [actual, predicted]
            )

            labels_match = (
                expected_labels.shape == current_labels.shape
                and np.array_equal(
                    expected_labels, current_labels
                )
            )

            print(
                "Exported test labels match:",
                labels_match
            )
            checks.append(labels_match)
        else:
            print("NOTE: No exported predictions found")

        status = (
            "PASS" if checks and all(checks)
            else "REVIEW REQUIRED"
        )

        return {
            "Disease": name,
            "Status": status,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Positives": int(fp),
            "False Negatives": int(fn),
        }

    except Exception as error:
        print(f"FAIL: {error}")
        return {
            "Disease": name,
            "Status": f"FAIL: {error}"
        }


def main():
    print("MEDIRISK AI - MODEL RELIABILITY VERIFICATION")

    metrics_path = RESULT_DIR / "evaluation_results.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Run model evaluation before verification."
        )

    saved_metrics = pd.read_csv(metrics_path)
    datasets = load_all_datasets()

    reports = []

    for name, df in datasets.items():
        report = verify_disease(
            name, df, saved_metrics
        )
        reports.append(report)

    summary = pd.DataFrame(reports)

    output_path = (
        OUTPUT_DIR / "model_verification_summary.csv"
    )

    summary.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION REPORT")
    print("=" * 60)

    print(summary.to_string(index=False))
    print(f"\nSaved report: {output_path}")

    if not summary["Status"].eq("PASS").all():
        print("\nSome models need further investigation.")
        raise SystemExit(1)

    print("\nAll available model checks passed.")


if __name__ == "__main__":
    main()
