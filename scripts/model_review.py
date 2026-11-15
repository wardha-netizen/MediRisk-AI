
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
METRICS_FILE = RESULTS_DIR / "evaluation_results.csv"

REPORT_DIR = RESULTS_DIR / "model_review"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, possible_names):
    for column in df.columns:
        normalised = (
            column.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        if normalised in possible_names:
            return column
    return None


def review_models():
    print("=" * 60)
    print("MEDIRISK AI - MODEL PERFORMANCE REVIEW")
    print("=" * 60)

    if not METRICS_FILE.exists():
        print("Evaluation file not found.")
        print("Expected:", METRICS_FILE)
        return

    df = pd.read_csv(METRICS_FILE)

    if df.empty:
        print("The evaluation file is empty.")
        return

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nCurrent model results:")
    print(df.to_string(index=False))

    disease_col = find_column(
        df, {"disease", "model", "dataset", "name"}
    )

    accuracy_col = find_column(
        df, {"accuracy", "accuracy_score"}
    )

    precision_col = find_column(
        df, {"precision", "precision_score"}
    )

    recall_col = find_column(
        df, {"recall", "recall_score"}
    )

    f1_col = find_column(
        df, {"f1", "f1_score", "f1score"}
    )

    notes = []

    for index, row in df.iterrows():
        disease = (
            str(row[disease_col])
            if disease_col else f"Model {index + 1}"
        )

        observations = []

        for label, column in [
            ("accuracy", accuracy_col),
            ("precision", precision_col),
            ("recall", recall_col),
            ("F1", f1_col),
        ]:
            if column is None:
                continue

            value = pd.to_numeric(
                row[column], errors="coerce"
            )

            if pd.isna(value):
                observations.append(
                    f"{label}: missing or invalid"
                )
                continue

            # Supports scores stored as 0-1 or 0-100.
            score = (
                value * 100
                if 0 <= value <= 1
                else value
            )

            if not 0 <= score <= 100:
                observations.append(
                    f"{label}: outside expected range"
                )
            elif score < 50:
                observations.append(
                    f"Low {label}: {score:.2f}%"
                )

        if "kidney" in disease.lower():
            observations.append(
                "Check duplicates, leakage and "
                "training/test separation."
            )

        if "stroke" in disease.lower():
            observations.append(
                "Investigate positive-class precision, "
                "recall and class imbalance."
            )

        if "liver" in disease.lower():
            observations.append(
                "Review class balance and "
                "Decision Tree generalisation."
            )

        if not observations:
            observations.append(
                "No automatic warning; "
                "manual validation still required."
            )

        notes.append({
            "Disease": disease,
            "Observations": " | ".join(observations)
        })

    review_df = pd.DataFrame(notes)

    output_path = REPORT_DIR / "review_summary.csv"
    review_df.to_csv(output_path, index=False)

    print("\nMODEL REVIEW FINDINGS")
    print(review_df.to_string(index=False))

    print("\nReport saved to:", output_path)
    print(
        "\nNote: These are screening observations, "
        "not proof of data leakage or model failure."
    )


if __name__ == "__main__":
    review_models()
