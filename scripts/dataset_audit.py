print("MEDIRISK DATASET AUDIT STARTED")

import pandas as pd
from pathlib import Path
from src.data_loader import load_all_datasets

RESULTS_DIR = Path("results/data_audit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = {
    "diabetes": "Diabetes_binary",
    "heart": "num",
    "kidney": "class",
    "liver": "Dataset",
    "stroke": "stroke",
}

MISSING_MARKERS = {"", "?", "NA", "N/A", "null", "None", "nan"}


def clean_missing_markers(df):
    data = df.copy()

    for column in data.select_dtypes(include=["object", "string"]):
        cleaned = data[column].astype("string").str.strip()
        data[column] = cleaned.replace(
            list(MISSING_MARKERS),
            pd.NA
        )

    return data


def audit_dataset(name, df):
    data = clean_missing_markers(df)

    print("\n" + "=" * 60)
    print(f"DATASET AUDIT: {name.upper()}")
    print("=" * 60)

    rows, columns = data.shape
    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())

    print(f"Records: {rows}")
    print(f"Columns: {columns}")
    print(f"Missing values: {missing_count}")
    print(f"Duplicate rows: {duplicate_count}")

    column_report = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isna().sum().values,
        "Unique Values": data.nunique(dropna=True).values,
    })

    column_report.to_csv(
        RESULTS_DIR / f"{name}_columns.csv",
        index=False
    )

    target = TARGET_COLUMNS.get(name.lower())

    minority_percentage = None

    if target in data.columns:
        counts = data[target].value_counts(
            dropna=False
        )

        print("\nTarget class distribution:")
        print(counts)

        counts.to_csv(
            RESULTS_DIR / f"{name}_class_distribution.csv"
        )

        valid_counts = data[target].dropna().value_counts()

        if not valid_counts.empty:
            minority_percentage = round(
                valid_counts.min() / valid_counts.sum() * 100,
                2
            )

            print(
                f"\nMinority class: "
                f"{minority_percentage}%"
            )
    else:
        print(
            f"\nWarning: Target column '{target}' "
            "not found. Check the dataset schema."
        )

    print("\nMissing values per column:")
    print(data.isna().sum().to_string())

    print("\nAudit completed.")

    return {
        "Dataset": name,
        "Records": rows,
        "Columns": columns,
        "Missing Values": missing_count,
        "Duplicate Rows": duplicate_count,
        "Minority Class (%)": minority_percentage,
    }


def main():
    print("MEDIRISK AI - DATASET QUALITY AUDIT")

    datasets = load_all_datasets()
    summaries = []

    for name, df in datasets.items():
        summary = audit_dataset(name, df)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        RESULTS_DIR / "summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print("\nReports saved in results/data_audit/")


if __name__ == "__main__":
    main()
