
import pandas as pd
from pathlib import Path


# Find the datasets folder
DATA_FOLDER = Path(__file__).resolve().parent.parent / "datasets"


# Column names for datasets without headers
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

LIVER_COLUMNS = [
    "Age", "Gender", "Total_Bilirubin",
    "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio", "Dataset"
]

KIDNEY_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc",
    "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc", "htn", "dm",
    "cad", "appet", "pe", "ane", "class"
]


def load_kidney_dataset():
    """Load the CKD ARFF file using a custom parser."""

    file_path = DATA_FOLDER / "chronic.arff"

    lines = file_path.read_text(encoding="utf-8").splitlines()

    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip().lower() == "@data"
    )

    records = []

    for line in lines[data_start + 1:]:
        if not line.strip():
            continue

        values = [value.strip() for value in line.split(",")]

        # Handle an optional trailing empty field
        if len(values) == 26 and values[-1] == "":
            values.pop()

        if len(values) != 25:
            raise ValueError(
                f"Invalid CKD row: expected 25 values, got {len(values)}"
            )

        values = [
            None if value in ["?", ""] else value
            for value in values
        ]

        records.append(values)

    return pd.DataFrame(records, columns=KIDNEY_COLUMNS)


def load_diabetes():
    """Load the diabetes dataset."""
    return pd.read_csv(DATA_FOLDER / "diabetes.csv")


def load_heart():
    """Load the heart disease dataset."""
    return pd.read_csv(
        DATA_FOLDER / "heart.data",
        header=None,
        names=HEART_COLUMNS,
        na_values="?"
    )


def load_kidney():
    """Load the chronic kidney disease dataset."""
    return load_kidney_dataset()


def load_liver():
    """Load the liver disease dataset."""
    return pd.read_csv(
        DATA_FOLDER / "liver.csv",
        header=None,
        names=LIVER_COLUMNS
    )


def load_stroke():
    """Load the stroke dataset."""
    return pd.read_csv(DATA_FOLDER / "stroke.csv")


def load_all_datasets():
    """Load and return all five disease datasets."""

    datasets = {
        "diabetes": load_diabetes(),
        "heart": load_heart(),
        "kidney": load_kidney(),
        "liver": load_liver(),
        "stroke": load_stroke()
    }

    return datasets
