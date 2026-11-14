
import pandas as pd
from pathlib import Path

# Find the datasets folder
DATA_FOLDER = Path("datasets")


# Column names for the heart disease dataset
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]


# Column names for the kidney disease dataset
KIDNEY_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc",
    "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc", "htn", "dm",
    "cad", "appet", "pe", "ane", "class"
]


# Column names for the liver disease dataset
LIVER_COLUMNS = [
    "Age", "Gender", "Total_Bilirubin",
    "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio", "Dataset"
]


def load_kidney_dataset(file_path):
    # Read the ARFF file as text
    lines = file_path.read_text(encoding="utf-8").splitlines()

    # Find where the actual patient records begin
    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip().lower() == "@data"
    )

    records = []
    bad_rows = []

    # Read patient records one by one
    for line_number, line in enumerate(
        lines[data_start + 1:], start=data_start + 2
    ):
        if not line.strip():
            continue

        values = [value.strip() for value in line.split(",")]

        # Remove an extra trailing empty field if present
        if len(values) == 26 and values[-1] == "":
            values.pop()

        # Check that every patient has exactly 25 fields
        if len(values) != 25:
            bad_rows.append((line_number, len(values)))
            continue

        # Convert empty values and question marks into missing values
        values = [
            pd.NA if value in ["?", ""] else value
            for value in values
        ]

        records.append(values)

    df = pd.DataFrame(records, columns=KIDNEY_COLUMNS)

    return df, bad_rows


def inspect_dataset(name, file_name, file_type, target_column):
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    file_path = DATA_FOLDER / file_name

    try:
        # Load the dataset according to its file format
        if file_type == "csv":
            df = pd.read_csv(file_path)

        elif file_type == "heart":
            df = pd.read_csv(
                file_path,
                header=None,
                names=HEART_COLUMNS,
                na_values="?"
            )

        elif file_type == "liver":
            df = pd.read_csv(
                file_path,
                header=None,
                names=LIVER_COLUMNS
            )

        elif file_type == "kidney":
            df, bad_rows = load_kidney_dataset(file_path)

            print("\nRows with unexpected field counts:", bad_rows)

        # Display dataset information
        print("\nDataset Shape (Rows, Columns):")
        print(df.shape)

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nData Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        print(df.isnull().sum())

        # Display target information
        print("\nTarget Column:", target_column)

        print("\nTarget Counts:")
        print(df[target_column].value_counts(dropna=False))

        # Display first five patient records
        print("\nFirst 5 Records:")
        print(df.head())

    except Exception as error:
        print(f"ERROR while loading {name}: {error}")


# Inspect all five disease datasets

inspect_dataset(
    "Diabetes", "diabetes.csv", "csv", "Diabetes_binary"
)

inspect_dataset(
    "Heart Disease", "heart.data", "heart", "num"
)

inspect_dataset(
    "Kidney Disease", "chronic.arff", "kidney", "class"
)

inspect_dataset(
    "Liver Disease", "liver.csv", "liver", "Dataset"
)

inspect_dataset(
    "Stroke", "stroke.csv", "csv", "stroke"
)
