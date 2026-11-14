
import pandas as pd
from pathlib import Path

# All 25 attribute names from the CKD dataset
columns = [
    "age", "bp", "sg", "al", "su", "rbc", "pc",
    "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc", "htn", "dm",
    "cad", "appet", "pe", "ane", "class"
]

file_path = Path("datasets/chronic.arff")

# Read the original file
lines = file_path.read_text(encoding="utf-8").splitlines()

# Find where the actual data starts
data_start = next(
    i for i, line in enumerate(lines)
    if line.strip().lower() == "@data"
)

records = []
bad_rows = []

# Read each data record
for line_number, line in enumerate(lines[data_start + 1:], start=data_start + 2):

    if not line.strip():
        continue

    values = [value.strip() for value in line.split(",")]

    # Remove an extra empty field caused by a trailing comma
    if len(values) == 26 and values[-1] == "":
        values.pop()

    # Check that each record has exactly 25 fields
    if len(values) != 25:
        bad_rows.append((line_number, len(values)))
        continue

    # Convert missing-value markers into proper missing values
    values = [
        pd.NA if value in ["?", ""] else value
        for value in values
    ]

    records.append(values)

# Create the DataFrame
df = pd.DataFrame(records, columns=columns)

print("KIDNEY DATASET INSPECTION")
print("-" * 45)

print("Dataset Shape:", df.shape)

print("\nTarget Counts:")
print(df["class"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Records:")
print(df.head())

print("\nRows with unexpected field counts:", bad_rows)
