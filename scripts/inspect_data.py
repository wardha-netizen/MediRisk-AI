
import pandas as pd
from scipy.io import arff
from pathlib import Path

# Find the datasets folder
DATA_FOLDER = Path("datasets")


def inspect_dataset(name, file_name, file_type):
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    file_path = DATA_FOLDER / file_name

    try:
        # Load the dataset according to its file format
        if file_type == "csv":
            df = pd.read_csv(file_path)

        elif file_type == "heart":
            df = pd.read_csv(file_path, header=None, na_values="?")

        elif file_type == "arff":
            data, metadata = arff.loadarff(file_path)
            df = pd.DataFrame(data)

            # Convert byte values into readable text
            for column in df.select_dtypes(include="object").columns:
                df[column] = df[column].apply(
                    lambda value: value.decode("utf-8")
                    if isinstance(value, bytes)
                    else value
                )

        # Display dataset information
        print("\nDataset Shape (Rows, Columns):")
        print(df.shape)

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nData Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nFirst 5 Records:")
        print(df.head())

    except Exception as error:
        print(f"ERROR while loading {name}: {error}")


# Inspect all five disease datasets
inspect_dataset("Diabetes", "diabetes.csv", "csv")

inspect_dataset("Heart Disease", "heart.data", "heart")

inspect_dataset("Kidney Disease", "chronic.arff", "arff")

inspect_dataset("Liver Disease", "liver.csv", "csv")

inspect_dataset("Stroke", "stroke.csv", "csv")
