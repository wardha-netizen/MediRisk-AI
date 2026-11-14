
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Numerical columns in the kidney dataset
KIDNEY_NUMERICAL = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
]


# Prepare a disease dataset and convert its target into binary labels
def prepare_disease_dataset(disease_name, df):

    df = df.copy()

    if disease_name == "diabetes":
        target_column = "Diabetes_binary"

    elif disease_name == "heart":
        target_column = "num"

        # Convert heart disease target into binary values
        df[target_column] = (
            pd.to_numeric(df[target_column], errors="coerce") > 0
        ).astype(int)

        # Convert coded categorical features into categories
        heart_categorical = [
            "sex", "cp", "fbs", "restecg",
            "exang", "slope", "thal"
        ]

        for column in heart_categorical:
            df[column] = df[column].astype("category")

    elif disease_name == "kidney":
        target_column = "class"

        # Convert numeric columns stored as text into numbers
        for column in KIDNEY_NUMERICAL:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        # Convert CKD labels into binary values
        df[target_column] = (
            df[target_column].str.strip().str.lower()
            .map({"ckd": 1, "notckd": 0})
        )

    elif disease_name == "liver":
        target_column = "Dataset"

        # 1 means liver disease; 2 means no liver disease
        df[target_column] = df[target_column].map({
            1: 1,
            2: 0
        })

    elif disease_name == "stroke":
        target_column = "stroke"

        # ID is an identifier, not a useful prediction feature
        df = df.drop(columns=["id"])

    else:
        raise ValueError(f"Unknown disease: {disease_name}")

    return prepare_data(df, target_column)


# Prepare features, handle missing values and split the dataset
def prepare_data(df, target_column, test_size=0.2, random_state=42):

    df = df.copy()

    # Remove rows with missing target values
    df = df.dropna(subset=[target_column])

    # Separate input features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # Handle missing numerical values using the median
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Handle missing categorical values and encode text
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Split data before fitting preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Learn preprocessing rules using training data only
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same rules to testing data
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
