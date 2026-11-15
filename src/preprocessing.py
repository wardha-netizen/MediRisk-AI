
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


# ==========================================
# PREPARE DISEASE DATASET
# ==========================================

def prepare_disease_dataset(disease_name, df):

    df = df.copy()

    # --------------------------------------
    # DIABETES
    # --------------------------------------

    if disease_name == "diabetes":

        target_column = "Diabetes_binary"

    # --------------------------------------
    # HEART DISEASE
    # --------------------------------------

    elif disease_name == "heart":

        target_column = "num"

        # Convert target labels into numeric values
        df[target_column] = pd.to_numeric(
            df[target_column],
            errors="coerce"
        )

        # Remove records with unknown target labels
        df = df.dropna(subset=[target_column])

        # Convert known outcomes into binary classes
        # 0 = negative, 1-4 = positive
        df[target_column] = (
            df[target_column] > 0
        ).astype(int)

        # Treat coded features as categorical
        heart_categorical = [
            "sex", "cp", "fbs", "restecg",
            "exang", "slope", "thal"
        ]

        for column in heart_categorical:
            if column in df.columns:
                df[column] = df[column].astype("category")

    # --------------------------------------
    # KIDNEY DISEASE
    # --------------------------------------

    elif disease_name == "kidney":

        target_column = "class"

        # Convert numeric columns stored as text
        for column in KIDNEY_NUMERICAL:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # Convert CKD labels into binary values
        df[target_column] = (
            df[target_column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "ckd": 1,
                "notckd": 0
            })
        )

    # --------------------------------------
    # LIVER DISEASE
    # --------------------------------------

    elif disease_name == "liver":

        target_column = "Dataset"

        # 1 means liver disease
        # 2 means no liver disease
        df[target_column] = (
            pd.to_numeric(
                df[target_column],
                errors="coerce"
            )
            .map({
                1: 1,
                2: 0
            })
        )

    # --------------------------------------
    # STROKE
    # --------------------------------------

    elif disease_name == "stroke":

        target_column = "stroke"

        # ID is an identifier, not a prediction feature
        df = df.drop(
            columns=["id"],
            errors="ignore"
        )

    # --------------------------------------
    # INVALID DISEASE NAME
    # --------------------------------------

    else:

        raise ValueError(
            f"Unknown disease: {disease_name}"
        )

    return prepare_data(df, target_column)


# ==========================================
# PREPARE DATA FOR MACHINE LEARNING
# ==========================================

def prepare_data(
    df,
    target_column,
    test_size=0.2,
    random_state=42
):

    df = df.copy()

    # Remove records with missing target labels
    df = df.dropna(
        subset=[target_column]
    )

    # Separate features and target
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column].astype(int)

    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # --------------------------------------
    # NUMERICAL PREPROCESSING
    # --------------------------------------

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    # --------------------------------------
    # CATEGORICAL PREPROCESSING
    # --------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    # --------------------------------------
    # COMBINE PREPROCESSING PIPELINES
    # --------------------------------------

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])

    # --------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Learn preprocessing rules from training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Apply the same preprocessing to testing data
    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------
    # RETURN PROCESSED DATA
    # --------------------------------------

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
