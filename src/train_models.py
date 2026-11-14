
import joblib

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import load_all_datasets
from src.preprocessing import prepare_disease_dataset


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"

# Create models folder if it does not exist
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


def train_model(disease_name, df):

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {disease_name.upper()}")
    print("=" * 60)

    # Prepare training and testing data
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_disease_dataset(disease_name, df)
    )

    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on testing data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Display evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Disease", "Disease"],
            zero_division=0
        )
    )

    # Save the model and its preprocessing pipeline together
    model_data = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": preprocessor.feature_names_in_.tolist()
    }

    model_path = MODEL_FOLDER / f"{disease_name}_model.joblib"

    joblib.dump(model_data, model_path)

    print(f"Model saved successfully: {model_path}")

    return {
        "disease": disease_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def train_all_models():

    # Load all five datasets
    datasets = load_all_datasets()

    results = []

    # Train one separate model for each disease
    for disease_name, df in datasets.items():

        result = train_model(disease_name, df)
        results.append(result)

    # Display final comparison
    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    for result in results:
        print(
            f"{result['disease'].capitalize():12} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"F1: {result['f1_score']:.4f}"
        )

    print("\nAll five models have been trained and saved!")


if __name__ == "__main__":
    train_all_models()
