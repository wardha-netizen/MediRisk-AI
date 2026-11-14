
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.data_loader import load_all_datasets
from src.preprocessing import prepare_disease_dataset


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"
RESULT_FOLDER = BASE_DIR / "results"

# Create results folder if it does not exist
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def evaluate_model(disease_name, df):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {disease_name.upper()}")
    print("=" * 60)

    # Prepare the same test split used during training
    X_train, X_test_processed, y_train, y_test, preprocessor = (
        prepare_disease_dataset(disease_name, df)
    )

    # Load the saved model and preprocessing pipeline
    model_path = MODEL_FOLDER / f"{disease_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    saved_preprocessor = saved_data["preprocessor"]

    # Generate predictions
    y_pred = model.predict(X_test_processed)

    # Calculate performance metrics
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
    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    # Calculate ROC-AUC using predicted probabilities
    y_probability = model.predict_proba(X_test_processed)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probability)

    # Display metrics
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")

    # Display detailed classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Disease", "Disease"],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Generate and save confusion matrix
    confusion = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["No Disease", "Disease"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{disease_name.capitalize()} Confusion Matrix")
    plt.tight_layout()

    matrix_path = RESULT_FOLDER / (
        f"{disease_name}_confusion_matrix.png"
    )

    plt.savefig(matrix_path, dpi=300)
    plt.close()

    # Save classification report
    report_path = RESULT_FOLDER / (
        f"{disease_name}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Confusion matrix saved: {matrix_path}")
    print(f"Classification report saved: {report_path}")

    return {
        "Disease": disease_name.capitalize(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Balanced Accuracy": balanced_accuracy,
        "ROC-AUC": roc_auc
    }


def evaluate_all_models():

    # Load all datasets
    datasets = load_all_datasets()

    results = []

    # Evaluate each saved model
    for disease_name, df in datasets.items():

        result = evaluate_model(disease_name, df)
        results.append(result)

    # Create performance comparison table
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("FINAL MODEL EVALUATION COMPARISON")
    print("=" * 75)

    print(results_df.round(4).to_string(index=False))

    # Save evaluation results as CSV
    results_path = RESULT_FOLDER / "evaluation_results.csv"

    results_df.to_csv(results_path, index=False)

    print(f"\nEvaluation results saved: {results_path}")
    print("\nEvaluation of all five models completed!")


if __name__ == "__main__":
    evaluate_all_models()
