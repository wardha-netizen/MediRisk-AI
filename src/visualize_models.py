
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.tree import plot_tree


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"
RESULT_FOLDER = BASE_DIR / "results"

TREE_FOLDER = RESULT_FOLDER / "decision_trees"
IMPORTANCE_FOLDER = RESULT_FOLDER / "feature_importance"

TREE_FOLDER.mkdir(parents=True, exist_ok=True)
IMPORTANCE_FOLDER.mkdir(parents=True, exist_ok=True)


DISEASES = [
    "diabetes",
    "heart",
    "kidney",
    "liver",
    "stroke"
]


def visualize_decision_tree(disease_name):
    print(f"\nVisualizing: {disease_name.upper()}")

    model_path = MODEL_FOLDER / f"{disease_name}_model.joblib"
    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    preprocessor = saved_data["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    # Create the Decision Tree diagram
    plt.figure(figsize=(24, 12))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=3
    )

    plt.title(
        f"{disease_name.capitalize()} Decision Tree",
        fontsize=18
    )

    plt.tight_layout()

    tree_path = TREE_FOLDER / f"{disease_name}_decision_tree.png"
    plt.savefig(tree_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Decision Tree saved: {tree_path}")

    # Get feature importance values
    importance_values = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Create the Feature Importance graph
    plt.figure(figsize=(12, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Feature Importance")
    plt.ylabel("Patient Features")

    plt.title(
        f"Top 15 Important Features - {disease_name.capitalize()}",
        fontsize=15
    )

    plt.tight_layout()

    importance_path = (
        IMPORTANCE_FOLDER / f"{disease_name}_feature_importance.png"
    )

    plt.savefig(importance_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature Importance saved: {importance_path}")


def visualize_all_models():
    for disease_name in DISEASES:
        visualize_decision_tree(disease_name)

    print("\nAll five Decision Trees visualized successfully!")
    print("All five Feature Importance graphs saved successfully!")


if __name__ == "__main__":
    visualize_all_models()
