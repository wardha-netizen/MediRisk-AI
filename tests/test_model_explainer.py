
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from src.model_explainer import explain_decision_tree


def test_explanation_rules_match_decisions():
    df = pd.DataFrame({
        "age": [20, 25, 35, 45, 55, 65],
        "bmi": [19, 21, 24, 30, 35, 40],
        "outcome": [0, 0, 0, 1, 1, 1]
    })

    X = df[["age", "bmi"]]
    y = df["outcome"]

    preprocessor = ColumnTransformer([
        ("numerical", "passthrough", ["age", "bmi"])
    ])

    X_processed = preprocessor.fit_transform(X)

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )
    model.fit(X_processed, y)

    sample = {"age": 55, "bmi": 35}

    rules = explain_decision_tree(
        model, preprocessor, sample
    )

    assert isinstance(rules, list)
    assert len(rules) > 0

    for rule in rules:
        assert rule["Feature"] in preprocessor.get_feature_names_out()
        assert rule["Rule"] in ("<=", ">")

        if rule["Rule"] == "<=":
            assert rule["Value"] <= rule["Threshold"] + 0.001
        else:
            assert rule["Value"] > rule["Threshold"] - 0.001
