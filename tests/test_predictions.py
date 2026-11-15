
import pytest
import numpy as np

from src.data_loader import load_all_datasets
from src.preprocessing import prepare_disease_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def datasets():
    return load_all_datasets()


@pytest.mark.parametrize(
    "disease",
    ["diabetes", "heart", "kidney", "liver", "stroke"]
)
def test_model_training_and_prediction(datasets, disease):

    df = datasets[disease]

    X_train, X_test, y_train, y_test, _ = (
        prepare_disease_dataset(disease, df)
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)

    assert set(np.unique(predictions)).issubset({0, 1})

    accuracy = accuracy_score(y_test, predictions)

    assert 0 <= accuracy <= 1


@pytest.mark.parametrize(
    "disease",
    ["diabetes", "heart", "kidney", "liver", "stroke"]
)
def test_preprocessed_data_has_valid_dimensions(
    datasets, disease
):

    df = datasets[disease]

    X_train, X_test, y_train, y_test, _ = (
        prepare_disease_dataset(disease, df)
    )

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    assert X_train.shape[1] == X_test.shape[1]

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
