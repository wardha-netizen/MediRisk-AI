
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.tree import _tree


def explain_decision_tree(model, preprocessor, inputs):
    """
    Explain the path taken by a trained Decision Tree
    for one submitted record.
    """

    if isinstance(inputs, dict):
        input_df = pd.DataFrame([inputs])
    else:
        input_df = inputs.copy()

    transformed = preprocessor.transform(input_df)

    if issparse(transformed):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    tree = model.tree_
    node_indicator = model.decision_path(transformed)
    leaf_id = model.apply(transformed)[0]

    sample_nodes = node_indicator.indices[
        node_indicator.indptr[0]:
        node_indicator.indptr[1]
    ]

    explanation = []

    for node_id in sample_nodes:

        if node_id == leaf_id:
            continue

        feature_id = tree.feature[node_id]

        if feature_id == _tree.TREE_UNDEFINED:
            continue

        feature_name = feature_names[feature_id]
        threshold = tree.threshold[node_id]
        feature_value = transformed[0, feature_id]

        if feature_value <= threshold:
            operator = "<="
        else:
            operator = ">"

        explanation.append({
            "Feature": feature_name,
            "Value": round(float(feature_value), 3),
            "Rule": operator,
            "Threshold": round(float(threshold), 3),
        })

    return explanation
