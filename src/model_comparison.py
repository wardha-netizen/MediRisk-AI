
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "results" / "evaluation_results.csv"

METRIC_NAMES = {
    "Accuracy": ["accuracy", "accuracy_score"],
    "Precision": ["precision", "precision_score"],
    "Recall": ["recall", "recall_score"],
    "F1-score": ["f1_score", "f1", "f1score"],
    "ROC-AUC": ["roc_auc", "roc_auc_score", "auc"],
}


def normalize_name(name):
    return str(name).strip().lower().replace(
        " ", "_"
    ).replace("-", "_")


def prepare_metrics(df):
    data = df.copy()

    columns = {
        normalize_name(column): column
        for column in data.columns
    }

    disease_col = next(
        (
            columns[key]
            for key in ["disease", "model", "dataset", "name"]
            if key in columns
        ),
        None
    )

    if disease_col is None:
        raise ValueError(
            "No disease/model name column found."
        )

    data = data.rename(columns={disease_col: "Disease"})

    found_metrics = []

    for display_name, aliases in METRIC_NAMES.items():
        source = next(
            (columns[key] for key in aliases if key in columns),
            None
        )

        if source is None:
            continue

        values = pd.to_numeric(
            data[source], errors="coerce"
        )

        if values.notna().any():
            maximum = values.max()
            minimum = values.min()

            if minimum >= 0 and maximum <= 1:
                values = values * 100
            elif minimum < 0 or maximum > 100:
                st.warning(
                    f"Unexpected values in {display_name}."
                )

        data[display_name] = values
        found_metrics.append(display_name)

    if not found_metrics:
        raise ValueError(
            "No supported evaluation metrics found."
        )

    return data, found_metrics


def render_model_comparison():
    st.title("Advanced Model Comparison")

    st.caption(
        "Compare the performance of MediRisk AI's "
        "five Decision Tree classifiers."
    )

    if not METRICS_PATH.exists():
        st.warning(
            "Evaluation results are missing. "
            "Run the existing model evaluation script."
        )
        return

    try:
        raw = pd.read_csv(METRICS_PATH)
        metrics, available = prepare_metrics(raw)

    except Exception as error:
        st.error(f"Unable to read model results: {error}")
        return

    st.subheader("Performance Overview")

    st.dataframe(
        metrics[["Disease"] + available],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Compare Evaluation Metrics")

    selected_metrics = st.multiselect(
        "Choose metrics",
        available,
        default=[
            metric for metric in
            ["Accuracy", "Precision", "Recall", "F1-score"]
            if metric in available
        ]
    )

    if selected_metrics:
        chart_data = metrics.melt(
            id_vars="Disease",
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Score"
        )

        chart_data = chart_data.dropna(
            subset=["Score"]
        )

        fig = px.bar(
            chart_data,
            x="Disease",
            y="Score",
            color="Metric",
            barmode="group",
            title="Decision Tree Model Comparison",
            color_discrete_sequence=[
                "#155E75",
                "#0D9488",
                "#F2B84B",
                "#64748B",
                "#8B5CF6"
            ]
        )

        fig.update_layout(
            yaxis_title="Score (%)",
            yaxis_range=[0, 100],
            xaxis_title="Disease",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Individual Model Analysis")

    disease = st.selectbox(
        "Choose a model to inspect",
        metrics["Disease"].astype(str).tolist()
    )

    selected = metrics[
        metrics["Disease"].astype(str) == disease
    ].iloc[0]

    if available:
        columns = st.columns(
            min(len(available), 5)
        )

        for i, metric in enumerate(available[:5]):
            value = selected[metric]

            columns[i].metric(
                metric,
                f"{value:.2f}%"
                if pd.notna(value) else "N/A"
            )

    st.markdown("### Model Review Notes")

    disease_lower = disease.lower()

    if "stroke" in disease_lower:
        st.warning(
            "Stroke classification requires careful review "
            "because the positive class may be much smaller "
            "than the negative class. Examine precision, "
            "recall and false negatives."
        )

    elif "kidney" in disease_lower:
        st.info(
            "The Kidney model has previously reported "
            "very high performance. Verify the train/test "
            "split, duplicate records and data leakage."
        )

    elif "liver" in disease_lower:
        st.info(
            "Review the Liver model's class balance, "
            "tree depth and generalisation performance."
        )

    else:
        st.info(
            "Review accuracy together with precision, "
            "recall and F1-score. No single metric fully "
            "describes classifier performance."
        )

    st.divider()

    st.subheader("Understanding the Metrics")

    with st.expander("What is Accuracy?"):
        st.write(
            "Accuracy measures the percentage of all "
            "test predictions that were correct."
        )

    with st.expander("What is Precision?"):
        st.write(
            "Precision measures how many predicted "
            "positive records were actually positive."
        )

    with st.expander("What is Recall?"):
        st.write(
            "Recall measures how many actual positive "
            "records the model correctly identified."
        )

    with st.expander("What is F1-score?"):
        st.write(
            "F1-score combines precision and recall "
            "using their harmonic mean."
        )

    st.warning(
        "These are experimental results on historical "
        "datasets. They are not clinically validated "
        "diagnostic performance measures."
    )
