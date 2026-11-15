
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

COLORS = {
    "Accuracy": "#155E75",
    "Precision": "#0D9488",
    "Recall": "#F2B84B",
    "F1-score": "#64748B",
    "ROC-AUC": "#8B5CF6",
}


def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


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

    data = data.rename(
        columns={disease_col: "Disease"}
    )

    found_metrics = []

    for display_name, aliases in METRIC_NAMES.items():
        source = next(
            (
                columns[key]
                for key in aliases
                if key in columns
            ),
            None
        )

        if source is None:
            continue

        values = pd.to_numeric(
            data[source],
            errors="coerce"
        )

        if values.notna().any():
            minimum = values.min()
            maximum = values.max()

            if minimum >= 0 and maximum <= 1:
                values = values * 100

            elif minimum < 0 or maximum > 100:
                st.warning(
                    f"Unexpected values detected in {display_name}."
                )

        data[display_name] = values
        found_metrics.append(display_name)

    if not found_metrics:
        raise ValueError(
            "No supported evaluation metrics were found."
        )

    return data, found_metrics


def style_chart(fig):
    """Apply a readable light theme to Plotly charts."""

    fig.update_layout(
        template="plotly_white",

        font=dict(
            family="Arial",
            size=14,
            color="#17324D"
        ),

        title=dict(
            font=dict(
                size=22,
                color="#17324D"
            ),
            x=0.02
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_font=dict(
                color="#17324D",
                size=15
            ),
            tickfont=dict(
                color="#334155",
                size=13
            ),
            showgrid=False,
            linecolor="#CBD5E1",
            zeroline=False
        ),

        yaxis=dict(
            title_font=dict(
                color="#17324D",
                size=15
            ),
            tickfont=dict(
                color="#334155",
                size=12
            ),
            range=[0, 100],
            gridcolor="#D9E3EC",
            zeroline=False
        ),

        legend=dict(
            title=dict(
                text="Evaluation Metric",
                font=dict(color="#17324D")
            ),
            font=dict(
                color="#334155",
                size=13
            ),
            bgcolor="rgba(255,255,255,0.9)",
            orientation="h",
            y=-0.28,
            x=0.5,
            xanchor="center"
        ),

        margin=dict(
            l=40,
            r=30,
            t=80,
            b=110
        ),

        height=520,

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(color="#17324D")
        )
    )

    return fig


def render_model_comparison():
    st.title("Advanced Model Comparison")

    st.caption(
        "Compare the performance of MediRisk AI's "
        "five Decision Tree classification models."
    )

    if not METRICS_PATH.exists():
        st.warning(
            "Evaluation results are missing. "
            "Please run the model evaluation script first."
        )
        return

    try:
        raw = pd.read_csv(METRICS_PATH)

        metrics, available = prepare_metrics(raw)

    except Exception as error:
        st.error(
            f"Unable to load evaluation results: {error}"
        )
        return

    # ==========================================
    # PERFORMANCE OVERVIEW
    # ==========================================

    st.subheader("Performance Overview")

    display_df = metrics[
        ["Disease"] + available
    ].copy()

    for metric in available:
        display_df[metric] = display_df[metric].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # MODEL METRIC COMPARISON
    # ==========================================

    st.subheader("Compare Evaluation Metrics")

    selected_metrics = st.multiselect(
        "Choose metrics",
        available,
        default=[
            metric
            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score"
            ]
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
            title="Decision Tree Model Performance Comparison",
            color_discrete_map=COLORS,
            text_auto=".1f"
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                color="#17324D",
                size=11
            ),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Score: %{y:.2f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            xaxis_title="Disease",
            yaxis_title="Score (%)"
        )

        fig = style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info(
            "Select at least one metric to display the chart."
        )

    st.divider()

    # ==========================================
    # INDIVIDUAL MODEL ANALYSIS
    # ==========================================

    st.subheader("Individual Model Analysis")

    disease = st.selectbox(
        "Choose a model to inspect",
        metrics["Disease"].astype(str).tolist()
    )

    selected = metrics[
        metrics["Disease"].astype(str) == disease
    ].iloc[0]

    metric_columns = st.columns(
        min(len(available), 5)
    )

    for index, metric in enumerate(available[:5]):

        value = selected[metric]

        metric_columns[index].metric(
            metric,
            f"{value:.2f}%"
            if pd.notna(value)
            else "N/A"
        )

    # ==========================================
    # MODEL REVIEW NOTES
    # ==========================================

    st.markdown("### Model Review Notes")

    disease_lower = disease.lower()

    if "stroke" in disease_lower:
        st.warning(
            "Stroke classification requires further investigation. "
            "Positive cases may be underrepresented in the dataset. "
            "Review precision, recall, false positives and "
            "false negatives."
        )

    elif "kidney" in disease_lower:
        st.info(
            "The Kidney Disease model previously reported very "
            "high accuracy. Check whether duplicates or data "
            "leakage may affect the results."
        )

    elif "liver" in disease_lower:
        st.info(
            "The Liver Disease model may benefit from reviewing "
            "class balance, Decision Tree depth and "
            "generalisation performance."
        )

    else:
        st.info(
            "Evaluate the model using accuracy, precision, "
            "recall and F1-score together. "
            "No single metric gives a complete picture."
        )

    st.divider()

    # ==========================================
    # PERFORMANCE INTERPRETATION
    # ==========================================

    st.subheader("Understanding Evaluation Metrics")

    with st.expander("What is Accuracy?"):
        st.write(
            "Accuracy is the proportion of all test records "
            "classified correctly."
        )

    with st.expander("What is Precision?"):
        st.write(
            "Precision measures how many records predicted "
            "as positive were actually positive."
        )

    with st.expander("What is Recall?"):
        st.write(
            "Recall measures how many actual positive records "
            "the model correctly identified."
        )

    with st.expander("What is F1-score?"):
        st.write(
            "F1-score is the harmonic mean of precision "
            "and recall. It is particularly useful when "
            "class distributions are imbalanced."
        )

    st.warning(
        "MediRisk AI is an educational machine learning project. "
        "Results are based on historical datasets and are not "
        "clinically validated medical assessments."
    )
