
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "results" / "evaluation_results.csv"


# ==========================================
# COLOUR PALETTE
# ==========================================

NAVY = "#17324D"
TEAL = "#0D9488"
BACKGROUND = "#F2F7FC"
SLATE = "#475569"

COLORS = {
    "Accuracy": "#155E75",
    "Precision": "#0D9488",
    "Recall": "#F2B84B",
    "F1-score": "#64748B",
    "ROC-AUC": "#8B5CF6",
}


# ==========================================
# METRIC COLUMN NAMES
# ==========================================

METRIC_NAMES = {
    "Accuracy": ["accuracy", "accuracy_score"],
    "Precision": ["precision", "precision_score"],
    "Recall": ["recall", "recall_score"],
    "F1-score": ["f1_score", "f1", "f1score"],
    "ROC-AUC": ["roc_auc", "roc_auc_score", "auc"],
}


def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ==========================================
# LOAD AND PREPARE METRICS
# ==========================================

def prepare_metrics(df):
    data = df.copy()

    columns = {
        normalize_name(column): column
        for column in data.columns
    }

    disease_column = next(
        (
            columns[name]
            for name in ["disease", "model", "dataset", "name"]
            if name in columns
        ),
        None
    )

    if disease_column is None:
        raise ValueError(
            "The evaluation file must contain a "
            "disease or model name column."
        )

    data = data.rename(
        columns={disease_column: "Disease"}
    )

    found_metrics = []

    for metric_name, aliases in METRIC_NAMES.items():

        source_column = next(
            (
                columns[name]
                for name in aliases
                if name in columns
            ),
            None
        )

        if source_column is None:
            continue

        values = pd.to_numeric(
            data[source_column],
            errors="coerce"
        )

        valid_values = values.dropna()

        if not valid_values.empty:
            minimum = valid_values.min()
            maximum = valid_values.max()

            # Convert fractions to percentages.
            if minimum >= 0 and maximum <= 1:
                values = values * 100

            elif minimum < 0 or maximum > 100:
                raise ValueError(
                    f"{metric_name} has scores "
                    "outside the expected range."
                )

        data[metric_name] = values
        found_metrics.append(metric_name)

    if not found_metrics:
        raise ValueError(
            "No supported evaluation metrics found."
        )

    return data, found_metrics


# ==========================================
# LIGHT PLOTLY CHART THEME
# ==========================================

def style_chart(fig):
    """
    Apply a consistent light theme with
    readable dark text and percentage labels.
    """

    fig.update_layout(
        template="plotly_white",

        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,

        font=dict(
            family="Arial",
            size=15,
            color=NAVY
        ),

        title=dict(
            text="Decision Tree Model Performance Comparison",
            font=dict(
                family="Arial",
                size=23,
                color=NAVY
            ),
            x=0.02,
            xanchor="left"
        ),

        xaxis=dict(
            title=dict(
                text="Disease",
                font=dict(
                    size=16,
                    color=NAVY
                )
            ),
            tickfont=dict(
                size=14,
                color=NAVY
            ),
            showgrid=False,
            showline=True,
            linecolor="#94A3B8",
            zeroline=False
        ),

        yaxis=dict(
            title=dict(
                text="Performance Score (%)",
                font=dict(
                    size=16,
                    color=NAVY
                )
            ),
            tickfont=dict(
                size=13,
                color=NAVY
            ),
            range=[0, 112],
            gridcolor="#D9E3EC",
            zeroline=False
        ),

        legend=dict(
            title=dict(
                text="Evaluation Metric",
                font=dict(
                    color=NAVY,
                    size=13
                )
            ),
            font=dict(
                color=NAVY,
                size=13
            ),
            bgcolor="#FFFFFF",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25
        ),

        margin=dict(
            l=65,
            r=35,
            t=90,
            b=120
        ),

        height=570,

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(
                color=NAVY,
                size=13
            )
        )
    )

    # Explicitly override axis text colours.
    fig.update_xaxes(
        tickfont=dict(color=NAVY, size=14),
        title_font=dict(color=NAVY, size=16)
    )

    fig.update_yaxes(
        tickfont=dict(color=NAVY, size=13),
        title_font=dict(color=NAVY, size=16)
    )

    # Show values above every bar.
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        textfont=dict(
            family="Arial",
            size=12,
            color=NAVY
        ),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Score: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    return fig


# ==========================================
# STREAMLIT MODEL COMPARISON PAGE
# ==========================================

def render_model_comparison():

    st.title("Advanced Model Comparison")

    st.caption(
        "Compare the evaluation results of MediRisk AI's "
        "five Decision Tree classification models."
    )

    # ======================================
    # READ METRICS
    # ======================================

    if not METRICS_PATH.exists():
        st.warning(
            "Evaluation results not found. "
            "Run the project's model evaluation script."
        )
        return

    try:
        raw_data = pd.read_csv(METRICS_PATH)

        if raw_data.empty:
            st.warning("The evaluation results file is empty.")
            return

        metrics, available = prepare_metrics(raw_data)

    except Exception as error:
        st.error(
            f"Unable to load evaluation metrics: {error}"
        )
        return

    # ======================================
    # PERFORMANCE OVERVIEW
    # ======================================

    st.subheader("Performance Overview")

    display_data = metrics[
        ["Disease"] + available
    ].copy()

    for metric in available:
        display_data[metric] = display_data[metric].round(2)

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Scores are loaded from the project's saved "
        "evaluation results. Values are displayed as percentages."
    )

    st.divider()

    # ======================================
    # METRIC COMPARISON CHART
    # ======================================

    st.subheader("Compare Evaluation Metrics")

    default_metrics = [
        metric
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score"
        ]
        if metric in available
    ]

    selected_metrics = st.multiselect(
        "Choose metrics",
        options=available,
        default=default_metrics,
        key="comparison_metrics"
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

        if not chart_data.empty:

            fig = px.bar(
                chart_data,
                x="Disease",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_map=COLORS,
                title="Decision Tree Model Performance Comparison"
            )

            # Apply all styling after the chart is created.
            fig = style_chart(fig)

            # IMPORTANT:
            # theme=None prevents Streamlit from overriding
            # the Plotly figure's own colours and fonts.
            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None,
                key="medirisk_model_comparison_chart"
            )

        else:
            st.info(
                "No valid scores are available "
                "for the selected metrics."
            )

    else:
        st.info(
            "Select at least one metric to display the chart."
        )

    st.divider()

    # ======================================
    # INDIVIDUAL MODEL ANALYSIS
    # ======================================

    st.subheader("Individual Model Analysis")

    disease = st.selectbox(
        "Choose a disease model",
        metrics["Disease"].astype(str).tolist(),
        key="individual_model_selection"
    )

    selected_row = metrics[
        metrics["Disease"].astype(str) == disease
    ].iloc[0]

    metric_columns = st.columns(
        min(len(available), 5)
    )

    for index, metric in enumerate(available[:5]):

        value = selected_row[metric]

        metric_columns[index].metric(
            label=metric,
            value=(
                f"{value:.2f}%"
                if pd.notna(value)
                else "N/A"
            )
        )

    st.divider()

    # ======================================
    # MODEL REVIEW NOTES
    # ======================================

    st.subheader("Model Review Notes")

    disease_lower = disease.lower()

    if "stroke" in disease_lower:

        st.warning(
            "The Stroke model requires closer evaluation. "
            "Examine positive-class precision, recall, "
            "false positives, false negatives and "
            "class imbalance. High accuracy alone "
            "does not demonstrate reliable positive-case "
            "classification."
        )

    elif "kidney" in disease_lower:

        st.info(
            "The Kidney model has reported very high "
            "performance. Review duplicate records, "
            "train-test separation and preprocessing "
            "to verify that the score is reliable."
        )

    elif "liver" in disease_lower:

        st.info(
            "Review Liver model performance using "
            "precision, recall, F1-score and class "
            "distribution. Consider testing different "
            "Decision Tree parameters with cross-validation."
        )

    else:

        st.info(
            "Compare accuracy with precision, recall "
            "and F1-score to better understand how "
            "the model performs on different classes."
        )

    # ======================================
    # MODEL PERFORMANCE INTERPRETATION
    # ======================================

    st.divider()

    st.subheader("Understanding Evaluation Metrics")

    with st.expander("What is Accuracy?"):
        st.write(
            "Accuracy measures the percentage of "
            "all test records classified correctly."
        )

    with st.expander("What is Precision?"):
        st.write(
            "Precision measures how many records "
            "predicted as positive actually belong "
            "to the positive class."
        )

    with st.expander("What is Recall?"):
        st.write(
            "Recall measures how many actual "
            "positive records the model identifies."
        )

    with st.expander("What is F1-score?"):
        st.write(
            "F1-score is the harmonic mean of "
            "precision and recall. It is useful "
            "when studying class imbalance."
        )

    with st.expander("What is ROC-AUC?"):
        st.write(
            "ROC-AUC measures how well a model "
            "ranks positive cases above negative "
            "cases across different classification "
            "thresholds."
        )

    # ======================================
    # EDUCATIONAL DISCLAIMER
    # ======================================

    st.divider()

    st.warning(
        "MediRisk AI is an educational machine "
        "learning project. Its predictions and "
        "performance metrics are based on historical "
        "datasets and are not clinically validated "
        "medical assessments."
    )
