
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "predictions"

DISEASES = [
    "diabetes",
    "heart",
    "kidney",
    "liver",
    "stroke"
]

COLORS = {
    "True Positive": "#0D9488",
    "True Negative": "#155E75",
    "False Positive": "#E9A23B",
    "False Negative": "#D64550"
}


def render_prediction_analysis():

    st.title("Actual vs Predicted Analysis")

    st.caption(
        "Explore how Decision Tree models classify "
        "previously unseen test records."
    )

    disease = st.selectbox(
        "Select Disease",
        DISEASES,
        format_func=lambda x: x.capitalize()
    )

    file_path = (
        RESULTS_DIR / f"{disease}_predictions.csv"
    )

    if not file_path.exists():
        st.warning(
            "Prediction results not found. "
            "Run: python -m scripts.export_predictions"
        )
        return

    df = pd.read_csv(file_path)

    required = {
        "Actual", "Predicted", "Result"
    }

    if not required.issubset(df.columns):
        st.error("Required prediction columns are missing.")
        return

    if df.empty:
        st.warning("No test records were found.")
        return

    actual = df["Actual"].astype(int)
    predicted = df["Predicted"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    correct = int(tp + tn)
    incorrect = int(fp + fn)
    accuracy = correct / len(df) * 100

    st.subheader("Test Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test Records", len(df))
    col2.metric("Correct", correct)
    col3.metric("Incorrect", incorrect)
    col4.metric("Accuracy", f"{accuracy:.2f}%")

    st.divider()

    st.subheader("Classification Outcomes")

    outcome_counts = (
        df["Result"]
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Records")
    )

    fig = px.bar(
        outcome_counts,
        x="Outcome",
        y="Records",
        color="Outcome",
        text="Records",
        color_discrete_map=COLORS,
        title="Correct and Incorrect Classifications"
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#F2F7FC",
        plot_bgcolor="#F2F7FC",
        font=dict(color="#17324D"),
        title_font=dict(color="#17324D"),
        xaxis=dict(
            tickfont=dict(color="#17324D"),
            title_font=dict(color="#17324D")
        ),
        yaxis=dict(
            tickfont=dict(color="#17324D"),
            title_font=dict(color="#17324D")
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None
    )

    st.divider()

    st.subheader("Confusion Matrix Breakdown")

    matrix_col1, matrix_col2 = st.columns(2)

    with matrix_col1:
        st.metric("True Positives", int(tp))
        st.metric("False Positives", int(fp))

    with matrix_col2:
        st.metric("True Negatives", int(tn))
        st.metric("False Negatives", int(fn))

    with st.expander("Understand the four outcomes"):
        st.write(
            "**True Positive:** Actual positive, "
            "predicted positive."
        )
        st.write(
            "**True Negative:** Actual negative, "
            "predicted negative."
        )
        st.write(
            "**False Positive:** Actual negative, "
            "predicted positive."
        )
        st.write(
            "**False Negative:** Actual positive, "
            "predicted negative."
        )

    st.divider()

    st.subheader("Inspect Individual Test Results")

    options = [
        "All",
        "True Positive",
        "True Negative",
        "False Positive",
        "False Negative"
    ]

    selected_filter = st.selectbox(
        "Filter classification results",
        options
    )

    filtered = df.copy()

    if selected_filter != "All":
        filtered = filtered[
            filtered["Result"] == selected_filter
        ]

    st.write(
        f"Showing {len(filtered)} of {len(df)} "
        "test records."
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Results as CSV",
        data=filtered.to_csv(index=False),
        file_name=f"{disease}_prediction_analysis.csv",
        mime="text/csv"
    )

    st.divider()

    st.info(
        "These classifications are based on "
        "held-out historical dataset records. "
        "They are for educational analysis, "
        "not medical diagnosis."
    )
