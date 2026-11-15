
from pathlib import Path
import ast

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
TUNING = RESULTS / "tuning"
KIDNEY = RESULTS / "kidney_audit"

NAVY = "#17324D"
TEAL = "#0D9488"
BLUE = "#155E75"
BACKGROUND = "#F2F7FC"

METRICS = {
    "Accuracy": ["accuracy", "accuracy_score"],
    "Precision": ["precision", "precision_score"],
    "Recall": ["recall", "recall_score"],
    "F1-score": ["f1_score", "f1score", "f1"],
}


def normalize(text):
    return (
        str(text).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, aliases):
    lookup = {normalize(c): c for c in df.columns}

    for name in aliases:
        if name in lookup:
            return lookup[name]

    return None


def load_csv(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load {path.name}: {error}")
        return None


def percentage(value):
    if pd.isna(value):
        return None

    value = float(value)

    # Project metrics may be stored as 0-1 or 0-100.
    if 0 <= value <= 1:
        return value * 100

    if 1 < value <= 100:
        return value

    return None


def extract_metrics(df, disease=None):
    if df is None or df.empty:
        return None

    data = df.copy()

    if disease:
        disease_column = find_column(
            data, ["disease", "model", "dataset", "name"]
        )

        if disease_column:
            data = data[
                data[disease_column].astype(str)
                .str.strip().str.lower()
                .str.contains(disease.lower(), regex=False)
            ]

    if data.empty:
        return None

    row = data.iloc[0]
    result = {}

    for metric, aliases in METRICS.items():
        column = find_column(data, aliases)

        if column:
            result[metric] = percentage(
                pd.to_numeric(row[column], errors="coerce")
            )

    return result or None


def chart_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=NAVY, size=14),
        title_font=dict(color=NAVY, size=20),
        xaxis=dict(
            tickfont=dict(color=NAVY),
            title_font=dict(color=NAVY),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color=NAVY),
            title_font=dict(color=NAVY),
            gridcolor="#D7E2EA"
        ),
        legend=dict(
            font=dict(color=NAVY),
            bgcolor="#FFFFFF"
        ),
        height=470,
        margin=dict(l=45, r=25, t=65, b=65)
    )

    return fig


def render_tuning_experiment(disease):
    st.subheader(f"{disease.title()} Model Tuning")

    baseline_file = RESULTS / "evaluation_results.csv"

    tuned_file = (
        TUNING / f"{disease}_tuned_metrics.csv"
    )

    cv_file = (
        TUNING / f"{disease}_cv_results.csv"
    )

    baseline = extract_metrics(
        load_csv(baseline_file), disease
    )

    tuned = extract_metrics(
        load_csv(tuned_file)
    )

    if baseline is None or tuned is None:
        st.warning(
            "Baseline or tuned metrics are missing. "
            "Check the evaluation and tuning CSV files."
        )
        return

    rows = []

    for metric in METRICS:
        original = baseline.get(metric)
        experimental = tuned.get(metric)

        if original is None or experimental is None:
            continue

        rows.append({
            "Metric": metric,
            "Baseline": original,
            "Tuned": experimental,
            "Difference": experimental - original
        })

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        st.warning("No comparable metrics found.")
        return

    st.markdown("### Baseline vs Experimental Results")

    st.dataframe(
        comparison.style.format({
            "Baseline": "{:.2f}%",
            "Tuned": "{:.2f}%",
            "Difference": "{:+.2f} pp"
        }),
        use_container_width=True,
        hide_index=True
    )

    chart_data = comparison.melt(
        id_vars="Metric",
        value_vars=["Baseline", "Tuned"],
        var_name="Model",
        value_name="Score"
    )

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_map={
            "Baseline": BLUE,
            "Tuned": TEAL
        },
        title=f"{disease.title()} Performance Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color=NAVY, size=12),
        cliponaxis=False
    )

    fig.update_yaxes(
        title_text="Score (%)",
        range=[0, 110]
    )

    chart_theme(fig)

    st.plotly_chart(
        fig,
        theme=None,
        use_container_width=True
    )

    st.info(
        "These results are comparable only if the "
        "baseline and tuned models used the same "
        "test records, target labels and preprocessing "
        "definitions. A higher score alone does not "
        "establish a clinically better model."
    )

    st.divider()
    st.markdown("### Cross-Validation Experiments")

    cv_results = load_csv(cv_file)

    if cv_results is None or cv_results.empty:
        st.warning("Cross-validation results not found.")
        return

    rank_column = find_column(
        cv_results, ["rank_test_score"]
    )

    if rank_column:
        cv_results = cv_results.sort_values(rank_column)

    top_results = cv_results.head(10).copy()

    st.dataframe(
        top_results,
        use_container_width=True,
        hide_index=True
    )

    if "mean_test_score" in top_results.columns:
        top_results["CV F1 (%)"] = (
            pd.to_numeric(
                top_results["mean_test_score"],
                errors="coerce"
            ) * 100
        )

        top_results["Configuration"] = [
            f"Rank {i + 1}"
            for i in range(len(top_results))
        ]

        fig = px.bar(
            top_results,
            x="Configuration",
            y="CV F1 (%)",
            color_discrete_sequence=[TEAL],
            title="Top Cross-Validation Configurations"
        )

        fig.update_traces(
            text=top_results["CV F1 (%)"],
            texttemplate="%{text:.2f}%",
            textposition="outside",
            textfont=dict(color=NAVY)
        )

        chart_theme(fig)
        fig.update_yaxes(rangemode="tozero")

        st.plotly_chart(
            fig,
            theme=None,
            use_container_width=True
        )

    if "params" in cv_results.columns:
        best_params = cv_results.iloc[0]["params"]

        try:
            parsed = ast.literal_eval(best_params)
            st.markdown("### Best Cross-Validation Parameters")
            st.json(parsed)
        except (ValueError, SyntaxError, TypeError):
            st.code(str(best_params))

    st.caption(
        "Configurations were ranked using training-set "
        "cross-validation F1-score."
    )


def render_kidney_experiment():
    st.subheader("Kidney Model Reliability")

    path = (
        KIDNEY / "kidney_train_test_comparison.csv"
    )

    data = load_csv(path)

    if data is None or data.empty:
        st.warning(
            "Kidney audit results not found. "
            "Run python -m scripts.kidney_reliability_audit"
        )
        return

    st.markdown("### Training vs Testing Performance")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    split_column = find_column(data, ["split"])

    if not split_column:
        st.warning("Split column not found.")
        return

    rows = []

    for metric, aliases in METRICS.items():
        column = find_column(data, aliases)

        if not column:
            continue

        for _, row in data.iterrows():
            value = percentage(
                pd.to_numeric(
                    row[column], errors="coerce"
                )
            )

            if value is not None:
                rows.append({
                    "Split": str(row[split_column]),
                    "Metric": metric,
                    "Score": value
                })

    if rows:
        chart_data = pd.DataFrame(rows)

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Split",
            barmode="group",
            text="Score",
            color_discrete_sequence=[BLUE, TEAL],
            title="Kidney Training vs Testing"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color=NAVY)
        )

        fig.update_yaxes(
            title_text="Score (%)",
            range=[0, 110]
        )

        chart_theme(fig)

        st.plotly_chart(
            fig,
            theme=None,
            use_container_width=True
        )

    st.info(
        "A small training-testing gap does not prove "
        "that data leakage is absent. Duplicate-record "
        "overlap, preprocessing consistency and external "
        "generalisation must be assessed separately."
    )

    st.caption(
        "The duplicate-overlap findings from the audit "
        "script should also be documented in the final report."
    )


def render_model_experiments():
    st.title("Model Experiments")

    st.caption(
        "Explore Decision Tree tuning experiments, "
        "cross-validation scores and model reliability."
    )

    tab1, tab2, tab3 = st.tabs([
        "Stroke Tuning",
        "Liver Tuning",
        "Kidney Reliability"
    ])

    with tab1:
        render_tuning_experiment("stroke")

    with tab2:
        render_tuning_experiment("liver")

    with tab3:
        render_kidney_experiment()

    st.divider()

    st.warning(
        "MediRisk AI is an academic machine learning "
        "demonstration, not a clinically validated "
        "medical diagnostic system."
    )
