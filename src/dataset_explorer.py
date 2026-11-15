
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_all_datasets


def render_dataset_explorer():
    st.title("Dataset Explorer")
    st.caption(
        "Explore the datasets used to train MediRisk AI's "
        "five Decision Tree classifiers."
    )

    try:
        datasets = load_all_datasets()
    except Exception as error:
        st.error(f"Unable to load datasets: {error}")
        return

    display_names = {
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
        "kidney": "Chronic Kidney Disease",
        "liver": "Liver Disease",
        "stroke": "Stroke",
    }

    disease = st.selectbox(
        "Select a disease dataset",
        list(datasets.keys()),
        format_func=lambda name: display_names.get(
            name, name.title()
        ),
    )

    df = datasets[disease].copy()

    missing_markers = [
        "?", "NA", "N/A", "null", "None", ""
    ]

    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace(missing_markers, pd.NA)
        )

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}"
    )
    col4.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )

    tab1, tab2, tab3 = st.tabs([
        "Dataset Preview",
        "Feature Analysis",
        "Data Quality"
    ])

    with tab1:
        st.markdown("### Sample Dataset Records")

        st.dataframe(
            df.head(30),
            use_container_width=True
        )

        st.markdown("### Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique Values": df.nunique().values,
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

    
    with tab2:
        st.markdown("### Target Class Distribution")

        target_columns = {
            "diabetes": "Diabetes_binary",
            "heart": "num",
            "kidney": "class",
            "liver": "Dataset",
            "stroke": "stroke",
        }

        target = target_columns.get(disease)

        if target in df.columns:
            class_counts = (
                df[target]
                .astype("string")
                .fillna("Missing")
                .value_counts()
                .reset_index()
            )

            class_counts.columns = ["Class", "Records"]

            fig = px.bar(
                class_counts,
                x="Class",
                y="Records",
                color="Class",
                title="Distribution of Dataset Labels",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                class_counts,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "Target column not found. "
                "Check the dataset's actual column names."
            )


        
        st.markdown("### Feature Distribution")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_feature = st.selectbox(
                "Select a numerical feature",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                title=f"{selected_feature} Distribution",
                color_discrete_sequence=["#167D8D"]
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#233547")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.info(
                "No numerical features found in "
                "this raw dataset."
            )

        st.markdown("### Correlation Heatmap")

        numeric_data = df.select_dtypes(
            include="number"
        )

        if numeric_data.shape[1] >= 2:
            correlation = numeric_data.corr()

            fig = px.imshow(
                correlation,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab3:
        st.markdown("### Missing Values by Column")

        missing = df.isna().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            missing_df = missing.reset_index()
            missing_df.columns = ["Feature", "Missing"]

            fig = px.bar(
                missing_df,
                x="Feature",
                y="Missing",
                color_discrete_sequence=["#167D8D"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.success(
                "No recognised missing values found."
            )

        st.markdown("### Dataset Quality Summary")

        st.write(
            f"**Duplicate records:** "
            f"{df.duplicated().sum()}"
        )

        st.write(
            f"**Missing values:** "
            f"{df.isna().sum().sum()}"
        )

    st.divider()

    st.download_button(
        "Download Dataset as CSV",
        data=df.to_csv(index=False),
        file_name=f"{disease}_dataset.csv",
        mime="text/csv"
    )

    st.info(
        "This page displays public datasets for academic "
        "analysis. MediRisk AI is not a medical diagnostic tool."
    )
