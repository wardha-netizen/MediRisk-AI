
import streamlit as st


DISEASE_GUIDANCE = {
    "diabetes": {
        "title": "Understanding Diabetes Inputs",
        "description": (
            "This model uses health indicators and "
            "lifestyle-related information from the "
            "CDC Diabetes Health Indicators dataset."
        ),
        "tips": [
            "BMI means Body Mass Index.",
            "Age is recorded as a dataset category, "
            "not an exact age.",
            "General Health is recorded on a scale "
            "from 1 (excellent) to 5 (poor).",
            "Mental and Physical Health represent "
            "the number of affected days in the "
            "previous 30 days.",
            "Binary indicators generally use "
            "0 for No and 1 for Yes."
        ]
    },

    "heart": {
        "title": "Understanding Heart Disease Inputs",
        "description": (
            "This model uses cardiovascular "
            "measurements and coded examination results."
        ),
        "tips": [
            "Resting blood pressure is measured in mmHg.",
            "Serum cholesterol is measured in mg/dL.",
            "Maximum heart rate represents the "
            "highest recorded heart rate.",
            "Chest pain type and ECG results use "
            "codes defined by the source dataset.",
            "ST depression (Oldpeak) is an "
            "electrocardiogram-related measurement."
        ]
    },

    "kidney": {
        "title": "Understanding Kidney Disease Inputs",
        "description": (
            "This model uses laboratory measurements "
            "and clinical indicators related to "
            "kidney function."
        ),
        "tips": [
            "Specific gravity is a urine measurement.",
            "Albumin and sugar are coded using "
            "the source dataset's categories.",
            "Serum creatinine and blood urea are "
            "laboratory measurements.",
            "Hemoglobin and red blood cell counts "
            "are blood-related measurements.",
            "Categorical values must match the "
            "labels used during model training."
        ]
    },

    "liver": {
        "title": "Understanding Liver Disease Inputs",
        "description": (
            "This model uses liver-related laboratory "
            "measurements and patient characteristics."
        ),
        "tips": [
            "Bilirubin measurements are commonly "
            "used in liver-function assessments.",
            "ALT and AST are liver-related enzymes.",
            "Total proteins and albumin are "
            "laboratory measurements.",
            "The albumin/globulin ratio compares "
            "two groups of blood proteins.",
            "All measurements should use units "
            "consistent with the training dataset."
        ]
    },

    "stroke": {
        "title": "Understanding Stroke Inputs",
        "description": (
            "This model uses demographic, health "
            "and lifestyle indicators from a "
            "historical Stroke dataset."
        ),
        "tips": [
            "Hypertension and heart disease history "
            "are binary indicators.",
            "Average glucose level is measured "
            "using the dataset's recorded units.",
            "BMI means Body Mass Index.",
            "Smoking status follows the categories "
            "defined in the dataset.",
            "Work type and residence type are "
            "categorical features."
        ]
    }
}


def render_input_guidance(disease):
    info = DISEASE_GUIDANCE.get(disease)

    if info is None:
        st.warning(
            "Input guidance is not available "
            "for this disease."
        )
        return

    with st.expander(
        "Understanding These Health Indicators",
        expanded=False
    ):
        st.markdown(f"### {info['title']}")

        st.write(info["description"])

        for tip in info["tips"]:
            st.markdown(f"- {tip}")

        st.info(
            "Use demonstration data for this academic "
            "application. Predictions are not "
            "medical diagnoses."
        )
