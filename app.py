import streamlit as st
import pandas as pd
from pathlib import Path

from src.prediction import predict_disease, DISEASES
from src.data_loader import load_kidney
from src.dataset_explorer import render_dataset_explorer
from src.model_comparison import render_model_comparison
from src.prediction_analysis import render_prediction_analysis
from src.model_experiments import render_model_experiments
from src.prediction import load_model
from src.model_explainer import explain_decision_tree
# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MediRisk AI | Health Risk Insights",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ASSET_FOLDER = BASE_DIR / "assets"
RESULT_FOLDER = BASE_DIR / "results"


# ==========================================
# LOAD CUSTOM CSS
# ==========================================

css_path = ASSET_FOLDER / "style.css"

if css_path.exists():
    with open(css_path, encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# ==========================================
# DISEASE INFORMATION
# ==========================================

DISEASE_INFO = {
    "diabetes": {
        "icon": "🩸",
        "name": "Diabetes",
        "description": "Explore diabetes risk using health and lifestyle indicators.",
        "color": "#EDF3FA"
    },
    "heart": {
        "icon": "❤️",
        "name": "Heart Disease",
        "description": "Explore heart disease risk using cardiovascular indicators.",
        "color": "#EDF3FA"
    },
    "kidney": {
        "icon": "🫘",
        "name": "Kidney Disease",
        "description": "Explore kidney disease risk using laboratory indicators.",
        "color": "#EDF3FA"
    },
    "liver": {
        "icon": "🧪",
        "name": "Liver Disease",
        "description": "Explore liver disease risk using liver function indicators.",
        "color": "#EDF3FA"
    },
    "stroke": {
        "icon": "🧠",
        "name": "Stroke",
        "description": "Explore stroke risk using health and lifestyle indicators.",
        "color": "#E7F1FC"
    }
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def section_heading(eyebrow, heading, description):
    st.markdown(
        f"""
        <div class="section-label">{eyebrow}</div>
        <h1>{heading}</h1>
        <p style="color:#858196; font-size:1rem;">
            {description}
        </p>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_result(result):
    st.markdown("---")
    st.subheader("Your Prediction Results")

    is_positive = result["prediction"] == 1

    if is_positive:
        st.markdown(
            """
            <div class="result-card result-positive">
                <div class="result-title">⚠️ Disease Risk Detected</div>
                <p>
                    The model classified the submitted information
                    into the positive disease-risk category.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="result-card result-negative">
                <div class="result-title">✓ No Disease Risk Detected</div>
                <p>
                    The model classified the submitted information
                    into the negative disease-risk category.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if result["probability"] is not None:
        probability = result["probability"] * 100

        st.write("### Model-estimated positive-class probability")

        st.markdown(
            f"""
            <div class="result-number">{probability:.1f}%</div>
            """,
            unsafe_allow_html=True
        )

        st.progress(result["probability"])

        st.caption(
            "This is the model's estimated probability for its "
            "positive class, not a clinically validated personal "
            "probability of disease."
        )

    st.warning(
        "Educational use only. This prediction is not a medical "
        "diagnosis or a substitute for professional medical advice."
    )
   
    # ==========================================
    # EXPLAINABLE AI SECTION
    # ==========================================

    st.divider()

    st.subheader("How the Decision Tree Made Its Prediction")

    st.caption(
        "Explore the exact decision rules followed by "
        "the trained Decision Tree for this input."
    )

    explanation = result.get("explanation", [])

    if explanation:

        explanation_df = pd.DataFrame(explanation)

        explanation_df.index = range(
            1, len(explanation_df) + 1
        )

        explanation_df.index.name = "Step"

        st.dataframe(
            explanation_df,
            use_container_width=True
        )

        with st.expander("Understand the Decision Tree rules"):

            st.write(
                "Each row represents a decision made by "
                "the model. A feature value is compared "
                "against a learned threshold."
            )

            st.write(
                "The model follows either the left or "
                "right branch until it reaches a leaf "
                "node containing the predicted class."
            )

    else:

        if result.get("explanation_error"):

            st.info(
                "The prediction was generated, but "
                "the explanation could not be displayed."
            )

            st.caption(
                result["explanation_error"]
            )

        else:

            st.info(
                "The model did not produce any "
                "intermediate decision rules."
            )

    st.caption(
        "These rules describe how the machine learning "
        "model operates. They are not medical explanations "
        "or evidence of disease causation."
    )



def save_result(disease, result, inputs):
    """Save prediction and generate its Decision Tree explanation."""

    saved_result = result.copy()

    try:
        model_data = load_model(disease)

        model = model_data["model"]
        preprocessor = model_data["preprocessor"]
        feature_columns = model_data["feature_columns"]

        # Use the exact same input feature order as prediction.py
        ordered_inputs = {
            column: inputs[column]
            for column in feature_columns
        }

        explanation = explain_decision_tree(
            model=model,
            preprocessor=preprocessor,
            inputs=ordered_inputs
        )

        saved_result["explanation"] = explanation

    except Exception as error:
        saved_result["explanation"] = []
        saved_result["explanation_error"] = str(error)

    st.session_state["last_result"] = {
        "disease": disease,
        "result": saved_result
    }



def go_to_predictions(disease):
    st.session_state["selected_disease"] = disease
    st.session_state["navigation"] = "Predictions"


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 25px;">
            <div style="
                background:#1769D2;
                color:white;
                width:58px;
                height:58px;
                border-radius:18px;
                display:inline-flex;
                align-items:center;
                justify-content:center;
                font-size:27px;
                ">
                ♡
            </div>
            <h2 style="margin-bottom:0;">MediRisk AI</h2>
            <p style="font-size:12px;color:#858196;">
                Intelligent Health Risk Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### MENU")

    page = st.radio(
    "Navigate",
    [
        "Home",
        "Predictions",
        "Dataset Explorer",
        "Model Insights",
        "Model Comparison",
        "Actual vs Predicted",
        "Model Experiments",
        "About Project",
    ],
    label_visibility="collapsed",
    key="navigation",
)

    st.markdown("---")
    st.markdown("### MODEL COLLECTION")

    for info in DISEASE_INFO.values():
        st.markdown(
            f"""
            <div style="
                padding:9px 11px;
                margin-bottom:7px;
                border-radius:11px;
                background:#EAF4FF;
                border:1px solid #DCEBFA;
                font-size:13px;
                font-weight:600;
                color:#142B45;
            ">
                {info["icon"]} &nbsp; {info["name"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption("MediRisk AI · Academic Project")
    st.caption("Five Decision Tree Classifiers")


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="eyebrow">✦ INTELLIGENT HEALTH RISK INSIGHTS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="padding-top:25px;padding-bottom:15px;">
            <div class="hero-title">
                Understand your health.<br>
                <span class="hero-highlight">
                    Explore your risk.
                </span>
            </div>
            <p class="hero-description">
                An interactive machine learning platform exploring
                five major health risk categories through data-driven
                classification models.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Your health insights, all in one place")
    st.write("Choose a health category to explore its prediction form.")

    disease_items = list(DISEASE_INFO.items())

    for start in range(0, len(disease_items), 2):
        columns = st.columns(2)

        for col_idx, (disease_key, info) in enumerate(
            disease_items[start:start + 2]
        ):
            with columns[col_idx]:
                st.markdown(
                    f"""
                    <div class="disease-card">
                        <div class="disease-icon"
                             style="background:{info['color']};">
                            {info['icon']}
                        </div>
                        <div class="disease-name">
                            {info['name']}
                        </div>
                        <div class="disease-description">
                            {info['description']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(
                    f"Explore {info['name']} →",
                    key=f"home_{disease_key}",
                    use_container_width=True,
                    on_click=go_to_predictions,
                    args=(disease_key,)
                ):
                    pass

    st.markdown(
        """
        <div class="info-card">
            <h3>🩺 How does MediRisk AI work?</h3>
            <p>
                The application collects selected health indicators,
                processes them using the saved preprocessing pipelines,
                and generates a classification using trained
                Decision Tree models.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            MediRisk AI · Built for educational and research purposes
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Predictions":

    section_heading(
        "HEALTH RISK EXPLORER",
        "Disease Prediction",
        "Enter the requested health indicators to explore a model prediction."
    )

    disease_keys = list(DISEASES.keys())

    default_disease = st.session_state.get(
        "selected_disease", "diabetes"
    )

    if default_disease not in disease_keys:
        default_disease = disease_keys[0]

    selected_disease = st.selectbox(
        "Choose a disease model",
        disease_keys,
        index=disease_keys.index(default_disease),
        format_func=lambda x: DISEASE_INFO[x]["name"]
    )

    st.session_state["selected_disease"] = selected_disease

    info = DISEASE_INFO[selected_disease]

    st.markdown(
        f"""
        <div class="info-card">
            <h3>{info['icon']} {info['name']}</h3>
            <p>{info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------
    # DIABETES FORM
    # --------------------------------------

    if selected_disease == "diabetes":

        st.subheader("Personal Health Indicators")

        with st.form("diabetes_form"):

            col1, col2 = st.columns(2)

            with col1:
                bmi = st.number_input(
                    "BMI",
                    12.0, 98.0, 25.0,
                    help="Body Mass Index"
                )

                age = st.selectbox(
                    "Age Category",
                    list(range(1, 14)),
                    index=7,
                    format_func=lambda x: f"Category {x}"
                )

                sex = st.selectbox(
                    "Sex (dataset coding)",
                    [0, 1]
                )

                genhlth = st.selectbox(
                    "General Health",
                    [1, 2, 3, 4, 5],
                    format_func=lambda x: {
                        1: "Excellent",
                        2: "Very good",
                        3: "Good",
                        4: "Fair",
                        5: "Poor"
                    }[x]
                )

                menthlth = st.number_input(
                    "Mental Health (days, last 30 days)",
                    0, 30, 0
                )

                physhlth = st.number_input(
                    "Physical Health (days, last 30 days)",
                    0, 30, 0
                )

                education = st.selectbox(
                    "Education Level (dataset code)",
                    list(range(1, 7)),
                    index=3
                )

                income = st.selectbox(
                    "Income Category (dataset code)",
                    list(range(1, 9)),
                    index=4
                )

            with col2:
                highbp = st.selectbox("High Blood Pressure", [0, 1])
                highchol = st.selectbox("High Cholesterol", [0, 1])
                cholcheck = st.selectbox("Cholesterol Check", [0, 1])
                smoker = st.selectbox("Smoker", [0, 1])
                previous_stroke = st.selectbox("Previous Stroke", [0, 1])
                heart_attack = st.selectbox("Heart Disease or Attack", [0, 1])
                physical_activity = st.selectbox("Physical Activity", [0, 1])
                fruits = st.selectbox("Consumes Fruits", [0, 1])
                veggies = st.selectbox("Consumes Vegetables", [0, 1])
                heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])
                healthcare = st.selectbox("Has Healthcare Coverage", [0, 1])
                no_doc = st.selectbox("Unable to Visit Doctor Due to Cost", [0, 1])
                diffwalk = st.selectbox("Difficulty Walking", [0, 1])

            submitted = st.form_submit_button(
                "Analyze Diabetes Risk →",
                use_container_width=True
            )

        if submitted:
            inputs = {
                "HighBP": highbp,
                "HighChol": highchol,
                "CholCheck": cholcheck,
                "BMI": bmi,
                "Smoker": smoker,
                "Stroke": previous_stroke,
                "HeartDiseaseorAttack": heart_attack,
                "PhysActivity": physical_activity,
                "Fruits": fruits,
                "Veggies": veggies,
                "HvyAlcoholConsump": heavy_alcohol,
                "AnyHealthcare": healthcare,
                "NoDocbcCost": no_doc,
                "GenHlth": genhlth,
                "MentHlth": menthlth,
                "PhysHlth": physhlth,
                "DiffWalk": diffwalk,
                "Sex": sex,
                "Age": age,
                "Education": education,
                "Income": income
            }

            with st.spinner("Analyzing health indicators..."):
                try:
                    result = predict_disease("diabetes", inputs)
                    save_result("diabetes", result, inputs)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    # --------------------------------------
    # HEART DISEASE FORM
    # --------------------------------------

    elif selected_disease == "heart":

        st.subheader("Cardiovascular Indicators")

        with st.form("heart_form"):

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 45)
                sex = st.selectbox("Sex (dataset coding)", [0, 1])

                cp = st.selectbox(
                    "Chest Pain Type",
                    [1, 2, 3, 4]
                )

                trestbps = st.number_input(
                    "Resting Blood Pressure",
                    60, 250, 120
                )

                chol = st.number_input(
                    "Serum Cholesterol",
                    50, 700, 200
                )

                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    [0, 1]
                )

                restecg = st.selectbox(
                    "Resting ECG Result",
                    [0, 1, 2]
                )

            with col2:
                thalach = st.number_input(
                    "Maximum Heart Rate Achieved",
                    40, 250, 150
                )

                exang = st.selectbox(
                    "Exercise Induced Angina",
                    [0, 1]
                )

                oldpeak = st.number_input(
                    "ST Depression (Oldpeak)",
                    0.0, 10.0, 1.0,
                    step=0.1
                )

                slope = st.selectbox(
                    "ST Segment Slope",
                    [1, 2, 3]
                )

                ca = st.selectbox(
                    "Major Vessels (CA)",
                    [0, 1, 2, 3]
                )

                thal = st.selectbox(
                    "Thalassemia (Thal)",
                    [3, 6, 7]
                )

            submitted = st.form_submit_button(
                "Analyze Heart Disease Risk →",
                use_container_width=True
            )

        if submitted:
            inputs = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }

            with st.spinner("Analyzing cardiovascular indicators..."):
                try:
                    result = predict_disease("heart", inputs)
                    save_result("heart", result, inputs)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    # --------------------------------------
    # KIDNEY DISEASE FORM
    # --------------------------------------

    elif selected_disease == "kidney":

        st.subheader("Kidney Function Indicators")
        kidney_df = load_kidney()

        with st.form("kidney_form"):

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 45)
                bp = st.number_input("Blood Pressure", 40, 250, 80)
                sg = st.selectbox(
                    "Specific Gravity",
                    [1.005, 1.010, 1.015, 1.020, 1.025],
                    index=3
                )
                al = st.selectbox("Albumin", list(range(6)))
                su = st.selectbox("Sugar", list(range(6)))

                rbc = st.selectbox(
                    "Red Blood Cells",
                    ["normal", "abnormal"]
                )

                pc = st.selectbox(
                    "Pus Cell",
                    ["normal", "abnormal"]
                )

                pcc = st.selectbox(
                    "Pus Cell Clumps",
                    ["present", "notpresent"]
                )

                ba = st.selectbox(
                    "Bacteria",
                    ["present", "notpresent"]
                )

                bgr = st.number_input(
                    "Blood Glucose Random",
                    40.0, 600.0, 120.0
                )

                bu = st.number_input(
                    "Blood Urea",
                    1.0, 400.0, 40.0
                )

                sc = st.number_input(
                    "Serum Creatinine",
                    0.1, 80.0, 1.2
                )

            with col2:
                sod = st.number_input(
                    "Sodium",
                    50.0, 200.0, 140.0
                )

                pot = st.number_input(
                    "Potassium",
                    1.0, 15.0, 4.5
                )

                hemo = st.number_input(
                    "Hemoglobin",
                    1.0, 25.0, 13.0
                )

                pcv = st.number_input(
                    "Packed Cell Volume",
                    1.0, 70.0, 40.0
                )

                wbcc = st.number_input(
                    "White Blood Cell Count",
                    100.0, 30000.0, 8000.0
                )

                rbcc = st.number_input(
                    "Red Blood Cell Count",
                    1.0, 10.0, 4.5
                )

                def kidney_choice(column, label):
                    if kidney_df is not None and column in kidney_df.columns:
                        values = (
                            kidney_df[column]
                            .dropna()
                            .drop_duplicates()
                            .tolist()
                        )
                    else:
                        values = ["normal", "abnormal"]
                    return st.selectbox(
                        label,
                        values,
                        key=f"kidney_{column}",
                        format_func=lambda x: str(x).strip()
                    )

                htn = kidney_choice("htn", "Hypertension")
                dm = kidney_choice("dm", "Diabetes Mellitus")
                cad = kidney_choice("cad", "Coronary Artery Disease")
                appet = kidney_choice("appet", "Appetite")
                pe = kidney_choice("pe", "Pedal Edema")
                ane = kidney_choice("ane", "Anemia")

            submitted = st.form_submit_button(
                "Analyze Kidney Disease Risk →",
                use_container_width=True
            )

        if submitted:
            inputs = {
                "age": age,
                "bp": bp,
                "sg": sg,
                "al": al,
                "su": su,
                "rbc": rbc,
                "pc": pc,
                "pcc": pcc,
                "ba": ba,
                "bgr": bgr,
                "bu": bu,
                "sc": sc,
                "sod": sod,
                "pot": pot,
                "hemo": hemo,
                "pcv": pcv,
                "wbcc": wbcc,
                "rbcc": rbcc,
                "htn": htn,
                "dm": dm,
                "cad": cad,
                "appet": appet,
                "pe": pe,
                "ane": ane
            }

            with st.spinner("Analyzing kidney indicators..."):
                try:
                    result = predict_disease("kidney", inputs)
                    save_result("kidney", result, inputs)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    # --------------------------------------
    # LIVER DISEASE FORM
    # --------------------------------------

    elif selected_disease == "liver":

        st.subheader("Liver Function Indicators")

        with st.form("liver_form"):

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 40)

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

                total_bilirubin = st.number_input(
                    "Total Bilirubin",
                    0.0, 100.0, 1.0
                )

                direct_bilirubin = st.number_input(
                    "Direct Bilirubin",
                    0.0, 100.0, 0.3
                )

                alkaline = st.number_input(
                    "Alkaline Phosphotase",
                    1, 5000, 200
                )

            with col2:
                alt = st.number_input(
                    "Alamine Aminotransferase (ALT)",
                    1, 5000, 30
                )

                ast = st.number_input(
                    "Aspartate Aminotransferase (AST)",
                    1, 5000, 35
                )

                proteins = st.number_input(
                    "Total Proteins",
                    0.0, 20.0, 6.5
                )

                albumin = st.number_input(
                    "Albumin",
                    0.0, 15.0, 3.5
                )

                ratio = st.number_input(
                    "Albumin and Globulin Ratio",
                    0.0, 10.0, 1.0
                )

            submitted = st.form_submit_button(
                "Analyze Liver Disease Risk →",
                use_container_width=True
            )

        if submitted:
            inputs = {
                "Age": age,
                "Gender": gender,
                "Total_Bilirubin": total_bilirubin,
                "Direct_Bilirubin": direct_bilirubin,
                "Alkaline_Phosphotase": alkaline,
                "Alamine_Aminotransferase": alt,
                "Aspartate_Aminotransferase": ast,
                "Total_Protiens": proteins,
                "Albumin": albumin,
                "Albumin_and_Globulin_Ratio": ratio
            }

            with st.spinner("Analyzing liver indicators..."):
                try:
                    result = predict_disease("liver", inputs)
                    save_result("liver", result, inputs)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    # --------------------------------------
    # STROKE FORM
    # --------------------------------------

    elif selected_disease == "stroke":

        st.subheader("Stroke Risk Indicators")

        with st.form("stroke_form"):

            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other"]
                )

                age = st.number_input(
                    "Age",
                    1.0, 120.0, 40.0
                )

                hypertension = st.selectbox(
                    "Hypertension",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )

                heart_disease = st.selectbox(
                    "Heart Disease",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )

                married = st.selectbox(
                    "Ever Married",
                    ["Yes", "No"]
                )

                work_type = st.selectbox(
                    "Work Type",
                    [
                        "Private",
                        "Self-employed",
                        "Govt_job",
                        "children",
                        "Never_worked"
                    ]
                )

            with col2:
                residence = st.selectbox(
                    "Residence Type",
                    ["Urban", "Rural"]
                )

                glucose = st.number_input(
                    "Average Glucose Level",
                    40.0, 300.0, 100.0
                )

                bmi = st.number_input(
                    "BMI",
                    10.0, 100.0, 25.0
                )

                smoking = st.selectbox(
                    "Smoking Status",
                    [
                        "formerly smoked",
                        "never smoked",
                        "smokes",
                        "Unknown"
                    ]
                )

            submitted = st.form_submit_button(
                "Analyze Stroke Risk →",
                use_container_width=True
            )

        if submitted:
            inputs = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": married,
                "work_type": work_type,
                "Residence_type": residence,
                "avg_glucose_level": glucose,
                "bmi": bmi,
                "smoking_status": smoking
            }

            with st.spinner("Analyzing stroke indicators..."):
                try:
                    result = predict_disease("stroke", inputs)
                    save_result("stroke", result, inputs)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    # --------------------------------------
    # DISPLAY SAVED RESULT
    # --------------------------------------

    previous = st.session_state.get("last_result")

    if previous and previous["disease"] == selected_disease:
        show_result(previous["result"])


# ==========================================
# MODEL INSIGHTS PAGE
# ==========================================
elif page == "Dataset Explorer":
    render_dataset_explorer()

elif page == "Model Insights":

    section_heading(
        "MODEL PERFORMANCE",
        "Behind the Predictions",
        "Explore the evaluation results and visualizations of the five Decision Tree classifiers."
    )

    metrics_path = RESULT_FOLDER / "evaluation_results.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)

        st.subheader("Model Performance Comparison")

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Metrics are based on the saved evaluation results. "
            "Performance may differ on new data."
        )
    else:
        st.info("Evaluation results have not been generated yet.")

    st.markdown("---")
    st.subheader("Decision Tree Visualizations")

    tree_folder = RESULT_FOLDER / "decision_trees"

    for disease in DISEASES:
        image_path = tree_folder / f"{disease}_decision_tree.png"

        if image_path.exists():
            with st.expander(
                f"{DISEASE_INFO[disease]['name']} — Decision Tree"
            ):
                st.image(
                    str(image_path),
                    use_container_width=True
                )

    st.markdown("---")
    st.subheader("Feature Importance")

    importance_folder = RESULT_FOLDER / "feature_importance"

    for disease in DISEASES:
        image_path = (
            importance_folder / f"{disease}_feature_importance.png"
        )

        if image_path.exists():
            with st.expander(
                f"{DISEASE_INFO[disease]['name']} — Feature Importance"
            ):
                st.image(
                    str(image_path),
                    use_container_width=True
                )

    st.markdown("---")
    st.subheader("Confusion Matrices")

    for disease in DISEASES:
        image_path = (
            RESULT_FOLDER / f"{disease}_confusion_matrix.png"
        )

        if image_path.exists():
            with st.expander(
                f"{DISEASE_INFO[disease]['name']} — Confusion Matrix"
            ):
                st.image(
                    str(image_path),
                    use_container_width=True
                )


# ==========================================
# ABOUT PAGE
# ==========================================
elif page == "Model Comparison":
    render_model_comparison()
elif page == "Actual vs Predicted":
    render_prediction_analysis()
elif page == "Model Experiments":
    render_model_experiments()
elif page == "About Project":

    section_heading(
        "ABOUT THE PROJECT",
        "Meet MediRisk AI",
        "An academic machine learning project focused on exploring health risk classification."
    )

    st.markdown(
        """
        <div class="info-card">
            <h2>🩺 Our Mission</h2>
            <p>
                MediRisk AI demonstrates how machine learning can
                process health-related datasets and classify
                patterns associated with five disease categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.subheader("Five Machine Learning Models")

    for info in DISEASE_INFO.values():
        st.markdown(
            f"""
            <div class="disease-card" style="margin-bottom:12px;">
                <div class="disease-name">
                    {info['icon']} &nbsp; {info['name']}
                </div>
                <div class="disease-description">
                    {info['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Language", "Python")

    with col2:
        metric_card("ML Algorithm", "Decision Tree")

    with col3:
        metric_card("Interface", "Streamlit")

    st.markdown("---")
    st.subheader("Important Disclaimer")

    st.warning(
        """
        MediRisk AI is an educational and academic project.
        Its predictions are generated by machine learning models
        and are not medical diagnoses.

        The system has not been clinically validated and should
        not be used to make healthcare decisions. Consult a
        qualified healthcare professional for medical advice.
        """
    )


# ==========================================
# APPLICATION FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        <b>MediRisk AI</b> · Intelligent Health Risk Insights<br>
        Educational Machine Learning Project · Not a Diagnostic Tool
    </div>
    """,
    unsafe_allow_html=True
)