# MediRisk AI
### Multi-Disease Risk Classification System Using Patient Data

**An Interactive Machine Learning Platform for Disease Classification, Dataset Exploration, Model Evaluation and Explainable AI**

**Developed by Team Triad Data Labs | Bahria University Karachi Campus**

---

## 1. Project Overview

MediRisk AI is a web-based machine learning application designed to demonstrate how supervised learning algorithms classify disease-related outcomes using publicly available patient datasets.

The system provides a centralised platform for exploring five health conditions through separate Decision Tree classification models.

Users can enter relevant patient measurements, explore model predictions, examine datasets, compare model performance and understand the decision-making process of trained classifiers.

The application also includes interactive visualisations, classification reports, experimental model comparisons and data-quality analysis.

**MediRisk AI is strictly an academic and educational project. It is not intended for medical diagnosis or clinical decision-making.**

---

## 2. Disease Classification Modules

MediRisk AI supports five independent disease classification modules.

| Disease | Dataset | Main Features |
|---|---|---|
| Diabetes | CDC Diabetes Health Indicators | BMI, age categories, blood pressure indicators, health and lifestyle information |
| Heart Disease | UCI Heart Disease (Cleveland) | Age, chest pain type, cholesterol, blood pressure, ECG and cardiac measurements |
| Chronic Kidney Disease | UCI Chronic Kidney Disease | Blood pressure, serum creatinine, albumin, haemoglobin and other laboratory measurements |
| Liver Disease | Indian Liver Patient Dataset | Bilirubin, liver enzymes, proteins, albumin and patient characteristics |
| Stroke | Stroke Prediction Dataset | Age, hypertension, heart disease history, glucose, BMI and lifestyle information |

Each module uses its own dataset, preprocessing steps, Decision Tree model and prediction form.

---

## 3. Key Features

### Interactive Home Dashboard

The application provides a central dashboard containing all five disease classification modules.

Users can navigate between health conditions and access their corresponding prediction interfaces.

### Disease Classification

Each disease module allows users to enter relevant measurements through interactive Streamlit forms.

The information is processed using the corresponding saved preprocessing pipeline and Decision Tree model.

The system displays the predicted dataset class and, when available, the model-estimated positive-class probability.

### Dataset Explorer

The Dataset Explorer provides:

- Dataset previews and feature information
- Number of records and columns
- Missing-value analysis
- Duplicate-record identification
- Target class distributions
- Feature histograms
- Correlation visualisations
- CSV download functionality

### Model Insights

The application provides visualisations and performance information for the trained classifiers, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, where available
- Confusion matrices
- Decision Tree diagrams
- Feature importance

### Advanced Model Comparison

Users can compare evaluation metrics across all five disease models using interactive charts and tables.

The dashboard highlights differences between model accuracy, precision, recall and F1-score.

### Actual vs Predicted Analysis

The system provides an interface for examining experimental test-set classifications.

Results are categorised into:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

Users can filter classification outcomes and export available analysis results.

### Model Experiments

The Model Experiments dashboard presents:

- Original versus experimental model results
- Decision Tree hyperparameter tuning
- Cross-validation performance
- Stroke and Liver model experiments
- Kidney model reliability analysis
- Training and testing performance comparisons

Experimental models are kept separate from the original application models unless properly validated and approved for integration.

### Explainable AI

MediRisk AI includes a Decision Tree explanation module.

For supported predictions, users can inspect the actual feature conditions and thresholds followed by the trained model.

These explanations describe the classifier's decision process rather than medical causes or clinical reasoning.

### Input Guidance

Disease-specific guidance panels help users understand the measurements and dataset categories required by each prediction form.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application and interface |
| Pandas | Dataset loading, cleaning and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Decision Tree training, preprocessing and evaluation |
| Matplotlib | Statistical and Decision Tree visualisations |
| Plotly | Interactive charts and dashboards |
| Joblib | Saving and loading trained models |
| Pytest | Automated Python testing |
| Git | Version control |
| GitHub | Collaborative development |
| GitHub Actions | Automated testing workflow |
| Streamlit Community Cloud | Intended web deployment platform |

---

## 5. Machine Learning Methodology

MediRisk AI uses **Decision Tree Classification**, a supervised machine learning technique.

A Decision Tree learns classification rules from labelled datasets by splitting records according to feature-based conditions.

### Step 1 — Dataset Collection

Five publicly available datasets were selected, with a separate dataset for each disease module.

### Step 2 — Data Inspection

Datasets are inspected for missing values, duplicated records, data types, feature characteristics and outcome class distributions.

### Step 3 — Data Preprocessing

Preprocessing includes:

- Handling missing values
- Converting numerical measurements
- Encoding categorical features
- Validating target labels
- Converting disease outcomes into binary categories

Numerical missing values are handled using median imputation, while categorical features are processed using imputation and one-hot encoding.

### Step 4 — Training and Testing

The datasets are divided into:

- **80% training data**
- **20% testing data**

Stratified splitting is used to preserve class proportions where appropriate.

Preprocessing transformations are fitted using training data and applied consistently to testing data.

### Step 5 — Model Training

Independent Decision Tree classifiers are trained for each disease.

Tree parameters such as maximum depth, minimum samples per leaf and class weighting are used to control model complexity.

### Step 6 — Model Evaluation

Models are evaluated using accuracy, precision, recall, F1-score and confusion matrices.

Evaluation focuses on overall classification performance and the identification of positive and negative outcomes.

### Step 7 — Model Experimentation

Additional experiments use cross-validation and hyperparameter search to investigate alternative model configurations.

The selection of experimental configurations is based on training-set validation, while held-out testing is used for final evaluation.

---

## 6. Project Architecture

```text
MediRisk AI
│
├── Streamlit Web Application
│   ├── Home Dashboard
│   ├── Disease Predictions
│   ├── Dataset Explorer
│   ├── Model Insights
│   ├── Model Comparison
│   ├── Actual vs Predicted
│   ├── Model Experiments
│   └── About Project
│
├── Dataset Management
│   ├── Data Loading
│   ├── Data Inspection
│   ├── Missing-Value Handling
│   └── Feature Encoding
│
├── Machine Learning
│   ├── Diabetes Decision Tree
│   ├── Heart Disease Decision Tree
│   ├── Kidney Disease Decision Tree
│   ├── Liver Disease Decision Tree
│   └── Stroke Decision Tree
│
├── Model Evaluation
│   ├── Classification Metrics
│   ├── Confusion Matrices
│   ├── Cross-Validation
│   └── Hyperparameter Experiments
│
└── Explainable AI
    ├── Decision Tree Visualisation
    ├── Feature Importance
    └── Prediction Decision Paths
```

### Prediction Workflow

```text
User Selects Disease
         |
         v
Enter Patient Measurements
         |
         v
Input Validation
         |
         v
Saved Preprocessing Pipeline
         |
         v
Disease-Specific Decision Tree
         |
         v
Predicted Dataset Class
         |
         v
Model Probability (if available)
         |
         v
Decision Path Explanation
         |
         v
Educational Disclaimer
```

---

## 7. Project Directory Structure

The main project is organised into the following folders and files:

```text
MediRisk-AI/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── style.css
│
├── datasets/
│   ├── diabetes.csv
│   ├── heart.data
│   ├── chronic.arff
│   ├── liver.csv
│   └── stroke.csv
│
├── models/
│   └── Trained model files (generated locally)
│
├── results/
│   ├── data_audit/
│   ├── model_review/
│   ├── predictions/
│   ├── tuning/
│   ├── kidney_audit/
│   ├── verification/
│   ├── decision_trees/
│   ├── feature_importance/
│   └── evaluation_results.csv
│
├── scripts/
│   ├── inspect_data.py
│   ├── kidney_check.py
│   ├── dataset_audit.py
│   ├── model_review.py
│   ├── export_predictions.py
│   ├── verify_models.py
│   ├── tune_stroke.py
│   ├── tune_liver.py
│   └── kidney_reliability_audit.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── prediction.py
│   ├── visualize_models.py
│   ├── dataset_explorer.py
│   ├── model_comparison.py
│   ├── prediction_analysis.py
│   ├── model_experiments.py
│   ├── model_explainer.py
│   └── input_guidance.py
│
├── tests/
│   ├── test_heart_preprocessing.py
│   ├── test_predictions.py
│   └── test_model_explainer.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Some model files and generated reports are created locally and may not be tracked in Git. The exact contents of the final repository depend on the merged development branches.

---

## 8. Installation and Setup

### Prerequisites

Install the following:

- Python with a version compatible with `requirements.txt`
- Git
- Visual Studio Code or another Python-compatible editor

### Step 1 — Clone the Repository

```bash
git clone https://github.com/wardha-netizen/MediRisk-AI.git
```

```bash
cd MediRisk-AI
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3 — Activate the Environment

On Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

On Windows using Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5 — Train the Models

If the required trained model files are not already available locally:

```bash
python -m src.train_models
```

The training script generates the trained classifiers and their saved preprocessing components.

### Step 6 — Generate Evaluation Results

```bash
python -m src.evaluate_models
```

Generate additional model visualisations using the project's visualisation module, where required.

### Step 7 — Launch the Application

```bash
python -m streamlit run app.py
```

The application should open at:

http://localhost:8501

---

## 9. Running Dataset Audits and Model Experiments

### Dataset Quality Audit

```bash
python -m scripts.dataset_audit
```

### Model Performance Review

```bash
python -m scripts.model_review
```

### Kidney Reliability Audit

```bash
python -m scripts.kidney_reliability_audit
```

### Stroke Model Tuning

```bash
python -m scripts.tune_stroke
```

### Liver Model Tuning

```bash
python -m scripts.tune_liver
```

### Model Verification

```bash
python -m scripts.verify_models
```

### Export Test Predictions

```bash
python -m scripts.export_predictions
```

Experimental results are stored separately from production models to avoid accidentally replacing the existing application classifiers.

---

## 10. Model Evaluation and Experimental Findings

Evaluation is conducted independently for each disease dataset.

The main classification metrics are:

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correctly classified test records |
| Precision | Correct positive predictions divided by all positive predictions |
| Recall | Correctly identified positive records divided by all actual positives |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Distribution of correct and incorrect classifications |
| ROC-AUC | Discrimination across classification thresholds, where available |

### Stroke Tuning Experiment

The Stroke Decision Tree was evaluated using GridSearchCV and five-fold stratified cross-validation.

The selected experimental model produced:

| Metric | Experimental Result |
|---|---:|
| Accuracy | 77.30% |
| Precision | 12.70% |
| Recall | 62.00% |
| F1-score | 21.09% |

The experiment did not demonstrate an improvement over the previously reported baseline metrics. Therefore, the experimental model was not selected to replace the original classifier.

### Liver Tuning Experiment

The experimental Liver Decision Tree produced:

| Metric | Experimental Result |
|---|---:|
| Accuracy | 70.09% |
| Precision | 73.08% |
| Recall | 91.57% |
| F1-score | 81.28% |

The experimental F1-score was higher than the previously reported baseline of 75.68%.

However, confirmation of identical evaluation conditions is required before concluding that the experimental classifier generalises better.

### Kidney Reliability Audit

The Kidney audit produced:

| Metric | Training | Testing |
|---|---:|---:|
| Accuracy | 98.75% | 98.75% |
| Precision | 100.00% | 100.00% |
| Recall | 98.00% | 98.00% |
| F1-score | 98.99% | 98.99% |

The audit also identified zero exact duplicate feature values shared across the training and testing subsets examined.

These results support the consistency of this particular experiment but do not establish that all forms of data leakage are absent.

**Important:** These are previously generated local experimental results. They should be reproduced and verified before being treated as final published model benchmarks.

---

## 11. Automated Testing

MediRisk AI includes automated tests developed using Pytest.

Tests cover:

- Disease dataset preprocessing
- Binary target validation
- Feature dimension consistency
- Decision Tree training and prediction
- Decision path explanation logic

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Check Python Syntax

```bash
python -m compileall -q app.py src scripts tests
```

The repository also includes a GitHub Actions workflow intended to run automated tests when code changes are pushed or submitted for review.

---

## 12. Model Explainability

MediRisk AI provides Decision Tree interpretation tools to help students understand classification behaviour.

These include:

- Decision Tree diagrams
- Feature importance visualisations
- Classification decision rules
- Model evaluation comparisons
- Prediction-specific decision paths

A decision path describes the sequence of feature-based conditions followed by the trained classifier.

**Model explanations describe computational behaviour, not medical causation.**

---

## 13. Responsible AI and Limitations

MediRisk AI has been developed for educational and academic purposes only.

The system has several important limitations:

- Predictions are not medical diagnoses.
- Models have not undergone clinical validation.
- Training datasets may contain missing or inconsistent information.
- Class imbalance may affect precision and recall.
- Historical datasets may not represent all populations.
- Model performance on new data may differ from testing results.
- High accuracy does not necessarily imply reliable positive-class classification.
- Model-estimated probabilities are not necessarily calibrated clinical risk probabilities.
- Decision Tree explanations do not establish causal relationships.
- The application is not intended to replace professional medical advice.

The project does not require personally identifiable patient records and should be demonstrated using public or synthetic data.

---

## 14. Project Team

**Team Name:** Triad Data Labs

**Institution:** Bahria University Karachi Campus

**Programme:** Bachelor of Science in Computer Science

| Team Member | Registration Number | Main Area of Contribution |
|---|---|---|
| Wardha Khalid | 02-134242-096 | Machine learning implementation, Decision Tree training and technical documentation |
| Maheen Fatima | 02-134242-011 | Dataset analysis, data-quality auditing, model evaluation, experimental tuning, testing and application enhancements |


All team members contribute to project testing, debugging, documentation, integration and presentation.

---

## 15. Future Enhancements

Potential future improvements include:

- Independent external dataset validation
- Additional supervised learning algorithms for comparison
- Model probability calibration
- More advanced explainability techniques
- Improved accessibility and responsive design
- Expanded automated testing
- Additional disease classification datasets
- More comprehensive model monitoring and reproducibility reporting

Any future clinical application would require appropriate validation, privacy protections and regulatory assessment.

---

## 16. Dataset Sources

The project uses publicly available datasets from:

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [UCI Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- [Indian Liver Patient Dataset](https://archive.ics.uci.edu/dataset/225/ilpd)
- [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Dataset ownership, attribution and usage conditions remain with their respective providers.

---

## 17. Project Status

**Current Stage:** Development, testing and final integration.

The project includes five disease classification modules, trained machine learning models, interactive dashboards, dataset analysis, evaluation tools, experimental model tuning and automated tests.

Final deployment readiness depends on reproducing model results, verifying all application features and ensuring saved model files are available to the deployment environment.

---

**MediRisk AI — Triad Data Labs**

*An educational exploration of machine learning, disease classification and responsible AI.*