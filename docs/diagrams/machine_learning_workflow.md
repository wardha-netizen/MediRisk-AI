
flowchart TD
    A[Collect Public Datasets] --> B[Inspect Data]
    B --> C[Clean Invalid and Missing Values]
    C --> D[Validate Target Labels]
    D --> E[Stratified Train Test Split]

    E --> F[Training Dataset]
    E --> G[Testing Dataset]

    F --> H[Fit Preprocessing]
    H --> I[Train Decision Tree]

    I --> J[Cross Validation and Tuning]
    J --> K[Selected Model]

    G --> L[Apply Saved Preprocessing]
    K --> M[Generate Predictions]
    L --> M

    M --> N[Accuracy Precision Recall F1]
    M --> O[Confusion Matrix]
    M --> P[Actual vs Predicted Analysis]

    K --> Q[Save Model and Preprocessor]
    Q --> R[Streamlit Application]
