
flowchart TD
    A[User] --> B[Streamlit Web Application]

    B --> C[Home Dashboard]
    B --> D[Disease Prediction Forms]
    B --> E[Dataset Explorer]
    B --> F[Model Insights and Comparison]
    B --> G[Model Experiments]

    D --> H[Input Validation]
    H --> I[Saved Preprocessing Pipelines]
    I --> J[Decision Tree Classifiers]

    J --> K[Predicted Dataset Class]
    J --> L[Explainable AI]
    J --> M[Model Probability]

    N[Public Disease Datasets] --> O[Data Processing]
    O --> P[Model Training]
    P --> Q[Evaluation]
    Q --> R[Saved Models and Reports]

    R --> I
    R --> F
    R --> G
    N --> E

    K --> S[Results Display]
    L --> S
    M --> S
