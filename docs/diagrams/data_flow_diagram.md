
flowchart LR
    A[User] -->|Select disease| B[Prediction Form]
    A -->|Enter measurements| B

    B -->|Input data| C[Input Validation]
    C -->|Validated data| D[Preprocessing]

    E[(Saved Model Files)] --> F[Decision Tree Prediction]
    D --> F

    F -->|Predicted class| G[Results Display]
    F -->|Estimated probability| G
    F -->|Decision path| H[Explainable AI]
    H --> G

    I[(Public Datasets)] --> J[Model Training]
    J --> E

    G -->|Classification explanation| A
