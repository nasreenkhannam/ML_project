# Breast Cancer Classification — Interview-Ready ML Project

A clean, modular machine learning project that predicts whether a breast tumor is **malignant (0)** or **benign (1)** using the scikit-learn Breast Cancer Wisconsin Diagnostic dataset.

## What this project demonstrates

- Data loading and cleaning
- Duplicate, missing-value, and IQR-based outlier handling
- Exploratory Data Analysis with Matplotlib/Seaborn
- Feature selection with `SelectKBest`
- Feature scaling with `StandardScaler`
- Stratified train/test split
- Majority-class baseline
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Accuracy, precision, recall, F1, and confusion matrix evaluation
- Best-model persistence with Joblib
- Reusable prediction script for new CSV data

## Project structure

```text
ML_project/
├── data/                         # Generated cleaned dataset
├── models/                       # Generated trained model
├── reports/                      # Generated EDA plots and metrics
├── src/
│   ├── __init__.py
│   ├── data.py                   # Loading + cleaning
│   ├── features.py               # Split + preprocessing
│   ├── evaluate.py               # Metrics + confusion matrices
│   ├── train.py                  # Training + model comparison
│   └── predict.py                # Inference on new CSV data
├── requirements.txt
└── README.md
```

## Dataset and target

The project uses `sklearn.datasets.load_breast_cancer`, so no external download is required. The target is binary: `0 = malignant`, `1 = benign`.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train and evaluate

From the repository root:

```bash
cd src
python train.py
```

This generates:

- `data/cleaned_data.csv`
- `reports/target_distribution.png`
- `reports/correlation_heatmap.png`
- confusion-matrix plots
- `reports/model_comparison.csv`
- `models/best_model.joblib`
- `models/model_name.txt`

The best model is selected automatically using **F1 score**.

## Make predictions on new data

Create a CSV containing the same feature columns used by the dataset, then run:

```bash
python src/predict.py path/to/new_data.csv --output predictions.csv
```

The output contains the original columns plus `prediction` and `prediction_label`.

## Interview talking points

**Why F1?** It balances precision and recall and is more informative than accuracy alone when the cost of false positives and false negatives matters.

**Why scale?** Logistic Regression is sensitive to feature magnitude, while tree-based models generally are not. The preprocessing pipeline keeps the workflow consistent and prevents data leakage because transformations are fitted only on training data.

**Why compare models?** A simple baseline establishes a minimum reference point, while linear and tree-based models provide different modeling assumptions. The final model is chosen using held-out test performance rather than model name.

## Reproducibility

A fixed random seed (`42`) is used for splitting and model training where supported. The saved model is created by running the training pipeline, rather than storing generated binary artifacts in source control.
