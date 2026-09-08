"""Train, compare, evaluate, and save classification models."""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from data import load_data, clean_data, save_clean_data
from features import split_data, build_preprocessor
from evaluate import evaluate_model

RANDOM_STATE = 42


def make_models():
    preprocessor = build_preprocessor(k_features=15)
    return {
        "Baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Logistic Regression": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def create_eda(df, output_dir="reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Target Distribution")
    plt.xlabel("Target (0 = malignant, 1 = benign)")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "target_distribution.png", dpi=150)
    plt.close()

    corr = df.drop(columns=["target"]).corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "correlation_heatmap.png", dpi=150)
    plt.close()


def main():
    df = clean_data(load_data())
    save_clean_data(df)
    create_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)

    results = []
    trained = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)

    # Save the best model by F1 score for inference on new observations.
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    Path("models/model_name.txt").write_text(best_name, encoding="utf-8")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved best model: {best_name} -> models/best_model.joblib")


if __name__ == "__main__":
    main()
