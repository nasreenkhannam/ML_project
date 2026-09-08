"""Model evaluation helpers."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(name, model, X_test, y_test, output_dir="reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\n{name}\n" + classification_report(y_test, predictions, zero_division=0))
    return metrics
