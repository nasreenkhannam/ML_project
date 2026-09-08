"""Load the saved model and predict on new patient measurements."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(model, data):
    """Return class predictions for a DataFrame containing the 30 feature columns."""
    return model.predict(data)


def predict_csv(input_path, output_path="predictions.csv"):
    data = pd.read_csv(input_path)
    model = load_model()
    predictions = predict(model, data)
    result = data.copy()
    result["prediction"] = predictions
    result["prediction_label"] = result["prediction"].map({0: "malignant", 1: "benign"})
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict breast cancer class from a CSV file.")
    parser.add_argument("input_csv", help="CSV containing the model's feature columns")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()
    print(predict_csv(args.input_csv, args.output).tail())
