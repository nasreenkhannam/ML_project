"""Dataset loading and cleaning utilities."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> pd.DataFrame:
    """Load the sklearn Breast Cancer Wisconsin dataset into a DataFrame."""
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"target": "target"})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicates, missing values, and extreme numeric outliers.

    Missing numeric values use the median. Duplicate rows are removed.
    Outliers are clipped to the IQR bounds rather than deleting observations.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_cols = cleaned.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    feature_cols = [c for c in numeric_cols if c != "target"]
    for col in feature_cols:
        q1, q3 = cleaned[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            cleaned[col] = cleaned[col].clip(lower, upper)
    return cleaned


def save_clean_data(df: pd.DataFrame, output_path: str = "data/cleaned_data.csv") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    df = clean_data(load_data())
    save_clean_data(df)
    print(f"Saved cleaned dataset: {df.shape}")
