"""Feature engineering and preprocessing."""
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(df, target_col="target", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(k_features=15):
    """Select useful numeric features and scale them for linear models."""
    return Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("scaler", StandardScaler()),
    ])
