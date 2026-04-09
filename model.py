"""model.py - Train a RandomForest to predict 5-day price direction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def add_target(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Add a binary target column:
    1 if closing price `forward_days` days in the future > today's close
    0 otherwise.
    """
    df = df.copy()
    df["Target"] = (df["close"].shift(-forward_days) > df["close"]).astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows with NaN and separate features (X) from target (y).
    """
    clean = df.dropna()
    feature_cols = [c for c in clean.columns if c not in ("close", "volume", "open", "high", "low", "Target")]
    X = clean[feature_cols]
    y = clean["Target"]
    return X, y


def train_random_forest(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, float, pd.Series]:
    """
    Split chronologically: first 80% train, last 20% test.
    Train a RandomForestClassifier, print test accuracy
    and feature importances ranked highest to lowest.

    Returns
    -------
    model : RandomForestClassifier
    accuracy : float
    importances : pd.Series ranked by importance descending
    """
    n = len(X)
    split = int(n * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Test Accuracy: {accuracy:.4f}")
    print("\nFeature Importances (ranked):")

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")

    return model, accuracy, importances


def predict_direction(model: RandomForestClassifier, X: pd.DataFrame) -> int:
    """Return the predicted class (0 or 1) for the last row."""
    last_row = X.iloc[[-1]]
    return int(model.predict(last_row)[0])
