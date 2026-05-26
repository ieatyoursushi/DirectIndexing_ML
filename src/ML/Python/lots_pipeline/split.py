"""Stratified train/test split.

Stratification is critical because Y_Oracle is ~0.8% positive — a naive split could
leave the test fold with single-digit positives. Even for the more balanced Y_Soft_BT
target (~31% positive), stratification produces more stable CV estimates.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .io import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Slice the predictor columns from a lots DataFrame.

    Includes numeric features and the Sector categorical. Drops metadata, labels.
    NaN Sector cells are filled with "Unknown" so OneHotEncoder doesn't choke —
    relevant for the current dataset where constituents.json was bootstrapped
    without real sector metadata.
    """
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in lots DataFrame: {missing}")
    X = df[cols].copy()
    X["Sector"] = X["Sector"].fillna("Unknown").astype(str)
    # Fill rare numeric NaN (e.g. DeltaMA200 at start of window) with column median
    for c in NUMERIC_FEATURES:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X


def train_test(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. Returns (X_train, X_test, y_train, y_test)."""
    X = feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(f"[split] train={len(X_train):,}  test={len(X_test):,}  "
          f"train_pos_rate={y_train.mean():.2%}  test_pos_rate={y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test
