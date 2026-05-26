"""Feature preprocessing — ColumnTransformer used by every supervised pipeline.

Standardises numeric features (LR is sensitive to scale via the L2 penalty) and
one-hot encodes the Sector categorical. The same recipe applies to PCA (which
also requires standardised inputs to avoid being dominated by high-variance features).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Numeric → StandardScaler;  Sector → OneHotEncoder(drop='first')."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def numeric_only_preprocessor() -> ColumnTransformer:
    """Standardise numeric features only — used by PCA where dummy variables are excluded."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), NUMERIC_FEATURES)],
        remainder="drop",
    )
