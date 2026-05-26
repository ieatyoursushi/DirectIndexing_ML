"""Target derivation — single source of truth for label columns.

Two supported targets:
  - soft_bt : binary indicator (Y_Soft_BT > cutpoint)  — PRIMARY learning task
  - oracle  : Y_Oracle as-is (already 0/1)             — SANITY BASELINE

Y_Soft_BT is the meaningful prediction target. It's the fraction of the next 30 actual
trading days the oracle would fire, computed from real forward prices. Thresholding it
gives a binary "will the oracle fire at all in the next 30 days?" task that is genuinely
uncertain given current features.

Y_Oracle is a deterministic Boolean function of the features (the four-gate oracle in
OracleBoundary.Label). Approximating it provides no genuine learning — a logistic
regression should achieve ROC AUC ≈ 1.0 on it. We keep it as a pipeline sanity check.
"""

from __future__ import annotations

import pandas as pd


def binary_softbt(df: pd.DataFrame, cutpoint: float = 0.0) -> pd.Series:
    """Y_Soft_BT > cutpoint → binary {0, 1}. NaN rows dropped by caller via .dropna().

    Cutpoint 0.0 means "at least one fire in the next 30 days" (~31% positive rate).
    Higher cutpoints (e.g. 0.1 ≈ "at least 3 fires in 30 days") tighten the positive class.
    """
    return (df["Y_Soft_BT"] > cutpoint).astype(int)


def binary_oracle(df: pd.DataFrame) -> pd.Series:
    """Y_Oracle is already 0/1; just pass it through with a clean dtype."""
    return df["Y_Oracle"].astype(int)


def get_target(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (df_filtered, y) for the requested target.

    Drops rows where the target is NaN. For Y_Soft_BT, NaN occurs in the last ~30
    rows per lot (no forward window available). For Y_Oracle, no NaNs are expected.
    """
    name = name.lower()
    if name in ("soft_bt", "softbt", "soft"):
        y = binary_softbt(df, cutpoint=0.0)
        mask = df["Y_Soft_BT"].notna()
    elif name in ("oracle", "y_oracle"):
        y = binary_oracle(df)
        mask = df["Y_Oracle"].notna()
    else:
        raise ValueError(f"Unknown target: {name!r}. Expected 'soft_bt' or 'oracle'.")

    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[targets] {name}: dropping {dropped:,} rows with NaN target "
              f"({dropped / len(df):.1%} of total)")
    df_clean = df.loc[mask].reset_index(drop=True)
    y_clean = y.loc[mask].reset_index(drop=True)
    pos = int(y_clean.sum())
    print(f"[targets] {name}: {len(y_clean):,} rows  |  positives = {pos:,} ({pos / len(y_clean):.2%})")
    return df_clean, y_clean
