"""I/O helpers — single source of truth for reads and writes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


# Column groups — mirror the LotStateVector schema produced by C# SimulationExporter
NUMERIC_FEATURES = [
    # Lot-level
    "L", "H", "S", "B", "W", "K",
    # Portfolio-level
    "G_YTD", "Sigma_TE", "WashClock",
    # Asset-level
    "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
    # Derived
    "TaxAlpha", "DaysToYE",
]

CATEGORICAL_FEATURES = ["Sector"]

METADATA_COLS = ["Symbol", "Timestep"]

LABEL_COLS = ["Y_Oracle", "Y_Soft_GBM", "Y_Soft_BT"]


def load_lots(path: str | Path = "../../../data/lots.csv") -> pd.DataFrame:
    """Load lots.csv. NaN handled by pandas default (empty strings → NaN)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"lots.csv not found at {path}. "
                                f"Run `dotnet run --project src -- simulate` first.")
    df = pd.read_csv(path)
    print(f"[io] Loaded {len(df):,} rows × {len(df.columns)} columns from {path}")
    return df


def save_json(obj, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    print(f"[io] Wrote {path}")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[io] Wrote {path}  ({len(df)} rows × {len(df.columns)} cols)")


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[io] Wrote {path}")


def _json_default(o):
    """Fallback for numpy types when JSON-serialising."""
    import numpy as np
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
