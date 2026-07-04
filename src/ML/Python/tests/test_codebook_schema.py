"""Guard: the codebook schema must match the real lots.csv header exactly."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.codebook_schema import COLUMNS, EXPECTED_HEADER
from scripts.report_helpers import NUMERIC_FEATURES, repo_root


# Schema v3 (v0.25): 26 columns = 17 numeric features + Sector + Symbol/Timestep
# metadata + 6 labels (Y_Oracle, Y_Soft_GBM, Y_Soft_BT, Y_TaxValue, Y_Utility,
# Y_Oracle_GatedSpec).
def test_schema_has_26_unique_columns():
    assert len(EXPECTED_HEADER) == 26
    assert len(set(EXPECTED_HEADER)) == 26


def test_every_entry_fully_documented():
    for c in COLUMNS:
        for key in ("name", "dtype", "units", "role", "description", "encoding", "missing", "source"):
            assert c.get(key), f"{c.get('name', '?')} missing '{key}'"


def test_numeric_features_subset_of_schema():
    assert set(NUMERIC_FEATURES) <= set(EXPECTED_HEADER)


def test_matches_lots_csv_header():
    lots = repo_root(Path(__file__).parent) / "data" / "lots.csv"
    if not lots.exists():
        pytest.skip("data/lots.csv not present (run `dotnet run simulate`)")
    header = list(pd.read_csv(lots, nrows=0).columns)
    assert header == EXPECTED_HEADER
