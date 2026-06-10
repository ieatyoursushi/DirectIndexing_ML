"""Guard: the codebook schema must match the real lots.csv header exactly."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.codebook_schema import COLUMNS, EXPECTED_HEADER
from scripts.report_helpers import NUMERIC_FEATURES, repo_root


def test_schema_has_21_unique_columns():
    assert len(EXPECTED_HEADER) == 21
    assert len(set(EXPECTED_HEADER)) == 21


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
