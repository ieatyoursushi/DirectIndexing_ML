"""Guards: the codebook schema must match the C# schema and the real lots.csv header."""
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.codebook_schema import COLUMNS, EXPECTED_HEADER, NUMERIC_FEATURES, repo_root


# Schema v5: 27 columns = 19 numeric features + Sector + Symbol/Timestep metadata
# + 5 labels (Y_Oracle, Y_Soft_GBM, Y_Soft_BT, Y_TaxValue, Y_Utility). v4 dropped
# the retired Y_Oracle_GatedSpec spectator (pre-v0.3 downsizing).
def test_schema_has_27_unique_columns():
    assert len(EXPECTED_HEADER) == 27
    assert len(set(EXPECTED_HEADER)) == 27


def test_every_entry_fully_documented():
    for c in COLUMNS:
        for key in ("name", "dtype", "units", "role", "description", "encoding", "missing", "source"):
            assert c.get(key), f"{c.get('name', '?')} missing '{key}'"


def test_numeric_features_subset_of_schema():
    assert set(NUMERIC_FEATURES) <= set(EXPECTED_HEADER)
    assert len(NUMERIC_FEATURES) == 19


def test_numeric_features_match_csharp_featurelists():
    """Cross-language drift check: the Python feature block must equal
    FeatureLists.NumericFeatures in C#, element for element and in order."""
    cs = (repo_root(Path(__file__).parent)
          / "src" / "ML" / "CSharp" / "MLNet" / "Schema" / "FeatureLists.cs").read_text()
    block = re.search(r"NumericFeatures\s*=\s*\{(.*?)\};", cs, re.S).group(1)
    assert re.findall(r'"([^"]+)"', block) == NUMERIC_FEATURES


def test_header_matches_csharp_exporter():
    """The exported CSV header (SimulationExporter.Header) must equal the schema."""
    cs = (repo_root(Path(__file__).parent) / "src" / "Export" / "SimulationExporter.cs").read_text()
    block = re.search(r"Header\s*=(.*?);", cs, re.S).group(1)
    header = "".join(re.findall(r'"([^"]*)"', block)).split(",")
    assert header == EXPECTED_HEADER


def test_matches_lots_csv_header():
    lots = repo_root(Path(__file__).parent) / "data" / "lots.csv"
    if not lots.exists():
        pytest.skip("data/lots.csv not present (run `dotnet run simulate`)")
    header = list(pd.read_csv(lots, nrows=0).columns)
    assert header == EXPECTED_HEADER
