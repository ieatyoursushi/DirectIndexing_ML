"""The math↔code spine: the live repo must be in sync, and the checker's parsers must be right."""
from pathlib import Path

from scripts import check_math_sync as cms


def test_repository_is_in_sync():
    rep = cms.check()
    assert rep.errors == [], "\n".join(rep.errors)


def test_split_cells_ignores_pipes_inside_backticks():
    assert cms.split_cells("| `a|b` | x | `C.d` |") == ["`a|b`", "x", "`C.d`"]


def test_members_in_reads_semicolon_lists():
    cell = "`OracleConfig.WashSaleDays; OracleBoundary.WashSaleDays`"
    assert cms.members_in(cell) == [("OracleConfig", "WashSaleDays"), ("OracleBoundary", "WashSaleDays")]


def test_as_number_strips_csharp_suffixes():
    assert cms.as_number("90_000m") == 90000.0
    assert cms.as_number("252f") == 252.0
    assert cms.as_number("0.20m") == 0.2
    assert cms.as_number("true") is None


def test_declared_value_reads_all_three_declaration_styles(tmp_path: Path):
    (tmp_path / "X.cs").write_text(
        "public sealed record Cfg {\n"
        "  public const decimal A = 0.37m;\n"
        "  public decimal B { get; init; } = 90_000m;\n"
        "  public static int C { get; set; } = 30;\n"
        "}\n")
    idx = cms.type_index(tmp_path)
    assert cms.declared_value(idx, "Cfg", "A") == "0.37m"
    assert cms.declared_value(idx, "Cfg", "B") == "90_000m"
    assert cms.declared_value(idx, "Cfg", "C") == "30"
