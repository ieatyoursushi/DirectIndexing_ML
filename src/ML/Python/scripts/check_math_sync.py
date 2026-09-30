"""docs-check: keep the math ↔ code ↔ test spine in sync.

Reads DataMemo/spec/SymbolTable.md (the typed symbol table) and FAILS (exit 1) when:

  1. anchors      — a `[math:id]` tag in src/**/*.cs has no SymbolTable row;
  2. coverage     — a live row (§A, §C–§G) with a code member has no `[math:id]` tag;
  3. members      — a `Class.Member` named in a code/test cell no longer exists;
  4. constants    — a §K value differs from its C# declaration;
  5. schema       — §B's coordinate ids differ from NUMERIC_FEATURES (schema order);
  6. table shape  — a row has the wrong number of cells (an unescaped `|` in LaTeX);
  7. links        — a relative markdown link anywhere in the repo does not resolve.

This generalizes the codebook header-drift assert: drift fails loudly instead of silently.
Run: `dotnet run --project src -- docs-check`, or `uv run python -m scripts.check_math_sync`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts.codebook_schema import NUMERIC_FEATURES, repo_root

LIVE_SECTIONS = {"A", "C", "D", "E", "F", "G"}
COLUMNS = {"J": 3, "K": 6}          # every other lettered section: 8 columns
FROZEN_MD = ("src/Export/report/", "src/Export/diagrams/")
ANCHOR = re.compile(r"\[math:([A-Za-z0-9_.]+)\]")


@dataclass
class Row:
    section: str
    planned: bool
    cells: list[str]
    line: int

    @property
    def id(self) -> str:
        return self.cells[0].strip().strip("`")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.errors.append(msg)


def split_cells(line: str) -> list[str]:
    """Split a markdown table row on `|`, ignoring pipes inside backtick spans."""
    cells, buf, in_code = [], [], False
    for ch in line.strip()[1:-1]:
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf))
    return [c.strip() for c in cells]


def parse_table(md: str) -> list[Row]:
    rows, section, planned = [], None, False
    for n, line in enumerate(md.splitlines(), 1):
        m = re.match(r"## ([A-K])\.", line)
        if m:
            section, planned = m.group(1), "planned" in line.lower()
        elif line.startswith("## "):
            section = None
        elif section and line.startswith("| `"):
            rows.append(Row(section, planned, split_cells(line), n))
    return rows


def members_in(cell: str) -> list[tuple[str, str]]:
    """`Class.Member` tokens in a cell (backticked, possibly `A.b; C.d` inside one span)."""
    out = []
    for span in re.findall(r"`([^`]*)`", cell):
        for part in span.split(";"):
            m = re.fullmatch(r"\s*([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*", part)
            if m:
                out.append((m.group(1), m.group(2)))
    return out


def type_index(src: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    decl = re.compile(r"\b(?:class|record|struct|interface|enum)\s+([A-Z][A-Za-z0-9_]*)\b")
    for f in src.rglob("*.cs"):
        if "/bin/" in f.as_posix() or "/obj/" in f.as_posix():
            continue
        for name in set(decl.findall(f.read_text(encoding="utf-8"))):
            idx.setdefault(name, []).append(f)
    return idx


def member_exists(idx, cls: str, member: str) -> bool:
    return any(re.search(rf"\b{re.escape(member)}\b", f.read_text(encoding="utf-8"))
               for f in idx.get(cls, []))


def declared_value(idx, cls: str, member: str) -> str | None:
    """Literal initializer of `member` in `cls`: const, field, or `{ get; init; } = v`."""
    pat = re.compile(rf"\b{re.escape(member)}\s*(?:\{{[^}}]*\}}\s*)?=\s*([^;,\n]+?)\s*;")
    for f in idx.get(cls, []):
        m = pat.search(f.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return None


def as_number(literal: str) -> float | None:
    s = literal.strip().replace("_", "").rstrip("mMfFdD")
    try:
        return float(s)
    except ValueError:
        return None


def check(root: Path | None = None) -> Report:
    root = root or repo_root(Path(__file__).parent)
    src = root / "src"
    table_md = root / "DataMemo" / "spec" / "SymbolTable.md"
    rep = Report()
    rows = parse_table(table_md.read_text(encoding="utf-8"))
    ids = {r.id for r in rows}
    idx = type_index(src)

    # 6. table shape
    for r in rows:
        want = COLUMNS.get(r.section, 8)
        if len(r.cells) != want:
            rep.fail(f"SymbolTable.md:{r.line} `{r.id}` has {len(r.cells)} cells, expected {want} "
                     "(unescaped `|` inside math? use \\lvert/\\rvert)")

    # 1. anchors → rows
    anchors: dict[str, list[str]] = {}
    for f in src.rglob("*.cs"):
        if "/bin/" in f.as_posix() or "/obj/" in f.as_posix():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for a in ANCHOR.findall(line):
                anchors.setdefault(a, []).append(f"{f.relative_to(root)}:{n}")
    for a, where in sorted(anchors.items()):
        if a not in ids:
            rep.fail(f"[math:{a}] at {where[0]} has no SymbolTable row")

    for r in rows:
        if r.section in COLUMNS or len(r.cells) < 7:
            continue
        code, test = r.cells[4], r.cells[5]
        # 2. coverage
        if r.section in LIVE_SECTIONS and not r.planned and not code.startswith("—") and r.id not in anchors:
            rep.fail(f"live row `{r.id}` (§{r.section}) has no [math:{r.id}] tag in src/")
        # 3. members
        for cls, mem in members_in(code) + members_in(test):
            if cls not in idx:
                rep.fail(f"`{r.id}`: type `{cls}` not found in src/")
            elif not member_exists(idx, cls, mem):
                rep.fail(f"`{r.id}`: member `{cls}.{mem}` not found")

    # 4. constants
    for r in (r for r in rows if r.section == "K"):
        value, code = r.cells[2], r.cells[4]
        if code.startswith("—"):          # an unnamed literal, recorded but not checkable
            continue
        want = as_number(value)
        for cls, mem in members_in(code):
            lit = declared_value(idx, cls, mem)
            if lit is None:
                rep.fail(f"constant `{r.id}`: no literal declaration for `{cls}.{mem}`")
            elif as_number(lit) is None or want is None or abs(as_number(lit) - want) > 1e-12 * max(1, abs(want)):
                rep.fail(f"constant `{r.id}`: SymbolTable says {value}, `{cls}.{mem}` = {lit}")

    # 5. schema order
    b_ids = [r.id.removeprefix("x.") for r in rows if r.section == "B"]
    if b_ids != NUMERIC_FEATURES:
        rep.fail(f"§B coordinates {b_ids} != NUMERIC_FEATURES {NUMERIC_FEATURES}")

    # 7. links
    try:
        tracked = subprocess.run(["git", "ls-files", "*.md"], cwd=root,
                                 capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        tracked = [str(p.relative_to(root)) for p in root.rglob("*.md") if ".venv" not in p.parts]
    for rel in tracked:
        if rel.startswith(FROZEN_MD):
            continue
        p = root / rel
        for t in re.findall(r"\]\(([^)\s]+)\)", p.read_text(encoding="utf-8")):
            if re.match(r"^[a-z][a-z0-9+.-]*:", t) or t.startswith(("#", "\\")):
                continue
            if not (p.parent / t.split("#")[0]).exists():
                rep.fail(f"broken link in {rel}: ({t})")
    return rep


def main() -> int:
    rep = check()
    if rep.errors:
        print(f"[docs-check] {len(rep.errors)} problem(s):", file=sys.stderr)
        for e in rep.errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1
    print("[docs-check] SymbolTable ↔ code ↔ tests ↔ links: in sync ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
