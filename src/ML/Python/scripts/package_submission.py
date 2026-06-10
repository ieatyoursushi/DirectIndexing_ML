"""Assemble the course-submission zip.

Invoked by `dotnet run submission`. Gathers the report deliverables (at the
zip root, grader-friendly), the source tree, the math memos, and the final
dataset + ML artifacts. Excludes bulk that is regenerable or oversized:
data/raw/ (~77MB of price JSON), model .zip binaries, build output, venvs.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

#: Deliverables copied to the zip ROOT — these must exist (run `dotnet run report` first).
DELIVERABLES = [
    "src/Export/report/final_report.ipynb",
    "src/Export/report/final_report.html",
    "src/Export/report/codebook.md",
    "src/Export/report/codebook.html",
]

EXCLUDED_DIR_NAMES = {"bin", "obj", "__pycache__", ".venv", ".ipynb_checkpoints",
                      "eda", "eda-mlnet", "models", "models-mlnet",
                      "report"}   # src/Export/report duplicates the zip-root deliverables


def excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    return rel.suffix == ".egg-info" or any(p.endswith(".egg-info") for p in rel.parts)


def add_tree(zf: zipfile.ZipFile, root: Path, top: str,
             keep: "callable[[Path], bool] | None" = None) -> int:
    n = 0
    base = root / top
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if excluded(rel):
            continue
        if keep is not None and not keep(rel):
            continue
        zf.write(p, str(rel))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--no-data", action="store_true",
                    help="omit data/lots.csv if an upload size cap bites")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out  = Path(args.out).resolve()

    missing = [d for d in DELIVERABLES if not (root / d).exists()]
    if missing:
        print("[submission] missing report deliverables:", file=sys.stderr)
        for m in missing:
            print(f"[submission]   {root / m}", file=sys.stderr)
        print("[submission] run `dotnet run report` first.", file=sys.stderr)
        sys.exit(2)

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # Report deliverables at the zip root.
        for d in DELIVERABLES:
            zf.write(root / d, Path(d).name)
            n += 1

        # Top-level project files + math memos.
        for f in ("README.md", "DirectIndexing.sln"):
            if (root / f).exists():
                zf.write(root / f, f)
                n += 1
        n += add_tree(zf, root, "DataMemo")

        # Full source tree (minus build output / venvs / regenerable exports).
        n += add_tree(zf, root, "src")

        # Final dataset + metadata (raw price cache excluded by design).
        if not args.no_data:
            zf.write(root / "data" / "lots.csv", "data/lots.csv")
            n += 1
        if (root / "data" / "constituents.json").exists():
            zf.write(root / "data" / "constituents.json", "data/constituents.json")
            n += 1

        # ML artifacts: JSON + CSV only — model .zip binaries are regenerable.
        n += add_tree(zf, root, "data/artifacts-mlnet",
                      keep=lambda rel: rel.suffix in (".json", ".csv"))

    size_mb = out.stat().st_size / 1e6
    print(f"[submission] wrote {out} — {n} files, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
