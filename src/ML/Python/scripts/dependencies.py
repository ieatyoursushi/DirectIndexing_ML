"""Brute-force C# dependency analyzer — couplings, calls, inheritance.

Regex-reads every .cs file under --src (no Roslyn, no compilation — same
approach as Zombtoy/DevTools/Diagrams) and emits ONE markdown file with
embedded Mermaid diagrams + tables:

  1. namespace-level layer dependency graph
  2. class-level dependency graph (type references, weighted, clustered)
  3. inheritance / interface-implementation diagram
  4. coupling metrics (fan-in / fan-out per type)
  5. cross-class call detail (Type.Method( call sites + new Type( constructions)

Mermaid renders natively on GitHub / VS Code preview — no plantuml/graphviz.

Known approximations (documented in the output): regex parsing can't resolve
instance-typed receivers (`loader.Load()` doesn't tell us loader's type), so
the call table covers static calls, constructions, and PascalCase receivers;
the *reference* graph (any mention of a known type name) is the complete
coupling picture.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── stripping (comments / strings must not produce fake edges) ──────────────
COMMENT_MULTI  = re.compile(r'/\*.*?\*/', re.DOTALL)
COMMENT_SINGLE = re.compile(r'//.*')
STRING_LIT     = re.compile(r'@?\$?"(?:""|\\.|[^"\n])*"')

# ── declarations ─────────────────────────────────────────────────────────────
NAMESPACE_RE = re.compile(r'\bnamespace\s+([A-Za-z_][A-Za-z0-9_.]*)')
# class / record / record struct / interface / struct / enum, with optional base list
TYPE_DECL_RE = re.compile(
    r'\b(?P<kind>class|record(?:\s+struct)?|interface|struct|enum)\s+'
    r'(?P<name>[A-Z][A-Za-z0-9_]*)'
    r'(?:\s*<[^>{;]*>)?'                       # generic params on the decl
    r'(?:\s*\([^)]*\))?'                       # primary-constructor records
    r'(?:\s*:\s*(?P<bases>[^{;]+?))?\s*[{;]')
NEW_RE  = re.compile(r'\bnew\s+([A-Z][A-Za-z0-9_]*)\s*[(<{]')
CALL_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*[(<]')

EXCLUDED_DIRS = {'bin', 'obj'}

# .NET / ML.NET vocabulary that would otherwise pollute the graph.
BUILTIN = {
    'Math', 'MathF', 'Console', 'Convert', 'Array', 'Enumerable', 'Path',
    'File', 'Directory', 'String', 'Int32', 'Environment', 'Task', 'Parallel',
    'DateTime', 'DateOnly', 'TimeSpan', 'Random', 'Encoding', 'CultureInfo',
    'JsonSerializer', 'Debug', 'GC', 'Activator', 'Type', 'Guid', 'List',
    'Dictionary', 'HashSet', 'SortedSet', 'Queue', 'Stack', 'Tuple', 'Func',
    'Action', 'Exception', 'ArgumentException', 'InvalidOperationException',
}


def strip(code: str) -> str:
    code = COMMENT_MULTI.sub('', code)
    code = COMMENT_SINGLE.sub('', code)
    return STRING_LIT.sub('""', code)


def short_ns(ns: str) -> str:
    """DirectIndexing.ML.MLNet.Models -> ML.MLNet.Models (drop the root prefix)."""
    parts = ns.split('.')
    return '.'.join(parts[1:]) if len(parts) > 1 else ns


def scan(src: Path, include_tests: bool = False):
    """Pass 1: declarations.  Pass 2: references/calls against the known-type set."""
    types = {}        # name -> {kind, ns, file}
    bodies = {}       # file -> stripped code
    extends = defaultdict(set)
    implements = defaultdict(set)
    interfaces = set()

    excluded = EXCLUDED_DIRS if include_tests else EXCLUDED_DIRS | {'Tests'}
    files = sorted(p for p in src.rglob('*.cs')
                   if not (set(p.parts) & excluded))

    for f in files:
        code = strip(f.read_text(encoding='utf-8', errors='ignore'))
        bodies[f] = code
        m_ns = NAMESPACE_RE.search(code)
        ns = short_ns(m_ns.group(1)) if m_ns else '(global)'
        for m in TYPE_DECL_RE.finditer(code):
            name, kind = m.group('name'), m.group('kind')
            if name in types:           # partials / nested duplicates: first wins
                continue
            types[name] = {'kind': kind, 'ns': ns, 'file': f}
            if kind == 'interface':
                interfaces.add(name)
            bases = m.group('bases')
            if bases:
                # Drop generic argument lists BEFORE splitting on commas, so
                # `CustomMappingFactory<SectorIn, SectorOut>` is one base, not two.
                while re.search(r'<[^<>]*>', bases):
                    bases = re.sub(r'<[^<>]*>', '', bases)
                for b in (x.strip() for x in bases.split(',')):
                    if not b or not b[0].isupper():
                        continue
                    (implements if (b in interfaces or b.startswith('I') and len(b) > 1
                                    and b[1].isupper()) else extends)[name].add(b)

    known = set(types)
    # owner type per file: map each decl span -> nearest preceding type decl.
    refs = defaultdict(lambda: defaultdict(int))     # src type -> dst type -> count
    calls = defaultdict(set)                         # (src, dst) -> {methods}

    # Top-level-statement entry point (Program.cs) declares no type but is the
    # orchestrator — synthesize a Program node so its outgoing edges are visible.
    for f, code in bodies.items():
        if not TYPE_DECL_RE.search(code) and re.search(r'\bswitch\s*\(', code) \
                and 'Program' not in types:
            types['Program'] = {'kind': 'entrypoint', 'ns': '(entrypoint)', 'file': f}
    known = set(types)

    for f, code in bodies.items():
        decls = [(m.start(), m.group('name')) for m in TYPE_DECL_RE.finditer(code)
                 if m.group('name') in known and types[m.group('name')]['file'] == f]
        if not decls and types.get('Program', {}).get('file') == f:
            decls = [(0, 'Program')]
        if not decls:
            continue

        def owner(pos: int) -> str:
            cur = decls[0][1]
            for start, name in decls:
                if start <= pos:
                    cur = name
                else:
                    break
            return cur

        for m in re.finditer(r'\b([A-Z][A-Za-z0-9_]*)\b', code):
            t = m.group(1)
            if t in known and t not in BUILTIN:
                o = owner(m.start())
                if o != t:
                    refs[o][t] += 1
        for m in CALL_RE.finditer(code):
            recv, meth = m.group(1), m.group(2)
            if recv in known and recv not in BUILTIN:
                o = owner(m.start())
                if o != recv:
                    calls[(o, recv)].add(meth)
        for m in NEW_RE.finditer(code):
            t = m.group(1)
            if t in known and t not in BUILTIN:
                o = owner(m.start())
                if o != t:
                    calls[(o, t)].add('.ctor')

    return types, refs, calls, extends, implements, interfaces, files


def mermaid_safe(name: str) -> str:
    return name  # type names are already [A-Za-z0-9_]


def emit_markdown(src: Path, types, refs, calls, extends, implements,
                  interfaces, files) -> str:
    by_ns = defaultdict(list)
    for name, info in types.items():
        by_ns[info['ns']].append(name)

    # namespace-level aggregation
    ns_edges = defaultdict(int)
    for s, dsts in refs.items():
        for d, w in dsts.items():
            a, b = types[s]['ns'], types[d]['ns']
            if a != b:
                ns_edges[(a, b)] += w

    fan_out = {t: sum(refs[t].values()) for t in types}
    fan_in = defaultdict(int)
    in_deg = defaultdict(int)
    out_deg = {t: len(refs[t]) for t in types}
    for s, dsts in refs.items():
        for d, w in dsts.items():
            fan_in[d] += w
            in_deg[d] += 1

    L = []
    L.append("# Dependency & Coupling Atlas — .NET layer")
    L.append("")
    L.append(f"> Generated {date.today()} by `dotnet run deps` "
             f"(`scripts/dependencies.py`, brute-force regex scan — no compiler). "
             f"Scanned **{len(files)} .cs files**, **{len(types)} types**, "
             f"**{sum(len(v) for v in refs.values())} reference edges**. "
             f"Same approach as Zombtoy `DevTools/Diagrams`, adapted for C# records. "
             f"Approximation note: instance-typed receivers (`loader.Load()`) can't be "
             f"resolved without a compiler, so §5's call table covers static calls and "
             f"constructions; §2's *reference* graph (any use of a known type name) is "
             f"the complete coupling picture.")
    L.append("")
    L.append("## 1. Layer graph — namespace level")
    L.append("")
    L.append("Arrows read \"references\"; weights are total type-name mentions.")
    L.append("")
    L.append("```mermaid")
    L.append("flowchart LR")
    ns_ids = {ns: f"NS{i}" for i, ns in enumerate(sorted(by_ns))}
    for ns, nid in ns_ids.items():
        L.append(f'    {nid}["{ns}<br/>({len(by_ns[ns])} types)"]')
    for (a, b), w in sorted(ns_edges.items(), key=lambda kv: -kv[1]):
        L.append(f"    {ns_ids[a]} -->|{w}| {ns_ids[b]}")
    L.append("```")
    L.append("")
    L.append("## 2. Class dependency graph — type references, clustered by namespace")
    L.append("")
    L.append("Edge weight = number of times the source type's body mentions the target type.")
    L.append("")
    L.append("```mermaid")
    L.append("flowchart LR")
    for ns in sorted(by_ns):
        L.append(f'    subgraph {ns_ids[ns]}["{ns}"]')
        for t in sorted(by_ns[ns]):
            kind = types[t]['kind']
            label = f"{t}" if kind in ('class',) else f"{t}<br/><i>{kind}</i>"
            L.append(f'        {mermaid_safe(t)}["{label}"]')
        L.append("    end")
    for s in sorted(refs):
        for d, w in sorted(refs[s].items(), key=lambda kv: -kv[1]):
            L.append(f"    {mermaid_safe(s)} -->|{w}| {mermaid_safe(d)}")
    L.append("```")
    L.append("")
    L.append("## 3. Inheritance & interface implementation")
    L.append("")
    if any(extends.values()) or any(implements.values()):
        L.append("```mermaid")
        L.append("classDiagram")
        involved = set()
        for c, bases in extends.items():
            for b in bases:
                involved.update((c, b))
        for c, ifs in implements.items():
            for i in ifs:
                involved.update((c, i))
        for t in sorted(involved):
            if t in interfaces:
                L.append(f"    class {t} {{ <<interface>> }}")
            elif t in types and types[t]['kind'].startswith('record'):
                L.append(f"    class {t} {{ <<record>> }}")
            else:
                L.append(f"    class {t}")
        for c, bases in sorted(extends.items()):
            for b in sorted(bases):
                L.append(f"    {b} <|-- {c}")
        for c, ifs in sorted(implements.items()):
            for i in sorted(ifs):
                L.append(f"    {i} <|.. {c}")
        L.append("```")
        L.append("")
        L.append("(`WeightedRow : LotStateVector` is the load-bearing one — the per-row "
                 "training weight rides on the same immutable schema the simulator wrote.)")
    else:
        L.append("*No inheritance edges found among project types.*")
    L.append("")
    L.append("## 4. Coupling metrics")
    L.append("")
    L.append("Fan-out = types this type references (breadth) / total mentions (weight). "
             "Fan-in = types that reference it. High fan-in = load-bearing schema; "
             "high fan-out = orchestrator.")
    L.append("")
    L.append("| Type | Kind | Namespace | Fan-out (types / refs) | Fan-in (types / refs) |")
    L.append("|---|---|---|---|---|")
    ranked = sorted(types, key=lambda t: -(in_deg[t] + out_deg.get(t, 0)))
    for t in ranked:
        if in_deg[t] + out_deg.get(t, 0) == 0:
            continue
        L.append(f"| `{t}` | {types[t]['kind']} | {types[t]['ns']} "
                 f"| {out_deg.get(t, 0)} / {fan_out.get(t, 0)} "
                 f"| {in_deg[t]} / {fan_in[t]} |")
    L.append("")
    L.append("## 5. Cross-class call detail")
    L.append("")
    L.append("Statically resolvable call sites: `Receiver.Method(...)` where the receiver "
             "is a known project type, plus `new Type(...)` constructions (`.ctor`).")
    L.append("")
    L.append("| Caller | Callee | Members used |")
    L.append("|---|---|---|")
    for (s, d), meths in sorted(calls.items()):
        L.append(f"| `{s}` | `{d}` | {', '.join(f'`{m}`' for m in sorted(meths))} |")
    L.append("")
    L.append("## 6. File inventory")
    L.append("")
    L.append("| Namespace | Type | Kind | File |")
    L.append("|---|---|---|---|")
    for ns in sorted(by_ns):
        for t in sorted(by_ns[ns]):
            relf = types[t]['file'].relative_to(src)
            L.append(f"| {ns} | `{t}` | {types[t]['kind']} | `{relf}` |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the C# source tree")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--include-tests", action="store_true",
                    help="also scan src/Tests (excluded by default for signal)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    types, refs, calls, extends, implements, interfaces, files = scan(
        src, include_tests=args.include_tests)
    md = emit_markdown(src, types, refs, calls, extends, implements, interfaces, files)
    out = out_dir / "Dependencies.md"
    out.write_text(md, encoding="utf-8")
    print(f"[deps] {len(files)} files, {len(types)} types, "
          f"{sum(len(v) for v in refs.values())} reference edges, "
          f"{len(calls)} call edges -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
