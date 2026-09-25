#!/usr/bin/env python3
"""Check, and optionally repair, every ``file.py:N`` anchor in the docs.

    python3 scripts/check_doc_anchors.py            # report
    python3 scripts/check_doc_anchors.py --fix      # rewrite what is certain

A page cites source lines as ``model.py:815``. The line moves whenever
anything above it moves, and nothing tells the page. This reads every anchor
on every page and compares it with the real definition lines, which it takes
from the syntax tree rather than from a regular expression.

Each anchor is judged against the symbols named on the same line of the page:

    OK      N is the definition line of a symbol named on that line
    FIX     a symbol on the line is defined elsewhere, so N is wrong
    NOSYM   the line names no symbol, so only a reader can judge it
    NOSRC   the file the anchor names is not in the tree, or is ambiguous

``Class`` and ``method`` named together resolve to that class's own method,
so ``MfccExtractor`` with ``forward`` means the ``forward`` inside
``MfccExtractor`` and not the twenty others. An anchor whose target is still
ambiguous is reported and never rewritten.

``--fix`` rewrites only anchors with exactly one resolved target, and keeps
the width of a range anchor (``feats.py:96-102`` moves as a block).

The exit code is 1 when any FIX or NOSRC remains. NOSYM does not fail the
run: those anchors point at statements, not definitions. Use ``--strict`` to
fail on them too.

Not wired into CI on purpose. It reads the source tree beside the docs, so a
job would redden unrelated pull requests every time a line moves. Run it in a
fix round, as T-2483 did.
"""
import argparse
import ast
import re
import sys
from pathlib import Path

# A range is written two ways in these pages: "feats.py:96-102", and
# "`feats.py:96`-`102`" where the markdown splits it across code spans. Both
# must move as a block, or a repair leaves a dangling end line.
ANCHOR = re.compile(
    r"\b((?:[A-Za-z0-9_.-]+/)*[A-Za-z_][A-Za-z0-9_]*\.py):(\d+)"
    r"(?:(?P<sep>-|`\s*[-\u2013\u2014]\s*`)(?P<end>\d+))?")


def symbol_table(path):
    """Return {'name': [lines]} and {('Class', 'method'): [lines]}."""
    flat, scoped = {}, {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return flat, scoped

    def walk(node, cls=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                flat.setdefault(child.name, []).append(child.lineno)
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flat.setdefault(child.name, []).append(child.lineno)
                if cls:
                    scoped.setdefault((cls, child.name), []).append(child.lineno)
                walk(child, cls)
            else:
                walk(child, cls)

    walk(tree)
    return flat, scoped


def candidates(words, flat):
    """Every definition line of every symbol the page line names."""
    lines = []
    for w in words:
        if w in flat:
            lines.extend(flat[w])
    return lines


def range_uses_a_named_symbol(path, start, end, words, flat):
    """True when the anchored lines MENTION a symbol the page line names.

    A definition anchor points at the line a symbol is defined on. A USE-SITE
    anchor points at the block that calls or builds it, which is usually far
    from the definition and is not a definition line of anything. The two are
    told apart by reading the anchored lines: if the symbol appears there, the
    anchor is already pointing at real uses of it.

    Without this, a use site on a page line that also names a symbol defined
    in the same file is resolved to that one definition and pulled to its
    header. That is what happened to `docs/guides/distillation.md:68`, which
    anchored the block building the teacher, the student and the head, and was
    rewritten to the header of the student class alone.
    """
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return False
    lo = max(start, 1)
    hi = min(end if end is not None else start, len(lines))
    if lo > hi:
        return False
    body = "\n".join(lines[lo - 1:hi])
    named = [w for w in words if w in flat]
    return any(re.search(rf"\b{re.escape(w)}\b", body) for w in named)


def is_definition_line(path, line_no, name):
    """True when `name` is really defined on that line of that file.

    A repair is only trustworthy if the line it moves to says what the table
    claimed. The table is built once per run, so this re-reads the file.
    """
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return False
    if not 1 <= line_no <= len(lines):
        return False
    return re.match(rf"\s*(class|def|async\s+def)\s+{re.escape(name)}\b",
                    lines[line_no - 1]) is not None


def resolve(words, flat, scoped):
    """The lines an anchor on a page line could mean, most specific first."""
    classes = [w for w in words if w in flat and w[:1].isupper()]
    for cls in classes:
        for w in words:
            if (cls, w) in scoped:
                return scoped[(cls, w)], f"{cls}.{w}"
    for w in words:
        if w in flat:
            return flat[w], w
    return [], ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--src", default=".")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite anchors whose target is unambiguous")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quiet-ok", action="store_true")
    args = ap.parse_args()

    sources = [p for p in Path(args.src).rglob("*.py")
               if ".git" not in p.parts and "node_modules" not in p.parts]
    tables = {str(p).lstrip("./"): symbol_table(p) for p in sources}
    paths = {str(p).lstrip("./"): p for p in sources}

    counts = {"OK": 0, "USE": 0, "FIX": 0, "NOSYM": 0, "NOSRC": 0,
              "REWROTE": 0}

    for page in sorted(Path(args.docs).rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        lines = text.split("\n")
        changed = False
        for n, line in enumerate(lines, 1):
            out = line
            for m in reversed(list(ANCHOR.finditer(line))):
                fname, start = m.group(1), int(m.group(2))
                end = int(m.group("end")) if m.group("end") else None
                sep = m.group("sep") or "-"
                hits = [k for k in tables
                        if k == fname or k.endswith("/" + fname)]
                if len(hits) != 1:
                    counts["NOSRC"] += 1
                    why = "no such file" if not hits else f"ambiguous file: {hits}"
                    print(f"NOSRC {page}:{n}: {m.group(0)} {why}")
                    continue
                flat, scoped = tables[hits[0]]
                words = re.findall(r"\w+", line)
                targets, named = resolve(words, flat, scoped)
                if not targets:
                    counts["NOSYM"] += 1
                    if not args.quiet_ok or True:
                        print(f"NOSYM {page}:{n}: {m.group(0)}")
                    continue
                # An anchor that already lands on a definition line of any
                # symbol the page names is right as written. Only when it
                # lands on none of them is the most specific target used as
                # the repair, so a correct class anchor is never "fixed" into
                # one of that class's methods.
                if start in targets or start in candidates(words, flat):
                    counts["OK"] += 1
                    if not args.quiet_ok:
                        print(f"OK    {page}:{n}: {m.group(0)} {named}")
                    continue
                if range_uses_a_named_symbol(paths[hits[0]], start, end,
                                             words, flat):
                    counts["USE"] += 1
                    print(f"USE   {page}:{n}: {m.group(0)} "
                          f"(the lines use {named}, not define it)")
                    continue
                counts["FIX"] += 1
                detail = f"{named}={targets}"
                if (args.fix and len(targets) == 1
                        and is_definition_line(paths[hits[0]], targets[0],
                                               named.split(".")[-1])):
                    new_start = targets[0]
                    new = f"{fname}:{new_start}"
                    if end is not None:
                        new += f"{sep}{new_start + (end - start)}"
                    out = out[:m.start()] + new + out[m.end():]
                    counts["REWROTE"] += 1
                    print(f"FIXED {page}:{n}: {m.group(0)} -> {new} ({named})")
                else:
                    print(f"FIX   {page}:{n}: {m.group(0)} {detail}")
            if out != line:
                lines[n - 1] = out
                changed = True
        if changed and args.fix:
            page.write_text("\n".join(lines), encoding="utf-8")

    print("COUNTS " + " ".join(f"{k}={v}" for k, v in counts.items()))
    bad = counts["FIX"] + counts["NOSRC"] + (counts["NOSYM"] if args.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
