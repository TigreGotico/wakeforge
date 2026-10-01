"""The configuration reference of genetic_search.ipynb against its config cell.

Every knob of the notebook is an environment variable read in one cell, and the
first markdown cell documents each one with its default. A wrong default there
is worse than no table: a reader who trusts it declines an option that is on,
or waits for an option that is off. Every table of defaults and every prose
claim of a default are parsed out of the notebook and compared value by value.
"""
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "genetic_search.ipynb"

ENV_GET = re.compile(r"""os\.environ\.get\(\s*["'](\w+)["']\s*,\s*["']([^"']*)["']\s*\)""")

PROSE_DEFAULT = re.compile(r"`([A-Z][A-Z0-9_]*)=([^`]*)`\s*\(default\)")


def _cell_sources(notebook: Path) -> list[tuple[str, str]]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return [(c["cell_type"], "".join(c["source"])) for c in cells]


def code_defaults(notebook: Path) -> dict[str, str]:
    """Environment variable to default string, as the code cells read them."""
    defaults: dict[str, str] = {}
    for cell_type, source in _cell_sources(notebook):
        if cell_type == "code":
            defaults.update(ENV_GET.findall(source))
    return defaults


def _tables(markdown: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every pipe table of *markdown*, as a header and its data rows."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if header is not None:
                tables.append((header, rows))
            header, rows = None, []
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
        elif set("".join(cells)) <= set("-: "):
            continue  # the separator row
        else:
            rows.append(cells)
    if header is not None:
        tables.append((header, rows))
    return tables


def documented_defaults(notebook: Path) -> dict[str, str]:
    """Environment variable to default string, as the markdown tables state it."""
    documented: dict[str, str] = {}
    for cell_type, source in _cell_sources(notebook):
        if cell_type != "markdown":
            continue
        for header, rows in _tables(source):
            if header[:2] != ["Variable", "Default"]:
                continue
            for row in rows:
                documented[row[0].strip("`")] = row[1].strip("`")
    return documented


def documented_variables(notebook: Path) -> set[str]:
    """Every variable named in a table whose first column is the variable."""
    named: set[str] = set()
    for cell_type, source in _cell_sources(notebook):
        if cell_type != "markdown":
            continue
        for header, rows in _tables(source):
            if header[0] != "Variable":
                continue
            named.update(row[0].strip("`") for row in rows)
    return named


def test_the_parser_finds_every_table_it_compares():
    """A silent parse of nothing would make the comparisons below vacuous."""
    documented = documented_defaults(NOTEBOOK)
    # one knob from each of the four tables that carry a Default column
    for variable in ("SEED", "MAX_NEGATIVE", "POPULATION", "FINAL_EPOCHS"):
        assert variable in documented, f"the table holding {variable} did not parse"
    assert code_defaults(NOTEBOOK)["WAKE_WORD"]


@pytest.mark.parametrize("variable", sorted(documented_defaults(NOTEBOOK)))
def test_each_documented_default_is_the_default_the_cell_reads(variable):
    documented = documented_defaults(NOTEBOOK)[variable]
    code = code_defaults(NOTEBOOK)
    assert variable in code, f"{variable} is documented but no cell reads it"
    assert documented == code[variable], (
        f"the table gives {variable} the default {documented!r}, "
        f"the config cell reads {code[variable]!r}"
    )


def test_every_variable_the_notebook_reads_is_documented():
    undocumented = sorted(set(code_defaults(NOTEBOOK)) - documented_variables(NOTEBOOK))
    assert undocumented == []


def test_every_prose_claim_of_a_default_states_the_default_the_cell_reads():
    """Markdown prose calls some values the default outside of any table."""
    code = code_defaults(NOTEBOOK)
    claims = [
        claim
        for cell_type, source in _cell_sources(NOTEBOOK)
        if cell_type == "markdown"
        for claim in PROSE_DEFAULT.findall(source)
    ]
    assert claims, "no prose claim of a default parsed; the comparison is vacuous"
    wrong = {
        variable: (claimed, code.get(variable))
        for variable, claimed in claims
        if code.get(variable) != claimed
    }
    assert wrong == {}, "; ".join(
        f"the prose calls {variable}={claimed} the default, "
        f"the config cell reads {actual!r}"
        for variable, (claimed, actual) in wrong.items()
    )
