"""Tests for scripts/check_doc_anchors.py.

The two rules that are easy to get wrong, and were got wrong while the
script was written, are covered first: a correct anchor must never be
rewritten, and a range must move as a block in both spellings.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_anchors.py"

SOURCE = '''\
class Alpha:
    """A class."""

    def forward(self, x):
        return x

    def export(self):
        return 1


class Beta:
    def forward(self, x):
        return x
'''
# line 1 Alpha, line 4 Alpha.forward, line 7 Alpha.export, line 11 Beta,
# line 12 Beta.forward


def _load():
    spec = importlib.util.spec_from_file_location("anchors", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path, page_text):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "thing.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text(page_text, encoding="utf-8")
    return tmp_path


def _run(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--docs", "docs", "--src", ".", *args],
        cwd=tmp_path, capture_output=True, text=True)


def _counts(out):
    line = [l for l in out.strip().split("\n") if l.startswith("COUNTS")][-1]
    return dict(p.split("=") for p in line.split()[1:])


def test_a_correct_class_anchor_is_left_alone(tmp_path):
    """`Alpha` with `forward` on one line, anchored at Alpha's own line.

    The scope rule would resolve Alpha.forward at line 4. The anchor names
    line 1, which is Alpha itself, so it is already right.
    """
    t = _tree(tmp_path, "`Alpha.forward` is defined in `thing.py:1`\n")
    r = _run(t, "--fix")
    assert _counts(r.stdout)["OK"] == "1"
    assert _counts(r.stdout)["REWROTE"] == "0"
    assert "thing.py:1" in (t / "docs" / "page.md").read_text()


def test_class_and_method_resolve_to_that_class(tmp_path):
    """Beta.forward is line 12, not Alpha's forward at line 4."""
    t = _tree(tmp_path, "`Beta.forward` lives at `thing.py:99`\n")
    r = _run(t, "--fix")
    assert _counts(r.stdout)["REWROTE"] == "1"
    assert "thing.py:12" in (t / "docs" / "page.md").read_text()


def test_a_plain_range_keeps_its_width(tmp_path):
    t = _tree(tmp_path, "`Alpha.export` is `thing.py:20-24`\n")
    _run(t, "--fix")
    assert "thing.py:7-11" in (t / "docs" / "page.md").read_text()


def test_a_split_range_keeps_its_width(tmp_path):
    """The spelling the pages use, where markdown splits the range."""
    t = _tree(tmp_path, "`Alpha.export` is `thing.py:20`-`24`\n")
    _run(t, "--fix")
    text = (t / "docs" / "page.md").read_text()
    assert "thing.py:7`-`11" in text
    assert "24" not in text, "the end of the range was left behind"


def test_a_line_with_no_symbol_is_reported_and_untouched(tmp_path):
    t = _tree(tmp_path, "the cache is filled at `thing.py:5`\n")
    r = _run(t, "--fix")
    assert _counts(r.stdout)["NOSYM"] == "1"
    assert "thing.py:5" in (t / "docs" / "page.md").read_text()


def test_an_unknown_file_is_reported(tmp_path):
    t = _tree(tmp_path, "`Alpha` is in `nowhere.py:3`\n")
    r = _run(t, "--fix")
    assert _counts(r.stdout)["NOSRC"] == "1"
    assert r.returncode == 1


def test_an_ambiguous_target_is_never_rewritten(tmp_path):
    """`forward` alone is defined twice, so the script must not choose."""
    t = _tree(tmp_path, "`forward` is at `thing.py:99`\n")
    r = _run(t, "--fix")
    assert _counts(r.stdout)["REWROTE"] == "0"
    assert _counts(r.stdout)["FIX"] == "1"
    assert "thing.py:99" in (t / "docs" / "page.md").read_text()


def test_exit_code_is_zero_once_every_anchor_is_right(tmp_path):
    t = _tree(tmp_path, "`Beta.forward` lives at `thing.py:12`\n")
    r = _run(t)
    assert r.returncode == 0


def test_symbol_table_reads_the_syntax_tree(tmp_path):
    mod = _load()
    src = tmp_path / "s.py"
    src.write_text(SOURCE, encoding="utf-8")
    flat, scoped = mod.symbol_table(src)
    assert flat["Alpha"] == [1]
    assert flat["forward"] == [4, 12]
    assert scoped[("Beta", "forward")] == [12]
