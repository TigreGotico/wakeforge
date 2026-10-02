"""Shell blocks in the markdown docs must survive being pasted into bash.

A backslash followed by anything but a newline (an inline comment, a trailing
space) ends the line continuation, so the rest of the command runs as separate
commands and fails.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
FENCE = re.compile(r"^```(bash|sh|shell|console)\s*\n(.*?)^```", re.S | re.M)
BAD_CONTINUATION = re.compile(r"\\[ \t]+\S|\\[ \t]+$")


def _bad_lines(path: Path):
    text = path.read_text(encoding="utf-8")
    for block in FENCE.finditer(text):
        first = text.count("\n", 0, block.start(2)) + 1
        for offset, line in enumerate(block.group(2).splitlines()):
            if BAD_CONTINUATION.search(line):
                yield first + offset, line


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_blocks_have_no_text_after_continuation(path):
    bad = [f"{path.relative_to(ROOT)}:{n}: {line}" for n, line in _bad_lines(path)]
    assert not bad, "line continuation followed by text:\n" + "\n".join(bad)


def test_scanner_flags_inline_comment():
    sample = "```bash\ncmd \\\n  --a 1 \\   # note\n  --b 2\n```\n"
    block = FENCE.search(sample)
    assert any(BAD_CONTINUATION.search(l) for l in block.group(2).splitlines())
