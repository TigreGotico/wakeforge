"""No script names a private host, a user or a personal path."""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# A private address, a user name, a home or removable-media path, an IDE
# workspace: none of these belong in a public repository.
FORBIDDEN = re.compile(r"192\.168\.|miro|/home/|/run/media/|PycharmProjects|sshfs")


def test_scripts_carry_no_private_host_user_or_path():
    hits = []
    for path in sorted(SCRIPTS.rglob("*")):
        if path.suffix not in {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json"}:
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(SCRIPTS.parent)}:{lineno}: {line.strip()}")
    assert not hits, "private host, user or path in scripts/:\n" + "\n".join(hits)
