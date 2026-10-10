"""Guard against invisible corruption (e.g. a regex \b turned into a backspace char)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
FILES = sorted(
    p
    for d in ("newsfeed", "tests", "scripts", "config")
    for p in (ROOT / d).rglob("*")
    if p.suffix in {".py", ".yaml", ".md"}
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_control_characters(path: Path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        assert not CONTROL.search(line), f"{path.name}:{n} contains a control character"
