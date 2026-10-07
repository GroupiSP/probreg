"""Tests that library docstrings write cross-references in autorefs form (ADR 0007).

mkdocstrings renders a Sphinx role such as ``:class:`Foo``` as literal text, and the strict
docs build does not fail on it, so a role left in the source is a silent dead reference.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent / "src" / "probreg"

SPHINX_ROLE = re.compile(r":(?:py:)?[a-z]+:`")


def test_no_sphinx_role_in_library_source() -> None:
    offenders = [
        f"{path.relative_to(SOURCE_DIR.parent)}:{number}: {line.strip()}"
        for path in sorted(SOURCE_DIR.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if SPHINX_ROLE.search(line)
    ]
    assert not offenders, "Sphinx roles found; use autorefs syntax:\n" + "\n".join(
        offenders
    )
