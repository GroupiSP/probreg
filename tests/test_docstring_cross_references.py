"""Tests that library and example docstrings write cross-references in autorefs form
(ADR 0007).

mkdocstrings renders a Sphinx role such as ``:class:`Foo``` as literal text, and the strict
docs build does not fail on it, so a role left in the source is a silent dead reference.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIRS = (REPO_ROOT / "src" / "probreg", REPO_ROOT / "examples")

SPHINX_ROLE = re.compile(r":(?:py:)?[a-z]+:`")


def test_no_sphinx_role_in_library_or_example_source() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
        for source_dir in SOURCE_DIRS
        for path in sorted(source_dir.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if SPHINX_ROLE.search(line)
    ]
    assert not offenders, "Sphinx roles found; use autorefs syntax:\n" + "\n".join(
        offenders
    )
