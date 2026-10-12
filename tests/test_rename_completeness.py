"""Tests that the library's old name does not come back after the rename to `stagereg`.

ADR 0010 records why the library is named `stagereg`; it is the only tracked file allowed
to keep the old name, as history. The check reads the files git tracks, so build output,
caches and gitignored research notes are not scanned.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).parent.parent
OLD_NAME = "prob" + "reg"  # Split so this file does not match its own pattern.
OLD_NAME_PATTERN = re.compile(OLD_NAME, re.IGNORECASE)
ALLOWED = {Path("docs/adr/0010-the-library-is-named-stagereg.md")}


def _tracked_files() -> list[Path]:
    """Return the repository's tracked files, relative to its root."""
    if shutil.which("git") is None or not (REPOSITORY / ".git").exists():
        pytest.skip("Not running from a git checkout.")
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [Path(name) for name in listing.split("\0") if name]


def test_old_name_appears_only_in_the_naming_adr() -> None:
    offenders = []
    for path in _tracked_files():
        if path in ALLOWED:
            continue
        if OLD_NAME_PATTERN.search(path.as_posix()):
            offenders.append(f"{path}: path")
        file = REPOSITORY / path
        if not file.is_file():
            continue
        text = file.read_bytes().decode(errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if OLD_NAME_PATTERN.search(line):
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, "The old library name is still in use:\n" + "\n".join(
        offenders
    )
