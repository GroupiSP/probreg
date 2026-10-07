"""Tests that every exported dataclass and NamedTuple documents exactly its fields.

The coding style asks each dataclass to describe its constructor fields in a Google-
style `Attributes:` section. A missing entry, or one left behind by a rename, still
builds and renders, so neither the strict docs build nor any other test would notice.
These tests check the public exports of every package the API reference documents, in
both directions.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest
from test_reference_completeness import PAGES, _import

ENTRY = re.compile(r"^(?P<name>\w+)(?:\s*\([^)]*\))?:")


def _fields(cls: type) -> set[str] | None:
    """Return a dataclass's or NamedTuple's constructor fields, else `None`."""
    if dataclasses.is_dataclass(cls):
        return {field.name for field in dataclasses.fields(cls) if field.init}
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):
        return set(cls._fields)  # type: ignore[attr-defined]
    return None


def _documented_attributes(cls: type) -> set[str]:
    """Return the names listed in the class docstring's `Attributes:` section."""
    lines = (inspect.getdoc(cls) or "").splitlines()
    names: set[str] = set()
    in_section = False
    entry_indent: int | None = None
    for line in lines:
        if not in_section:
            in_section = line.strip() == "Attributes:"
            continue
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        if entry_indent is None:
            entry_indent = indent
        if indent == entry_indent and (match := ENTRY.match(line.strip())):
            names.add(match["name"])
    return names


@pytest.mark.parametrize("package", PAGES)
def test_every_dataclass_field_is_documented_exactly_once(package: str) -> None:
    module = _import(package)
    mismatches = []
    for name in sorted(module.__all__):
        cls = getattr(module, name)
        if not inspect.isclass(cls) or not (fields := _fields(cls)):
            continue
        documented = _documented_attributes(cls)
        missing, extra = sorted(fields - documented), sorted(documented - fields)
        if missing or extra:
            mismatches.append(f"{package}.{name}: missing {missing}, extra {extra}")
    assert not mismatches, "Attributes: sections out of sync:\n" + "\n".join(mismatches)
