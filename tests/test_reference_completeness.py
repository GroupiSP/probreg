"""Tests that the API reference pages and the packages' exports cannot drift.

Each reference page holds one `:::` directive per exported symbol, named by its public
path (`probreg.core.GaussianNLLLoss`), under a `## probreg.core.losses`-style section for
the module that defines it. A page missing a directive still builds, so the strict
docs build cannot catch it; these tests do, in both directions.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections import Counter
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).parent.parent / "docs_site" / "reference"
PAGES = {"probreg.core": "core.md", "probreg.jax": "jax.md"}

SECTION = re.compile(r"^## `?(?P<module>[\w.]+)`?\s*$")
DIRECTIVE = re.compile(r"^::: (?P<target>[\w.]+)\s*$")


def _entries(package: str) -> list[tuple[str, str]]:
    """Return each directive's `(section module, target)`, in page order."""
    section = ""
    entries = []
    for line in (REFERENCE_DIR / PAGES[package]).read_text().splitlines():
        if match := SECTION.match(line):
            section = match["module"]
        elif match := DIRECTIVE.match(line):
            entries.append((section, match["target"]))
    return entries


def _import(package: str):
    if package == "probreg.jax":
        for dependency in ("jax", "flax.nnx", "optax"):
            pytest.importorskip(dependency)
    return importlib.import_module(package)


def _bound_names(node: ast.stmt) -> set[str]:
    """Return the names a top-level statement binds by definition or assignment."""
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return {node.name}
    if isinstance(node, ast.TypeAlias):
        return {node.name.id}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _defines(module_name: str, name: str) -> bool:
    """Whether the module binds `name` itself, rather than importing it."""
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    return any(name in _bound_names(node) for node in tree.body)


@pytest.mark.parametrize("package", PAGES)
def test_every_export_has_a_reference_entry(package: str) -> None:
    exported = {f"{package}.{name}" for name in _import(package).__all__}
    documented = {target for _, target in _entries(package)}
    assert sorted(exported - documented) == []


@pytest.mark.parametrize("package", PAGES)
def test_every_reference_entry_is_an_export(package: str) -> None:
    exported = {f"{package}.{name}" for name in _import(package).__all__}
    documented = {target for _, target in _entries(package)}
    assert sorted(documented - exported) == []


@pytest.mark.parametrize("package", PAGES)
def test_no_symbol_is_documented_twice(package: str) -> None:
    counts = Counter(target for _, target in _entries(package))
    assert sorted(target for target, count in counts.items() if count > 1) == []


@pytest.mark.parametrize("package", PAGES)
def test_every_entry_sits_under_its_defining_module(package: str) -> None:
    _import(package)
    misplaced = [
        (section, target)
        for section, target in _entries(package)
        if not section.startswith(f"{package}.")
        or not _defines(section, target.rpartition(".")[2])
    ]
    assert misplaced == []
