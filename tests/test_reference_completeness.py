"""Tests that the API reference pages and the packages' exports cannot drift.

Each reference page holds one `:::` directive per exported symbol, named by its public
path (`probreg.core.GaussianNLLLoss`), under a `## probreg.core.losses`-style section for
the module that defines it. A page missing a directive still builds, so the strict
docs build cannot catch it; these tests do, in both directions.

The packages checked are the ones the docs build imports, read from `mkdocs_hooks.py`
without importing it, so these tests run without the docs dependency group.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections import Counter
from pathlib import Path
from types import ModuleType


REPOSITORY = Path(__file__).parent.parent
REFERENCE_DIR = REPOSITORY / "docs_site" / "reference"
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


def _documented_packages() -> tuple[str, ...]:
    """Return the docs build's `DOCUMENTED_PACKAGES`, parsed rather than imported."""
    tree = ast.parse((REPOSITORY / "mkdocs_hooks.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DOCUMENTED_PACKAGES"
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("mkdocs_hooks.py does not define DOCUMENTED_PACKAGES.")


def _exported_and_documented(module: ModuleType) -> tuple[set[str], set[str]]:
    """Return the package's exports and its reference page's directive targets."""
    exported = {f"{module.__name__}.{name}" for name in module.__all__}
    documented = {target for _, target in _entries(module.__name__)}
    return exported, documented


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


def test_every_documented_package_has_a_reference_page(
    documented_packages: tuple[str, ...],
) -> None:
    assert sorted(PAGES) == sorted(_documented_packages())
    assert sorted(documented_packages) == sorted(_documented_packages())


def test_every_export_has_a_reference_entry(documented_module: ModuleType) -> None:
    exported, documented = _exported_and_documented(documented_module)
    assert sorted(exported - documented) == []


def test_every_reference_entry_is_an_export(documented_module: ModuleType) -> None:
    exported, documented = _exported_and_documented(documented_module)
    assert sorted(documented - exported) == []


def test_no_symbol_is_documented_twice(documented_package: str) -> None:
    counts = Counter(target for _, target in _entries(documented_package))
    assert sorted(target for target, count in counts.items() if count > 1) == []


def test_every_entry_sits_under_its_defining_module(
    documented_package: str, documented_module: ModuleType
) -> None:
    misplaced = [
        (section, target)
        for section, target in _entries(documented_package)
        if not section.startswith(f"{documented_package}.")
        or not _defines(section, target.rpartition(".")[2])
    ]
    assert misplaced == []
