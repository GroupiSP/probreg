"""Shared fixtures for the JAX examples' tests.

The examples live under `examples/`, not in the installed package, so
their modules are loaded from their file paths. That loading happened
three times over, once per test module; it happens here once instead,
behind one session-scoped fixture per module. Session scope also keeps
Hypothesis happy: a function-scoped fixture in a `@given` test trips its
`function_scoped_fixture` health check.

Each fixture guards its own optional dependencies, so a test module is
collected even when the extras it needs are absent and only the tests that
actually touch that module skip.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "jax"
_CMAPSS_DIR = _EXAMPLES_DIR / "cmapss"
_TRACKING_DIR = _EXAMPLES_DIR / "tracking"
if str(_CMAPSS_DIR) not in sys.path:
    sys.path.insert(0, str(_CMAPSS_DIR))


def _load_example_module(
    directory: Path, module_name: str, file_name: str
) -> ModuleType:
    """Load one of the examples' modules from its file path.

    Args:
        directory: The example's directory.
        module_name: Name to register the loaded module under, so that
            anything importing it a second time gets the same object.
        file_name: The module's file name within the example's directory.

    Returns:
        The executed module.

    Raises:
        RuntimeError: If the module could not be loaded from its path.
    """
    spec = importlib.util.spec_from_file_location(module_name, directory / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the example module {file_name!r}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cmapss_preprocessing() -> ModuleType:
    """The example's windowing and standardization module."""
    for dependency in ("pandas", "sklearn"):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_preprocessing", "preprocessing.py")


@pytest.fixture(scope="session")
def cmapss_plots() -> ModuleType:
    """The example's RUL-curve plotting module."""
    for dependency in ("matplotlib", "pandas", "sklearn"):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_plots", "plots.py")


@pytest.fixture(scope="session")
def cmapss_run() -> ModuleType:
    """The example's end-to-end training and evaluation module."""
    for dependency in (
        "jax",
        "jax.numpy",
        "flax.nnx",
        "optax",
        "pandas",
        "matplotlib",
        "sklearn",
    ):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_run", "run.py")


@pytest.fixture(scope="session")
def spanning_units(
    cmapss_preprocessing: ModuleType,
) -> Callable[[int, int, int], Any]:
    """Return a factory naming three units in figure column order."""

    def make(shortest: int, median: int, longest: int) -> Any:
        return cmapss_preprocessing.LifetimeSpanningUnits(
            shortest=shortest, median=median, longest=longest
        )

    return make


@pytest.fixture(scope="session")
def tracking_tracker() -> ModuleType:
    """The tracking example's TensorBoard experiment tracker module.

    Deliberately unguarded: the tracker's mapping logic is testable with
    neither `tensorboardX` nor JAX installed, and a skip here would hide
    that property breaking.
    """
    return _load_example_module(
        _TRACKING_DIR, "tracking_tensorboard_tracker", "tensorboard_tracker.py"
    )


@pytest.fixture(scope="session")
def tracking_run() -> ModuleType:
    """The tracking example's end-to-end run module."""
    for dependency in (
        "jax",
        "jax.numpy",
        "flax.nnx",
        "optax",
        "matplotlib",
        "tensorboardX",
    ):
        pytest.importorskip(dependency)
    return _load_example_module(_TRACKING_DIR, "tracking_run", "run.py")
