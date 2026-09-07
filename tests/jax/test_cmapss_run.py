"""Tests for the CMAPSS end-to-end mean-stage training and evaluation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_CMAPSS_DIR = Path(__file__).parents[2] / "examples" / "jax" / "cmapss"
if str(_CMAPSS_DIR) not in sys.path:
    sys.path.insert(0, str(_CMAPSS_DIR))

_RUN_PATH = _CMAPSS_DIR / "run.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_run", _RUN_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS run module.")
_RUN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUN
_SPEC.loader.exec_module(_RUN)


def _synthetic_windows(
    rng: np.random.Generator, *, n_windows: int, window_length: int, n_sensors: int
) -> tuple[np.ndarray, np.ndarray]:
    windows = rng.normal(size=(n_windows, window_length, n_sensors))
    targets = rng.uniform(low=1.0, high=100.0, size=n_windows)
    return windows, targets


def test_train_mean_model_and_evaluate_rmse_end_to_end() -> None:
    rng = np.random.default_rng(0)
    train_windows, train_targets = _synthetic_windows(
        rng, n_windows=64, window_length=30, n_sensors=9
    )
    validation_windows, validation_targets = _synthetic_windows(
        rng, n_windows=16, window_length=30, n_sensors=9
    )
    test_windows, test_rul = _synthetic_windows(
        rng, n_windows=10, window_length=30, n_sensors=9
    )
    config = _RUN.CmapssConfig(
        batch_size=16, hidden_channels=4, kernel_size=3, epochs=3, seed=1
    )

    model = _RUN.train_mean_model(
        train_windows, train_targets, validation_windows, validation_targets, config
    )
    test_rmse = _RUN.evaluate_rmse(model, test_windows, test_rul)

    assert math.isfinite(test_rmse)
    assert test_rmse >= 0.0


def test_cmapss_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="window_length and batch_size"):
        _RUN.CmapssConfig(window_length=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        _RUN.CmapssConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="epochs"):
        _RUN.CmapssConfig(epochs=0)
