"""Tests for the CMAPSS 1D-CNN mean model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

_MODEL_PATH = Path(__file__).parents[2] / "examples" / "jax" / "cmapss" / "model.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS model module.")
_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODEL
_SPEC.loader.exec_module(_MODEL)


def test_cnn1d_mean_model_outputs_one_scalar_per_example() -> None:
    batch_size, window_length, n_sensors = 4, 30, 9
    model = _MODEL.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(
        jax.random.key(1), (batch_size, window_length, n_sensors)
    )

    predictions = model(inputs)

    assert predictions.shape == (batch_size, 1)


def test_cnn1d_mean_model_handles_a_single_example() -> None:
    model = _MODEL.Cnn1DMeanModel(3, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(2), (1, 30, 3))

    predictions = model(inputs)

    assert predictions.shape == (1, 1)


def test_cnn1d_mean_model_varies_hidden_channels_and_kernel_size() -> None:
    model = _MODEL.Cnn1DMeanModel(5, hidden_channels=8, kernel_size=3, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(3), (6, 30, 5))

    predictions = model(inputs)

    assert predictions.shape == (6, 1)
