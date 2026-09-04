from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import jax.numpy as jnp

_BENCHMARK_PATH = (
    Path(__file__).parents[2] / "examples" / "jax" / "xsin" / "benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("benchmark", _BENCHMARK_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the XSin benchmark module.")
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)

XSinConfig = _BENCHMARK.XSinConfig
make_xsin_data = _BENCHMARK.make_xsin_data
run_xsin_mve = _BENCHMARK.run_xsin_mve
run_xsin_two_step = _BENCHMARK.run_xsin_two_step
xsin_mean = _BENCHMARK.xsin_mean
xsin_variance = _BENCHMARK.xsin_variance


def test_xsin_data_is_deterministic_with_positive_variance() -> None:
    config = XSinConfig(train_size=32, evaluation_size=21, seed=4)

    first = make_xsin_data(config)
    second = make_xsin_data(config)

    assert jnp.array_equal(first.train_inputs, second.train_inputs)
    assert jnp.array_equal(first.train_targets, second.train_targets)
    assert bool(jnp.all(first.train_inputs > config.train_min))
    assert bool(jnp.all(first.train_inputs < config.train_max))
    assert first.evaluation_inputs[0, 0] == config.evaluation_min
    assert first.evaluation_inputs[-1, 0] == config.evaluation_max
    assert bool(jnp.any(first.evaluation_inputs < config.train_min))
    assert bool(jnp.any(first.evaluation_inputs > config.train_max))
    assert first.true_mean.shape == (21, 1)
    assert first.true_variance.shape == (21, 1)
    assert bool(jnp.all(first.true_variance > 0.0))
    assert jnp.allclose(first.true_mean, xsin_mean(first.evaluation_inputs))
    assert jnp.allclose(
        first.true_variance,
        xsin_variance(first.evaluation_inputs),
    )


def test_xsin_methods_are_reproducible_and_improve_interpolation() -> None:
    config = XSinConfig(
        train_size=512,
        evaluation_size=101,
        batch_size=64,
        hidden_features=32,
        mve_epochs=300,
        mean_epochs=300,
        variance_epochs=300,
        learning_rate=0.01,
        seed=7,
    )
    data = make_xsin_data(config)

    mve = run_xsin_mve(data, config)
    duplicate_mve = run_xsin_mve(data, config)
    two_step = run_xsin_two_step(data, config)
    duplicate_two_step = run_xsin_two_step(data, config)

    assert two_step.mean.shape == mve.mean.shape == data.true_mean.shape
    assert two_step.variance.shape == mve.variance.shape == data.true_variance.shape
    assert jnp.array_equal(mve.mean, duplicate_mve.mean)
    assert jnp.array_equal(two_step.mean, duplicate_two_step.mean)
    assert all(
        math.isfinite(value)
        for result in (mve, two_step)
        for value in (
            result.mean_rmse,
            result.variance_rmse,
            result.interpolation_mean_rmse,
            result.interpolation_variance_rmse,
            result.extrapolation_mean_rmse,
            result.extrapolation_variance_rmse,
        )
    )
    assert two_step.interpolation_mean_rmse < mve.interpolation_mean_rmse
    assert two_step.interpolation_variance_rmse < mve.interpolation_variance_rmse
