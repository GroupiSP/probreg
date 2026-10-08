from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

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


run_xsin_posterior = _BENCHMARK.run_xsin_posterior
xsin_bayes_by_backprop = _BENCHMARK.xsin_bayes_by_backprop
xsin_psgld = _BENCHMARK.xsin_psgld


@pytest.mark.parametrize(
    "make_method", [xsin_bayes_by_backprop, xsin_psgld], ids=["bbb", "psgld"]
)
def test_xsin_posterior_is_reproducible_with_finite_scores(
    make_method: Callable[[Any], Any],
) -> None:
    config = XSinConfig(
        train_size=128,
        evaluation_size=41,
        batch_size=32,
        hidden_features=8,
        mean_epochs=5,
        variance_epochs=5,
        posterior_epochs=2,
        posterior_num_draws=4,
        psgld_burn_in=2,
        psgld_thinning=1,
        seed=3,
    )
    data = make_xsin_data(config)

    first = run_xsin_posterior(data, config, make_method(config))
    second = run_xsin_posterior(data, config, make_method(config))

    assert first.mean.shape == data.true_mean.shape
    assert first.aleatoric_variance.shape == data.true_variance.shape
    assert first.epistemic_variance.shape == data.true_variance.shape
    assert bool(jnp.all(first.epistemic_variance >= 0.0))
    assert jnp.array_equal(first.mean, second.mean)
    assert jnp.array_equal(first.epistemic_variance, second.epistemic_variance)
    assert first.posterior_scores == second.posterior_scores
    assert all(
        math.isfinite(value)
        for scores in (first.posterior_scores, first.variance_stage_scores)
        for value in (
            scores.nll,
            scores.crps,
            scores.interpolation_nll,
            scores.interpolation_crps,
            scores.extrapolation_nll,
            scores.extrapolation_crps,
        )
    )
