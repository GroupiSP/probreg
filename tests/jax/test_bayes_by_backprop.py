from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from posterior_runs import (
    LINE_INPUTS,
    LINE_PRIOR_PRECISION,
    fit,
    line,
    line_posterior,
    line_problem,
)

from stagereg.core.types import PyTree
from stagereg.jax import (
    BayesByBackprop,
    IsotropicGaussianPrior,
    MeanFieldGaussianPosterior,
    Prior,
)

# The line's exact posterior factorizes over (w, b), so a mean-field Gaussian
# can match it.


@dataclass(frozen=True)
class OpaqueGaussianPrior:
    """The isotropic Gaussian prior, hidden behind a bare ``log_prob``."""

    precision: float

    def log_prob(self, parameters: PyTree) -> jax.Array:
        return IsotropicGaussianPrior(self.precision).log_prob(parameters)


def _analytic_predictive(inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The exact posterior predictive mean and epistemic std of the line."""
    mean, covariance = line_posterior()
    features = jnp.concatenate([inputs, jnp.ones_like(inputs)], axis=1)
    variance = jnp.einsum("ni,ij,nj->n", features, covariance, features)
    return features @ mean, jnp.sqrt(variance)


@pytest.mark.parametrize(
    "prior",
    [
        IsotropicGaussianPrior(LINE_PRIOR_PRECISION),
        OpaqueGaussianPrior(LINE_PRIOR_PRECISION),
    ],
    ids=["closed-form KL", "Monte Carlo KL"],
)
def test_draws_match_the_analytic_posterior_predictive_of_linear_regression(
    prior: Prior,
) -> None:
    steps = 1500
    method = BayesByBackprop(
        optimizer=optax.adam(optax.cosine_decay_schedule(0.05, steps)),
        initial_std=0.1,
    )
    fit(method, line_problem(prior), epochs=steps // 2)
    inputs = jnp.linspace(-1.5, 1.5, 7)[:, None]

    draws = method.posterior().sample_means(inputs, jax.random.key(2), 4000)

    mean, std = _analytic_predictive(inputs)
    assert draws.shape == (4000, 7, 1)
    assert jnp.allclose(draws.mean(axis=0)[:, 0], mean, atol=0.2 * std.min())
    assert jnp.allclose(draws.std(axis=0)[:, 0], std, rtol=0.1)


@pytest.fixture(scope="module")
def warm_posterior() -> MeanFieldGaussianPosterior:
    """A posterior a few steps from the warm start, with a wide spread."""
    method = BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.5)
    fit(method, line_problem(IsotropicGaussianPrior(LINE_PRIOR_PRECISION)), epochs=2)
    return method.posterior()


@settings(deadline=None, max_examples=20)
@given(
    split=st.integers(1, 6),
    num_samples=st.integers(1, 5),
    seed=st.integers(0, 2**16),
)
def test_a_draw_is_the_same_function_at_every_input(
    warm_posterior: MeanFieldGaussianPosterior,
    split: int,
    num_samples: int,
    seed: int,
) -> None:
    inputs = jnp.linspace(-2.0, 2.0, 7)[:, None]
    key = jax.random.key(seed)

    together = warm_posterior.sample_means(inputs, key, num_samples)
    apart = jnp.concatenate(
        [
            warm_posterior.sample_means(inputs[:split], key, num_samples),
            warm_posterior.sample_means(inputs[split:], key, num_samples),
        ],
        axis=1,
    )

    assert jnp.allclose(together, apart)
    # Each draw of the line model is itself a line through all seven inputs.
    slopes = jnp.diff(together[..., 0], axis=1)
    assert jnp.allclose(slopes, slopes[:, :1], atol=1e-5)


def test_draw_s_does_not_depend_on_how_many_draws_are_taken(
    warm_posterior: MeanFieldGaussianPosterior,
) -> None:
    key = jax.random.key(4)

    few = warm_posterior.sample_means(LINE_INPUTS, key, 3)
    many = warm_posterior.sample_means(LINE_INPUTS, key, 8)

    assert jnp.array_equal(few, many[:3])
    assert not jnp.allclose(many[0], many[1])


def test_the_variational_posterior_is_unlimited(
    warm_posterior: MeanFieldGaussianPosterior,
) -> None:
    assert warm_posterior.num_draws is None
    with pytest.raises(ValueError, match="num_samples"):
        warm_posterior.sample_means(LINE_INPUTS, jax.random.key(0))


def test_the_default_warm_start_draws_stay_close_to_the_trained_mean() -> None:
    problem = dataclasses.replace(
        line_problem(IsotropicGaussianPrior()),
        initial_parameters={"w": jnp.asarray(1.5), "b": jnp.asarray(-0.5)},
    )
    method = BayesByBackprop()
    method.init(problem)

    draws = method.posterior().sample_means(LINE_INPUTS, jax.random.key(0), 64)

    trained = line(problem.initial_parameters, LINE_INPUTS, jax.random.key(1))
    assert jnp.max(jnp.abs(draws - trained)) < 0.01


@pytest.mark.parametrize("initial_std", [0.0, -0.1, math.inf, math.nan])
def test_a_non_positive_or_non_finite_initial_std_is_refused(
    initial_std: float,
) -> None:
    with pytest.raises(ValueError, match="initial_std"):
        BayesByBackprop(initial_std=initial_std)


def test_an_uninitialized_method_has_no_posterior() -> None:
    with pytest.raises(ValueError, match="init"):
        BayesByBackprop().posterior()
