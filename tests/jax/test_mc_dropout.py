"""MC-Dropout on hand-built posterior problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from posterior_runs import (
    LINE_INPUTS,
    LINE_PRIOR_PRECISION,
    fit,
    line_posterior,
    line_problem,
)

from stagereg.jax import IsotropicGaussianPrior, MCDropout, PosteriorProblem

# Without dropout, MC-Dropout's objective is the line's log-posterior, so it
# converges to the MAP, which is the analytic posterior mean.


def _problem() -> PosteriorProblem:
    return line_problem(IsotropicGaussianPrior(LINE_PRIOR_PRECISION))


def test_without_dropout_it_converges_to_the_map_of_linear_regression() -> None:
    method = MCDropout(optimizer=optax.adam(optax.cosine_decay_schedule(0.05, 1000)))
    fit(method, _problem(), epochs=500)

    fitted = method.posterior_state()

    expected, _ = line_posterior()
    assert jnp.allclose(jnp.stack([fitted["w"], fitted["b"]]), expected, atol=1e-3), (
        fitted,
        expected,
    )


def test_the_posterior_is_unlimited() -> None:
    method = MCDropout()
    method.init(_problem())
    posterior = method.posterior()

    assert posterior.num_draws is None
    with pytest.raises(ValueError, match="num_samples"):
        posterior.sample_means(LINE_INPUTS, jax.random.key(0))


def test_an_uninitialized_method_has_no_posterior() -> None:
    assert not MCDropout().has_posterior
    with pytest.raises(ValueError, match="init"):
        MCDropout().posterior()
