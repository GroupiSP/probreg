from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

from probreg.core.types import Batch, PyTree
from probreg.jax import IsotropicGaussianPrior, PosteriorProblem, PreconditionedSGLD

_NOISE_SCALE = 0.5
_PRIOR_PRECISION = 1.0


def _linear_mean(parameters: PyTree, inputs: jax.Array) -> jax.Array:
    """A line through the inputs: ``weight * x + bias``."""
    return parameters["weight"] * inputs[:, 0] + parameters["bias"]


def _linear_regression_batch() -> Batch:
    """Thirty-two noisy points on ``y = 1.5x - 0.5``."""
    inputs = jnp.linspace(-1.0, 1.0, 32)[:, None]
    noise = _NOISE_SCALE * jax.random.normal(jax.random.key(11), (32,))
    return Batch(inputs=inputs, targets=1.5 * inputs[:, 0] - 0.5 + noise)


def _linear_regression_problem(batch: Batch) -> PosteriorProblem:
    """Bayesian linear regression with known noise, one full batch per step."""

    def log_likelihood(parameters: PyTree, batch: Batch) -> jax.Array:
        means = _linear_mean(parameters, batch.inputs)
        return jnp.sum(jstats.norm.logpdf(batch.targets, means, _NOISE_SCALE))

    return PosteriorProblem(
        initial_parameters={"weight": jnp.asarray(0.0), "bias": jnp.asarray(0.0)},
        mean_function=_linear_mean,
        log_likelihood=log_likelihood,
        prior=IsotropicGaussianPrior(precision=_PRIOR_PRECISION),
        train_loader=lambda *, split, epoch: [batch],
        dataset_size=int(batch.inputs.shape[0]),
    )


def _analytic_predictive(
    batch: Batch, inputs: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """The exact posterior mean and spread of the line at ``inputs``.

    Returns:
        The mean and the standard deviation of ``weight * x + bias`` under the
        Gaussian posterior.
    """
    features = np.column_stack([np.asarray(batch.inputs[:, 0]), np.ones(32)])
    targets = np.asarray(batch.targets)
    precision = _PRIOR_PRECISION * np.eye(2) + features.T @ features / _NOISE_SCALE**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ features.T @ targets / _NOISE_SCALE**2
    query = np.column_stack([np.asarray(inputs[:, 0]), np.ones(inputs.shape[0])])
    spread = np.sqrt(np.einsum("ij,jk,ik->i", query, covariance, query))
    return query @ mean, spread


def _run_chain(
    method: PreconditionedSGLD, problem: PosteriorProblem, steps: int, seed: int = 0
) -> None:
    method.init(problem)
    batch = next(iter(problem.train_loader(split="train", epoch=0)))
    for key in jax.random.split(jax.random.key(seed), steps):
        method.update(batch, key)


def test_retained_samples_match_the_bayesian_linear_regression_posterior() -> None:
    batch = _linear_regression_batch()
    method = PreconditionedSGLD(step_size=0.03, burn_in=1_000, thinning=20)
    inputs = jnp.linspace(-2.0, 2.0, 9)[:, None]

    _run_chain(method, _linear_regression_problem(batch), steps=41_000)
    draws = np.asarray(method.posterior().sample_means(inputs, jax.random.key(0)))

    expected_mean, expected_spread = _analytic_predictive(batch, inputs)
    assert draws.shape == (2_000, 9)
    np.testing.assert_allclose(
        draws.mean(axis=0), expected_mean, atol=0.25 * expected_spread.min()
    )
    np.testing.assert_allclose(draws.std(axis=0), expected_spread, rtol=0.15)
