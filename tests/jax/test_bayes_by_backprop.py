from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pytest

from probreg.core.types import Batch, PyTree
from probreg.jax import BayesByBackprop, IsotropicGaussianPrior, PosteriorProblem

# Bayesian linear regression y = w x + b + noise with known noise. The inputs
# are symmetric about zero, so the design's Gram matrix is diagonal, the exact
# posterior over (w, b) factorizes, and a mean-field Gaussian can match it.
_INPUTS = jnp.linspace(-1.0, 1.0, 20)[:, None]
_NOISE_STD = 0.5
_PRIOR_PRECISION = 2.0
_BATCH_SIZE = 10


def _targets() -> jax.Array:
    noise = _NOISE_STD * jax.random.normal(jax.random.key(0), _INPUTS.shape)
    return 1.5 * _INPUTS - 0.5 + noise


def _line(parameters: PyTree, inputs: PyTree) -> jax.Array:
    return inputs * parameters["w"] + parameters["b"]


def _scaled_log_likelihood(parameters: PyTree, batch: Batch) -> jax.Array:
    targets = jnp.asarray(batch.targets)
    log_density = jax.scipy.stats.norm.logpdf(
        targets, _line(parameters, batch.inputs), _NOISE_STD
    )
    return _INPUTS.shape[0] / targets.shape[0] * jnp.sum(log_density)


def _loader(*, split: str, epoch: int) -> list[Batch]:
    del split, epoch
    targets = _targets()
    return [
        Batch(
            inputs=_INPUTS[start : start + _BATCH_SIZE],
            targets=targets[start : start + _BATCH_SIZE],
        )
        for start in range(0, _INPUTS.shape[0], _BATCH_SIZE)
    ]


@dataclass(frozen=True)
class OpaqueGaussianPrior:
    """The isotropic Gaussian prior, hidden behind a bare ``log_prob``."""

    precision: float

    def log_prob(self, parameters: PyTree) -> jax.Array:
        return IsotropicGaussianPrior(self.precision).log_prob(parameters)


def _problem(prior: object) -> PosteriorProblem:
    return PosteriorProblem(
        initial_parameters={"w": jnp.zeros(()), "b": jnp.zeros(())},
        mean_function=_line,
        log_likelihood=_scaled_log_likelihood,
        prior=prior,  # type: ignore[arg-type]
        train_loader=_loader,
        dataset_size=_INPUTS.shape[0],
    )


def _analytic_predictive(inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The exact posterior predictive mean and epistemic std of the line."""
    design = jnp.concatenate([_INPUTS, jnp.ones_like(_INPUTS)], axis=1)
    precision = _PRIOR_PRECISION * jnp.eye(2) + design.T @ design / _NOISE_STD**2
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ design.T @ _targets()[:, 0] / _NOISE_STD**2
    features = jnp.concatenate([inputs, jnp.ones_like(inputs)], axis=1)
    variance = jnp.einsum("ni,ij,nj->n", features, covariance, features)
    return features @ mean, jnp.sqrt(variance)


def _fit(method: BayesByBackprop, problem: PosteriorProblem, epochs: int) -> None:
    method.init(problem)
    key = jax.random.key(1)
    for epoch in range(epochs):
        for batch in problem.train_loader(split="train", epoch=epoch):
            key, step_key = jax.random.split(key)
            method.update(batch, step_key)


@pytest.mark.parametrize(
    "prior",
    [IsotropicGaussianPrior(_PRIOR_PRECISION), OpaqueGaussianPrior(_PRIOR_PRECISION)],
    ids=["closed-form KL", "Monte Carlo KL"],
)
def test_draws_match_the_analytic_posterior_predictive_of_linear_regression(
    prior: object,
) -> None:
    steps = 1500
    method = BayesByBackprop(
        optimizer=optax.adam(optax.cosine_decay_schedule(0.05, steps)),
        initial_std=0.1,
    )
    _fit(method, _problem(prior), epochs=steps // 2)
    inputs = jnp.linspace(-1.5, 1.5, 7)[:, None]

    draws = method.posterior().sample_means(inputs, jax.random.key(2), 4000)

    mean, std = _analytic_predictive(inputs)
    assert draws.shape == (4000, 7, 1)
    assert jnp.allclose(draws.mean(axis=0)[:, 0], mean, atol=0.2 * std.min())
    assert jnp.allclose(draws.std(axis=0)[:, 0], std, rtol=0.1)
