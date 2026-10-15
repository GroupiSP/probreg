"""MC-Dropout on hand-built posterior problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from stagereg.core.types import Batch, PyTree
from stagereg.jax import IsotropicGaussianPrior, MCDropout, PosteriorProblem

# Bayesian linear regression y = w x + b + noise with known noise: without
# dropout, MC-Dropout's objective is the log-posterior, so it converges to the
# MAP, which is the analytic posterior mean.
_INPUTS = jnp.linspace(-1.0, 1.0, 20)[:, None]
_NOISE_STD = 0.5
_PRIOR_PRECISION = 2.0
_BATCH_SIZE = 10


def _targets() -> jax.Array:
    noise = _NOISE_STD * jax.random.normal(jax.random.key(0), _INPUTS.shape)
    return 1.5 * _INPUTS - 0.5 + noise


def _line(parameters: PyTree, inputs: PyTree, key: jax.Array) -> jax.Array:
    del key
    return inputs * parameters["w"] + parameters["b"]


def _scaled_log_likelihood(
    parameters: PyTree, batch: Batch, key: jax.Array
) -> jax.Array:
    targets = jnp.asarray(batch.targets)
    log_density = jax.scipy.stats.norm.logpdf(
        targets, _line(parameters, batch.inputs, key), _NOISE_STD
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


def _problem() -> PosteriorProblem:
    return PosteriorProblem(
        initial_parameters={"w": jnp.zeros(()), "b": jnp.zeros(())},
        mean_function=_line,
        log_likelihood=_scaled_log_likelihood,
        prior=IsotropicGaussianPrior(_PRIOR_PRECISION),
        train_loader=_loader,
        dataset_size=_INPUTS.shape[0],
    )


def _analytic_map() -> jax.Array:
    """The exact posterior mean, and so MAP, of ``(w, b)``."""
    design = jnp.concatenate([_INPUTS, jnp.ones_like(_INPUTS)], axis=1)
    precision = _PRIOR_PRECISION * jnp.eye(2) + design.T @ design / _NOISE_STD**2
    return jnp.linalg.solve(precision, design.T @ _targets()[:, 0] / _NOISE_STD**2)


def _fit(method: MCDropout, problem: PosteriorProblem, epochs: int) -> None:
    method.init(problem)
    key = jax.random.key(1)
    for epoch in range(epochs):
        for batch in problem.train_loader(split="train", epoch=epoch):
            key, step_key = jax.random.split(key)
            method.update(batch, step_key)


def test_without_dropout_it_converges_to_the_map_of_linear_regression() -> None:
    method = MCDropout(optimizer=optax.adam(optax.cosine_decay_schedule(0.05, 1000)))
    _fit(method, _problem(), epochs=500)

    fitted = method.posterior_state()

    expected = _analytic_map()
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
        posterior.sample_means(_INPUTS, jax.random.key(0))


def test_an_uninitialized_method_has_no_posterior() -> None:
    assert not MCDropout().has_posterior
    with pytest.raises(ValueError, match="init"):
        MCDropout().posterior()
