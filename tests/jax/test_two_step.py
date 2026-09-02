from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probreg.core.losses import GammaResidualNLLLoss
from probreg.jax.two_step import (
    make_gamma_residual_loss,
    make_mean_squared_error_loss,
)


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


class LogPrediction:
    batch_shape = (2, 1)
    event_shape = ()

    def __init__(self, values: jax.Array) -> None:
        self.values = values

    def log_prob(self, targets: jax.Array) -> jax.Array:
        return jnp.log(targets) + self.values

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        del key
        return jnp.zeros(sample_shape + self.batch_shape)

    def mean(self) -> jax.Array:
        return self.values

    def variance(self) -> jax.Array:
        return jnp.ones_like(self.values)


class LogPredictionModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> LogPrediction:
        return LogPrediction(jnp.zeros_like(inputs))


def test_mean_squared_error_loss_supports_weights_and_reduction() -> None:
    model = LinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_mean_squared_error_loss(reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([[1.0], [5.0]]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_supports_weights_and_reduction() -> None:
    loss = make_gamma_residual_loss(
        GammaResidualNLLLoss(epsilon=1e-6),
        reduction=jnp.sum,
    )

    value = loss(
        LogPredictionModel(),
        jnp.ones((2, 1)),
        jnp.array([[0.0], [1.0]]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    expected = -2.0 * jnp.log(1e-6) - 3.0 * jnp.log(1.000001)
    assert value == pytest.approx(float(expected))


def test_gamma_residual_loss_has_finite_gradient_at_zero() -> None:
    loss = make_gamma_residual_loss(GammaResidualNLLLoss(epsilon=1e-6))

    def evaluate(target: jax.Array) -> jax.Array:
        return loss(
            LogPredictionModel(),
            jnp.ones((1, 1)),
            target,
            None,
            jax.random.key(0),
            True,
        )

    gradient = jax.grad(evaluate)(jnp.zeros((1, 1)))

    assert bool(jnp.all(jnp.isfinite(gradient)))
