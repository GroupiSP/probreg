"""JAX objectives for explicit mean and Gamma variance training stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.losses import GammaResidualNLLLoss
from probreg.core.types import Batch
from probreg.jax.evaluation import SupervisedLoss

_DEFAULT_GAMMA_RESIDUAL_LOSS = GammaResidualNLLLoss()


def make_mean_squared_error_loss(
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a supervised mean-squared-error objective.

    Args:
        reduction: Callable reducing the optionally weighted squared errors
            to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A supervised loss for deterministic mean models.
    """

    def mean_squared_error(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        errors = jnp.square(model(inputs) - targets)
        if sample_weight is not None:
            errors = errors * sample_weight
        return reduction(errors)

    return mean_squared_error


def make_gamma_residual_loss(
    loss: GammaResidualNLLLoss = _DEFAULT_GAMMA_RESIDUAL_LOSS,
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a supervised Gamma NLL objective for squared residual targets.

    Args:
        loss: Backend-neutral Gamma residual objective.
        reduction: Callable reducing the optionally weighted per-example
            losses to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A supervised loss for models returning Gamma predictions.
    """

    def gamma_residual_loss(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        prediction = model(inputs)
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        per_example = loss.per_example(prediction, batch)
        if sample_weight is not None:
            per_example = per_example * sample_weight
        return reduction(per_example)

    return gamma_residual_loss
