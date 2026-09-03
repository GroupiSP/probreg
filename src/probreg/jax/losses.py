"""JAX adaptation of backend-neutral per-example objectives."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.distributions import DistributionLoss, PredictionLoss
from probreg.core.types import Batch, PyTree
from probreg.jax.evaluation import SupervisedLoss


def make_supervised_loss(
    objective: DistributionLoss | PredictionLoss,
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Adapt a per-example objective to the JAX supervised-runner contract.

    Args:
        objective: Backend-neutral objective computing unreduced loss values
            from model predictions and a batch.
        reduction: Callable reducing optionally weighted loss values to a
            scalar. Defaults to ``jnp.mean``.

    Returns:
        A supervised loss that invokes the model, applies the objective and
        optional sample weights, then reduces to a scalar.
    """

    def supervised_loss(
        model: nnx.Module,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        prediction = model(inputs)
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        per_example = objective.per_example(prediction, batch)
        if sample_weight is not None:
            per_example = per_example * sample_weight
        return reduction(per_example)

    return supervised_loss
