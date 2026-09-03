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
            sample_weight = _align_sample_weight(sample_weight, per_example)
            per_example = per_example * sample_weight
        return reduction(per_example)

    return supervised_loss


def _align_sample_weight(
    sample_weight: jax.Array,
    per_example: jax.Array,
) -> jax.Array:
    """Align one weight per batch item with an unreduced loss array.

    Args:
        sample_weight: Per-sample weights containing a leading batch dimension.
        per_example: Unreduced losses containing the same batch dimension.

    Returns:
        Weights broadcastable to ``per_example`` without crossing batch items.

    Raises:
        ValueError: If either value lacks a batch dimension, batch sizes differ,
            or the remaining dimensions are not broadcast-compatible.
    """
    if sample_weight.ndim == 0:
        raise ValueError("sample_weight must include a batch dimension.")
    if per_example.ndim == 0:
        raise ValueError("per-example loss must include a batch dimension.")
    if sample_weight.shape[0] != per_example.shape[0]:
        raise ValueError(
            "sample_weight and per-example loss must have matching batch sizes."
        )
    if sample_weight.ndim == 1 and per_example.ndim > 1:
        return sample_weight.reshape(
            (sample_weight.shape[0],) + (1,) * (per_example.ndim - 1)
        )
    try:
        jax.lax.broadcast_shapes(sample_weight.shape, per_example.shape)
    except ValueError as error:
        raise ValueError(
            "sample_weight must be broadcastable to the per-example loss shape."
        ) from error
    return sample_weight
