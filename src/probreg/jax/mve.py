"""A mean-variance estimation (MVE) loss for the JAX supervised runner.

MVE trains a single distribution-valued model — most commonly a
:class:`~probreg.jax.distributions.GaussianHead`, or any NNX module composing
a backbone with such a head — to predict both a mean and an aleatoric scale
under a likelihood-based loss (e.g. Gaussian NLL or beta-NLL). This module
adapts that composition into the :class:`~probreg.jax.evaluation.SupervisedLoss`
protocol so it can be passed directly to
:func:`probreg.jax.supervised.run_supervised` without any change to the
runner itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.distributions import Loss
from probreg.core.types import Batch
from probreg.jax.evaluation import SupervisedLoss


def make_mve_loss(
    loss: Loss,
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a :class:`SupervisedLoss` for mean-variance estimation.

    The returned callable expects ``model`` to be an NNX module that maps
    ``inputs`` directly to a
    :class:`~probreg.core.distributions.PredictiveDistribution` (e.g. a
    :class:`~probreg.jax.distributions.GaussianHead`, or a backbone composed
    with one), matching how :func:`probreg.jax.supervised.run_supervised`
    already calls a plain regression model. The PRNG key and ``training``
    flag are accepted for :class:`~probreg.jax.evaluation.SupervisedLoss`
    compatibility but unused, since a distribution head has no stochastic
    training-time behavior of its own.

    Args:
        loss: A backend-agnostic :class:`~probreg.core.distributions.Loss`
            (e.g. :class:`~probreg.core.losses.GaussianNLLLoss` or
            :class:`~probreg.core.losses.BetaNLLLoss`) computing an
            unreduced per-example loss from a prediction and a batch.
        reduction: A callable reducing the (optionally sample-weighted)
            per-example loss array to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A callable ``mve_loss(model, inputs, targets, sample_weight, key,
        training)`` returning the scalar reduced MVE loss for a batch.
    """

    def mve_loss(
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

    return mve_loss
