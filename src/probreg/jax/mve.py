"""A mean-variance estimation (MVE) loss for the JAX supervised runner.

MVE (Nix & Weigend, 1994, "Estimating the mean and variance of the target
probability distribution") trains a single distribution-valued model — most
commonly a :class:`~probreg.jax.distributions.GaussianHead`, or any NNX
module composing a backbone with such a head — to predict both a mean and an
aleatoric scale under a Gaussian negative log-likelihood loss, optionally
beta-weighted (Seitzer et al., 2022). This module adapts that composition
into the :class:`~probreg.jax.evaluation.SupervisedLoss` protocol so it can
be passed directly to :func:`probreg.jax.supervised.run_supervised` without
any change to the runner itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.losses import BetaNLLLoss, GaussianNLLLoss
from probreg.core.types import Batch
from probreg.jax.evaluation import SupervisedLoss


def make_mve_loss(
    loss: GaussianNLLLoss | BetaNLLLoss,
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a :class:`SupervisedLoss` for mean-variance estimation (MVE).

    MVE (Nix & Weigend, 1994) trains a single model to jointly predict a
    mean and an aleatoric variance under a Gaussian likelihood. The returned
    callable expects ``model`` to be an NNX module that maps ``inputs``
    directly to a :class:`~probreg.core.distributions.PredictiveDistribution`
    (e.g. a :class:`~probreg.jax.distributions.GaussianHead`, or a backbone
    composed with one), matching how :func:`probreg.jax.supervised.run_supervised`
    already calls a plain regression model. The PRNG key and ``training``
    flag are accepted for :class:`~probreg.jax.evaluation.SupervisedLoss`
    compatibility but unused, since a distribution head has no stochastic
    training-time behavior of its own.

    Args:
        loss: The MVE objective, either plain Gaussian NLL
            (:class:`~probreg.core.losses.GaussianNLLLoss`, Nix & Weigend,
            1994) or its beta-weighted variant
            (:class:`~probreg.core.losses.BetaNLLLoss`, Seitzer et al.,
            2022), computing an unreduced per-example loss from a
            prediction and a batch.
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
