"""A mean-variance estimation (MVE) loss for the JAX supervised runner.

MVE (Nix & Weigend, 1994, "Estimating the mean and variance of the target
probability distribution") trains a single distribution-valued model — most
commonly a :class:`~probreg.jax.distributions.GaussianHead`, or any NNX
module composing a backbone with such a head — to predict both a mean and an
aleatoric scale under a Gaussian negative log-likelihood loss, optionally
beta-weighted (Seitzer et al., 2022). This compatibility module forwards the
historical ``make_mve_loss`` API to the generic
:func:`probreg.jax.losses.make_supervised_loss` adapter.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.jax.evaluation import SupervisedLoss
from probreg.jax.losses import make_supervised_loss


def make_mve_loss(
    loss: NegativeLogLikelihoodLoss,
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
        loss: Gaussian negative-log-likelihood objective, optionally
            beta-weighted, computing unreduced values from a prediction and
            batch.
        reduction: A callable reducing the (optionally sample-weighted)
            per-example loss array to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A callable ``mve_loss(model, inputs, targets, sample_weight, key,
        training)`` returning the scalar reduced MVE loss for a batch.
    """

    return make_supervised_loss(loss, reduction=reduction)
