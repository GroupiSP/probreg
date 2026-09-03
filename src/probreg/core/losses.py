"""Backend-neutral per-example objectives for probabilistic regression."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from probreg.core.distributions import PredictiveDistribution
from probreg.core.types import Array, Batch


def _identity(value: Array) -> Array:
    """Return ``value`` unchanged; the default no-op gradient-stopping hook."""
    return value


@dataclass(frozen=True)
class NegativeLogLikelihoodLoss:
    """Configurable negative log-likelihood for predictive distributions.

    Attributes:
        beta: Variance-reweighting exponent in ``[0, 1]``. Zero recovers
            ordinary negative log-likelihood.
        stop_gradient: Callable excluding the variance weight from gradient
            propagation when required by an autodiff backend.
        target_transform: Callable applied to targets before evaluating the
            predictive distribution. Defaults to the identity transform.
    """

    beta: float = 0.0
    stop_gradient: Callable[[Array], Array] = field(default=_identity)
    target_transform: Callable[[Array], Array] = field(default=_identity)

    def __post_init__(self) -> None:
        """Validate the variance-reweighting exponent.

        Raises:
            ValueError: If ``beta`` is outside ``[0, 1]``.
        """
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1].")

    def per_example(self, prediction: PredictiveDistribution, batch: Batch) -> Array:
        """Compute the optionally transformed and beta-weighted NLL.

        Args:
            prediction: The predictive distribution produced for ``batch``.
            batch: Batch whose targets are scored under ``prediction``.

        Returns:
            An array of per-example negative log-likelihood values.

        Raises:
            ValueError: If ``batch.targets`` is missing.
        """
        if batch.targets is None:
            raise ValueError("batch.targets must be provided.")
        targets = self.target_transform(batch.targets)
        nll = -prediction.log_prob(targets)
        if self.beta == 0.0:
            return nll
        weight = self.stop_gradient(prediction.variance()) ** self.beta
        return nll * weight


@dataclass(frozen=True)
class SquaredErrorLoss:
    """Per-example squared error for deterministic predictions."""

    def per_example(self, prediction: Array, batch: Batch) -> Array:
        """Compute squared prediction errors.

        Args:
            prediction: Deterministic model predictions.
            batch: Batch containing regression targets.

        Returns:
            Elementwise squared prediction errors.

        Raises:
            ValueError: If ``batch.targets`` is missing.
        """
        if batch.targets is None:
            raise ValueError("batch.targets must be provided.")
        return (prediction - batch.targets) ** 2


def add_epsilon(epsilon: float = 1e-12) -> Callable[[Array], Array]:
    """Create a target transform that adds a positive stabilization offset.

    Args:
        epsilon: Positive finite offset added to each target.

    Returns:
        A callable adding ``epsilon`` to an array.

    Raises:
        ValueError: If ``epsilon`` is not positive and finite.
    """
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite.")

    def transform(targets: Array) -> Array:
        return targets + epsilon

    return transform
