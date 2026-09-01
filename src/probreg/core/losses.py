"""Backend-neutral losses operating on predictive distributions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from probreg.core.distributions import PredictiveDistribution
from probreg.core.types import Array, Batch


def _identity(value: Array) -> Array:
    """Return ``value`` unchanged; the default no-op gradient-stopping hook."""
    return value


@dataclass(frozen=True)
class GaussianNLLLoss:
    """Per-example negative log-likelihood under a predictive distribution."""

    def per_example(self, prediction: PredictiveDistribution, batch: Batch) -> Array:
        """Compute the per-example negative log-likelihood.

        Args:
            prediction: The predictive distribution produced for ``batch``.
            batch: The batch whose ``targets`` are scored under
                ``prediction``.

        Returns:
            An array of per-example negative log-likelihood values.
        """
        return -prediction.log_prob(batch.targets)


@dataclass(frozen=True)
class BetaNLLLoss:
    """Beta-weighted negative log-likelihood (Seitzer et al., 2022).

    Reweights the negative log-likelihood by
    ``stop_gradient(variance) ** beta`` so that low-uncertainty examples
    contribute less to the mean-parameter gradient than under plain NLL,
    while ``beta == 0`` recovers plain NLL exactly.

    Because gradient-stopping is backend-specific, callers integrating an
    autodiff backend (e.g. JAX) should supply that backend's
    gradient-stopping primitive as ``stop_gradient`` (e.g.
    ``jax.lax.stop_gradient``) so the reweighting factor is excluded from
    backpropagation, matching the reference beta-NLL definition. The default
    no-op is appropriate for backends without autodiff, e.g. plain NumPy
    evaluation and tests.

    Attributes:
        beta: The reweighting exponent, constrained to ``[0, 1]``.
        stop_gradient: A callable applied to ``prediction.variance()``
            before exponentiation, to detach it from backpropagation.
            Defaults to the identity function.
    """

    beta: float
    stop_gradient: Callable[[Array], Array] = field(default=_identity)

    def __post_init__(self) -> None:
        """Validate that ``beta`` lies within its supported range.

        Raises:
            ValueError: If ``beta`` is not within ``[0, 1]``.
        """
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1].")

    def per_example(self, prediction: PredictiveDistribution, batch: Batch) -> Array:
        """Compute the per-example beta-weighted negative log-likelihood.

        Args:
            prediction: The predictive distribution produced for ``batch``.
            batch: The batch whose ``targets`` are scored under
                ``prediction``.

        Returns:
            An array of per-example beta-weighted negative log-likelihood
            values, equal to plain NLL when ``beta == 0``.
        """
        nll = -prediction.log_prob(batch.targets)
        if self.beta == 0.0:
            return nll
        weight = self.stop_gradient(prediction.variance()) ** self.beta
        return nll * weight
