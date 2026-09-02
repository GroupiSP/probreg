"""Protocols for distribution-valued model predictions and objectives."""

from __future__ import annotations

from typing import Protocol

from probreg.core.types import Array, Batch


class PredictiveDistribution(Protocol):
    """A probability distribution produced by a regression model.

    This protocol is the backend-neutral model/loss boundary. Epoch metrics use backend
    adapters to materialize host arrays rather than retaining distribution objects,
    which intentionally have no concatenation or indexing contract.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    def log_prob(self, targets: Array) -> Array: ...

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array: ...

    def mean(self) -> Array: ...

    def variance(self) -> Array: ...


class DistributionHead(Protocol):
    """Produces a predictive distribution from model features."""

    def __call__(self, features: Array) -> PredictiveDistribution: ...


class Likelihood(Protocol):
    """Evaluates a target under a predictive distribution."""

    def log_prob(self, prediction: PredictiveDistribution, targets: Array) -> Array: ...


class Loss(Protocol):
    """Computes an unreduced loss for each example in a batch."""

    def per_example(
        self, prediction: PredictiveDistribution, batch: Batch
    ) -> Array: ...
