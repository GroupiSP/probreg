from __future__ import annotations

from typing import Any

import numpy as np

from probreg.core.distributions import (
    DistributionHead,
    Likelihood,
    Loss,
    PredictiveDistribution,
)
from probreg.core.types import Batch


class ExampleDistribution:
    batch_shape = (2,)
    event_shape = ()

    def log_prob(self, targets: Any) -> np.ndarray:
        return -(np.asarray(targets, dtype=float) ** 2)

    def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> np.ndarray:
        del key
        return np.zeros(sample_shape + self.batch_shape)

    def mean(self) -> np.ndarray:
        return np.zeros(self.batch_shape)

    def variance(self) -> np.ndarray:
        return np.ones(self.batch_shape)


def example_head(features: Any) -> ExampleDistribution:
    del features
    return ExampleDistribution()


def example_likelihood(prediction: PredictiveDistribution, targets: Any) -> np.ndarray:
    return prediction.log_prob(targets)


def example_loss(prediction: PredictiveDistribution, batch: Batch) -> np.ndarray:
    return -prediction.log_prob(batch.targets)


def test_distribution_protocols_support_distribution_valued_predictions() -> None:
    distribution: PredictiveDistribution = ExampleDistribution()
    head: DistributionHead = example_head
    likelihood: Likelihood = example_likelihood
    loss: Loss = example_loss

    assert head(np.ones(2)).variance().tolist() == [1.0, 1.0]
    assert likelihood(distribution, np.array([1, 2])).tolist() == [-1.0, -4.0]
    assert loss(distribution, Batch(inputs=[], targets=np.array([1, 2]))).tolist() == [
        1.0,
        4.0,
    ]
