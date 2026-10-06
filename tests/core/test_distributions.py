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


def example_likelihood(prediction: PredictiveDistribution, targets: Any) -> np.ndarray:
    return prediction.log_prob(targets)


def example_loss(prediction: PredictiveDistribution, batch: Batch) -> np.ndarray:
    return -prediction.log_prob(batch.targets)


def test_distribution_protocols_support_distribution_valued_predictions(
    example_distribution: type[Any],
) -> None:
    def example_head(features: Any) -> PredictiveDistribution:
        del features
        return example_distribution()

    distribution: PredictiveDistribution = example_distribution()
    head: DistributionHead = example_head
    likelihood: Likelihood = example_likelihood
    loss: Loss = example_loss

    assert head(np.ones(2)).variance().tolist() == [1.0, 1.0]
    assert likelihood(distribution, np.array([1, 2])).tolist() == [-1.0, -4.0]
    assert loss(distribution, Batch(inputs=[], targets=np.array([1, 2]))).tolist() == [
        1.0,
        4.0,
    ]
