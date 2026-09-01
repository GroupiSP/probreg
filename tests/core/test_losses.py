from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from probreg.core.distributions import PredictiveDistribution
from probreg.core.losses import BetaNLLLoss, GaussianNLLLoss
from probreg.core.types import Batch


class ExampleDistribution:
    batch_shape = (2,)
    event_shape = ()

    def __init__(self, variance: Any = 1.0) -> None:
        self._variance = np.asarray(variance, dtype=float)

    def log_prob(self, targets: Any) -> np.ndarray:
        return -(np.asarray(targets, dtype=float) ** 2)

    def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> np.ndarray:
        del key
        return np.zeros(sample_shape + self.batch_shape)

    def mean(self) -> np.ndarray:
        return np.zeros(self.batch_shape)

    def variance(self) -> np.ndarray:
        return self._variance


def test_gaussian_nll_loss_matches_negative_log_prob() -> None:
    loss = GaussianNLLLoss()
    distribution: PredictiveDistribution = ExampleDistribution()
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    assert loss.per_example(distribution, batch).tolist() == [1.0, 4.0]


def test_beta_nll_loss_zero_beta_matches_plain_nll() -> None:
    loss = BetaNLLLoss(beta=0.0)
    distribution: PredictiveDistribution = ExampleDistribution(variance=[2.0, 5.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    assert loss.per_example(distribution, batch).tolist() == [1.0, 4.0]


def test_beta_nll_loss_reweights_by_variance_power_beta() -> None:
    loss = BetaNLLLoss(beta=0.5)
    distribution: PredictiveDistribution = ExampleDistribution(variance=[4.0, 9.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    # nll = [1.0, 4.0]; weight = variance ** 0.5 = [2.0, 3.0]
    assert loss.per_example(distribution, batch).tolist() == [2.0, 12.0]


def test_beta_nll_loss_uses_provided_stop_gradient_hook() -> None:
    loss = BetaNLLLoss(beta=1.0, stop_gradient=lambda value: value * 0.0)
    distribution: PredictiveDistribution = ExampleDistribution(variance=[4.0, 9.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    # weight = stop_gradient(variance) ** 1 = 0, so the loss collapses to 0.
    assert loss.per_example(distribution, batch).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("invalid_beta", [-0.1, 1.1])
def test_beta_nll_loss_rejects_beta_outside_unit_interval(invalid_beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        BetaNLLLoss(beta=invalid_beta)
