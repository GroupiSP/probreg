from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from probreg.core.distributions import PredictiveDistribution
from probreg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
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


class RecordingLogDistribution(ExampleDistribution):
    def __init__(self) -> None:
        super().__init__()
        self.targets: Any = None

    def log_prob(self, targets: Any) -> np.ndarray:
        self.targets = targets
        return np.log(np.asarray(targets, dtype=float))


def test_negative_log_likelihood_matches_negative_log_prob() -> None:
    loss = NegativeLogLikelihoodLoss()
    distribution: PredictiveDistribution = ExampleDistribution()
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    assert loss.per_example(distribution, batch).tolist() == [1.0, 4.0]


def test_negative_log_likelihood_applies_target_transform() -> None:
    loss = NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6))
    distribution = RecordingLogDistribution()
    batch = Batch(inputs=[], targets=np.array([0.0, 2.0]))

    values = loss.per_example(distribution, batch)

    assert np.allclose(distribution.targets, [1e-6, 2.000001])
    assert np.allclose(values, -np.log([1e-6, 2.000001]))


def test_negative_log_likelihood_requires_targets() -> None:
    with pytest.raises(ValueError, match="targets"):
        NegativeLogLikelihoodLoss().per_example(
            ExampleDistribution(),
            Batch(inputs=[]),
        )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.inf, np.nan])
def test_add_epsilon_rejects_invalid_offset(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        add_epsilon(epsilon)


def test_zero_beta_matches_plain_negative_log_likelihood() -> None:
    loss = NegativeLogLikelihoodLoss(beta=0.0)
    distribution: PredictiveDistribution = ExampleDistribution(variance=[2.0, 5.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    assert loss.per_example(distribution, batch).tolist() == [1.0, 4.0]


def test_negative_log_likelihood_reweights_by_variance_power_beta() -> None:
    loss = NegativeLogLikelihoodLoss(beta=0.5)
    distribution: PredictiveDistribution = ExampleDistribution(variance=[4.0, 9.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    # nll = [1.0, 4.0]; weight = variance ** 0.5 = [2.0, 3.0]
    assert loss.per_example(distribution, batch).tolist() == [2.0, 12.0]


def test_negative_log_likelihood_uses_stop_gradient_hook() -> None:
    loss = NegativeLogLikelihoodLoss(
        beta=1.0,
        stop_gradient=lambda value: value * 0.0,
    )
    distribution: PredictiveDistribution = ExampleDistribution(variance=[4.0, 9.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 2.0]))

    # weight = stop_gradient(variance) ** 1 = 0, so the loss collapses to 0.
    assert loss.per_example(distribution, batch).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("invalid_beta", [-0.1, 1.1])
def test_negative_log_likelihood_rejects_invalid_beta(invalid_beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        NegativeLogLikelihoodLoss(beta=invalid_beta)


def test_squared_error_loss_returns_per_example_errors() -> None:
    loss = SquaredErrorLoss()
    prediction = np.array([2.0, 4.0])
    batch = Batch(inputs=[], targets=np.array([1.0, 6.0]))

    assert loss.per_example(prediction, batch).tolist() == [1.0, 4.0]


def test_squared_error_loss_requires_targets() -> None:
    with pytest.raises(ValueError, match="targets"):
        SquaredErrorLoss().per_example(np.array([1.0]), Batch(inputs=[]))
