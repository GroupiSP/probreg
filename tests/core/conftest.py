"""Test doubles shared by the core tests.

Session-scoped fixtures return classes rather than instances, so a test can build as
many doubles as it needs, with whatever arguments it needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class ExampleDistribution:
    """A two-example predictive distribution with ``log_prob(y) = -y**2``."""

    batch_shape = (2,)
    event_shape = ()

    def __init__(self, variance: Any = (1.0, 1.0)) -> None:
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


@pytest.fixture(scope="session")
def example_distribution() -> type[ExampleDistribution]:
    """The example predictive distribution class."""
    return ExampleDistribution
