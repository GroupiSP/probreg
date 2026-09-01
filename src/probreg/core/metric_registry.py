"""Backend-neutral metric registration contracts and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from probreg.core.metrics import coverage, rmse, wsu
from probreg.core.types import Batch


@dataclass(frozen=True, slots=True)
class MetricInputs:
    """Materialized arrays consumed by host-side epoch metrics.

    Attributes:
        targets: Target values.
        mean: Predictive means.
        variance: Predictive variances when available.
        lower: Lower predictive interval bound when available.
        upper: Upper predictive interval bound when available.
        metadata: Optional metric-specific auxiliary values.
    """

    targets: NDArray[np.float64]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64] | None = None
    lower: NDArray[np.float64] | None = None
    upper: NDArray[np.float64] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EpochMetric(Protocol):
    """A host-side metric computed from materialized epoch inputs."""

    @property
    def name(self) -> str:
        """Return the metric name used in emitted metric mappings."""
        ...

    def __call__(self, data: MetricInputs, /) -> float:
        """Compute a metric value from materialized epoch inputs."""
        ...


@runtime_checkable
class SupportsMetricInputs(Protocol):
    """Protocol for models that can decode a batch into metric inputs."""

    def produce_metric_inputs(self, batch: Batch, /) -> MetricInputs:
        """Return metric inputs for a single batch."""
        ...


def _as_vector(
    values: NDArray[np.float64] | object, *, name: str
) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return vector


@dataclass(frozen=True, slots=True)
class RootMeanSquaredError:
    """Epoch metric adapter for :func:`probreg.core.metrics.rmse`."""

    metric_name: str = "rmse"

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    def __call__(self, data: MetricInputs, /) -> float:
        """Compute RMSE from epoch targets and predicted means.

        Args:
            data: Materialized metric inputs.

        Returns:
            The root mean squared error.
        """
        return rmse(
            _as_vector(data.targets, name="targets"),
            _as_vector(data.mean, name="mean"),
        )


@dataclass(frozen=True, slots=True)
class IntervalCoverage:
    """Epoch metric adapter for :func:`probreg.core.metrics.coverage`."""

    level: float = 0.95
    metric_name: str = "coverage"

    def __post_init__(self) -> None:
        """Validate interval coverage configuration.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
        """
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be between 0 and 1.")

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    def __call__(self, data: MetricInputs, /) -> float:
        """Compute interval coverage at ``level``.

        Args:
            data: Materialized metric inputs.

        Returns:
            The observed interval coverage.

        Raises:
            ValueError: If neither explicit bounds nor variances are available.
        """
        if data.lower is not None and data.upper is not None:
            lower = _as_vector(data.lower, name="lower")
            upper = _as_vector(data.upper, name="upper")
        else:
            if data.variance is None:
                raise ValueError(
                    "IntervalCoverage requires lower/upper bounds or predictive variance."
                )
            variance = _as_vector(data.variance, name="variance")
            if np.any(variance < 0.0):
                raise ValueError("variance must be non-negative.")
            z = NormalDist().inv_cdf((1.0 + self.level) / 2.0)
            mean = _as_vector(data.mean, name="mean")
            sigma = np.sqrt(variance)
            lower = mean - z * sigma
            upper = mean + z * sigma
        return coverage(_as_vector(data.targets, name="targets"), lower, upper)


@dataclass(frozen=True, slots=True)
class WeightedSpread:
    """Epoch metric adapter for :func:`probreg.core.metrics.wsu`."""

    coordinate_key: str = "x"
    metric_name: str = "wsu"

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    def __call__(self, data: MetricInputs, /) -> float:
        """Compute weighted spread of uncertainty intervals.

        Args:
            data: Materialized metric inputs.

        Returns:
            The normalized weighted spread.

        Raises:
            ValueError: If required interval bounds or coordinate metadata are missing.
        """
        if data.lower is None or data.upper is None:
            raise ValueError("WeightedSpread requires lower and upper interval bounds.")
        if self.coordinate_key not in data.metadata:
            raise ValueError(
                f"WeightedSpread requires metadata[{self.coordinate_key!r}] coordinates."
            )
        return wsu(
            _as_vector(data.lower, name="lower"),
            _as_vector(data.upper, name="upper"),
            _as_vector(data.metadata[self.coordinate_key], name=self.coordinate_key),
        )
