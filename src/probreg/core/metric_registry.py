"""Typed host-side epoch metric contracts and numerical adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from probreg.core.metrics import coverage, crps, point_crps, rmse, wsu

FloatArray = NDArray[np.float64]


def _vector(values: object, *, name: str) -> FloatArray:
    """Return a validated, immutable C-contiguous vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional scalar scoring units.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _matrix(values: object, *, name: str) -> FloatArray:
    """Return a validated, immutable C-contiguous scoring-unit matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_scoring_units, n_draws).")
    if not all(array.shape):
        raise ValueError(f"{name} axes must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EvaluationGrid:
    """A shared scalar CRPS integration grid.

    Attributes:
        values: Finite, strictly increasing integration coordinates.
    """

    values: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize grid values.

        Raises:
            ValueError: If the grid has fewer than two points or is not increasing.
        """
        values = _vector(self.values, name="evaluation_grid")
        if values.size < 2:
            raise ValueError("evaluation_grid must contain at least two values.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("evaluation_grid must be strictly increasing.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """Host-resident bounds labelled by their central confidence level."""

    level: float
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Validate interval level, bounds, and scoring-unit shape.

        Raises:
            ValueError: If the level or bounds are invalid.
        """
        if not np.isfinite(self.level) or not 0.0 < self.level < 1.0:
            raise ValueError("interval level must be finite and between 0 and 1.")
        lower = _vector(self.lower, name="interval.lower")
        upper = _vector(self.upper, name="interval.upper")
        if lower.shape != upper.shape:
            raise ValueError(
                "interval lower and upper bounds must have matching shapes."
            )
        if np.any(lower > upper):
            raise ValueError("interval lower bounds must not exceed upper bounds.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class EpochPredictionData:
    """Typed, host-materialized scalar predictions for one batch or epoch.

    Rows are independent scalar scoring units. Sample matrices use rows for scoring
    units and columns for draws; non-scalar distribution events are intentionally not
    represented by this contract.
    """

    targets: FloatArray
    mean: FloatArray
    variance: FloatArray | None = None
    predictive_samples: FloatArray | None = None
    reference_samples: FloatArray | None = None
    intervals: tuple[PredictionInterval, ...] = ()
    coordinate: FloatArray | None = None
    evaluation_grid: EvaluationGrid | None = None

    def __post_init__(self) -> None:
        """Validate all fields against the common scoring-unit axis.

        Raises:
            ValueError: If a field is non-finite, malformed, or shape-incompatible.
        """
        targets = _vector(self.targets, name="targets")
        mean = _vector(self.mean, name="mean")
        if targets.shape != mean.shape:
            raise ValueError("targets and mean must have matching scoring-unit shapes.")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "mean", mean)
        n_units = targets.size

        if self.variance is not None:
            variance = _vector(self.variance, name="variance")
            if variance.shape != targets.shape:
                raise ValueError("variance must match the scoring-unit shape.")
            if np.any(variance < 0.0):
                raise ValueError("variance must be non-negative.")
            object.__setattr__(self, "variance", variance)

        for field_name in ("predictive_samples", "reference_samples"):
            values = getattr(self, field_name)
            if values is None:
                continue
            samples = _matrix(values, name=field_name)
            if samples.shape[0] != n_units:
                raise ValueError(
                    f"{field_name} must have one row per scalar scoring unit."
                )
            object.__setattr__(self, field_name, samples)

        intervals = tuple(self.intervals)
        levels: set[float] = set()
        for interval in intervals:
            if not isinstance(interval, PredictionInterval):
                raise TypeError("intervals must contain PredictionInterval values.")
            if interval.lower.size != n_units:
                raise ValueError("interval bounds must match the scoring-unit shape.")
            if interval.level in levels:
                raise ValueError(f"duplicate interval level {interval.level}.")
            levels.add(interval.level)
        object.__setattr__(self, "intervals", intervals)

        if self.evaluation_grid is not None and not isinstance(
            self.evaluation_grid, EvaluationGrid
        ):
            raise TypeError("evaluation_grid must be an EvaluationGrid.")

        if self.coordinate is not None:
            coordinate = _vector(self.coordinate, name="coordinate")
            if coordinate.shape != targets.shape:
                raise ValueError("coordinate must match the scoring-unit shape.")
            object.__setattr__(self, "coordinate", coordinate)

    def interval(self, level: float) -> PredictionInterval:
        """Return the prediction interval registered at exactly ``level``.

        Args:
            level: Required central confidence level.

        Returns:
            The matching labelled interval.

        Raises:
            ValueError: If no interval at ``level`` is present.
        """
        for interval in self.intervals:
            if interval.level == level:
                return interval
        raise ValueError(f"prediction interval at level {level} is required.")


@dataclass(frozen=True, slots=True)
class MetricRequirements:
    """Materialized fields required by one or more epoch metrics."""

    variance: bool = False
    predictive_samples: bool = False
    reference_samples: bool = False
    interval_levels: frozenset[float] = frozenset()
    coordinate: bool = False
    evaluation_grid: bool = False

    def __post_init__(self) -> None:
        """Validate requested interval levels.

        Raises:
            ValueError: If an interval level is outside ``(0, 1)``.
        """
        if any(
            not np.isfinite(level) or not 0.0 < level < 1.0
            for level in self.interval_levels
        ):
            raise ValueError(
                "required interval levels must be finite and between 0 and 1."
            )

    def union(self, other: MetricRequirements) -> MetricRequirements:
        """Combine this requirement set with another.

        Args:
            other: Additional requirements.

        Returns:
            A requirement set containing the union of both inputs.
        """
        return MetricRequirements(
            variance=self.variance or other.variance,
            predictive_samples=self.predictive_samples or other.predictive_samples,
            reference_samples=self.reference_samples or other.reference_samples,
            interval_levels=self.interval_levels | other.interval_levels,
            coordinate=self.coordinate or other.coordinate,
            evaluation_grid=self.evaluation_grid or other.evaluation_grid,
        )


class EpochMetric(Protocol):
    """A host-side scalar metric over typed epoch prediction data."""

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        ...

    @property
    def requirements(self) -> MetricRequirements:
        """Return fields that must be materialized for this metric."""
        ...

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute the scalar metric value."""
        ...


@dataclass(frozen=True, slots=True)
class RootMeanSquaredError:
    """Epoch adapter for root mean squared error."""

    metric_name: str = "rmse"

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    @property
    def requirements(self) -> MetricRequirements:
        """Return the base target/mean requirements."""
        return MetricRequirements()

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute RMSE from scalar targets and predictive means."""
        return rmse(data.targets, data.mean)


@dataclass(frozen=True, slots=True)
class IntervalCoverage:
    """Observed coverage of an explicitly labelled prediction interval."""

    level: float = 0.95
    metric_name: str = "coverage"

    def __post_init__(self) -> None:
        """Validate the requested confidence level.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
        """
        MetricRequirements(interval_levels=frozenset({self.level}))

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    @property
    def requirements(self) -> MetricRequirements:
        """Require an interval at the configured level."""
        return MetricRequirements(interval_levels=frozenset({self.level}))

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute inclusive interval coverage."""
        interval = data.interval(self.level)
        return coverage(data.targets, interval.lower, interval.upper)


@dataclass(frozen=True, slots=True)
class WeightedSpread:
    """Weighted spread for an interval and explicit numeric coordinate."""

    level: float = 0.95
    metric_name: str = "wsu"

    def __post_init__(self) -> None:
        """Validate the requested confidence level.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
        """
        MetricRequirements(interval_levels=frozenset({self.level}))

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    @property
    def requirements(self) -> MetricRequirements:
        """Require labelled bounds and a coordinate."""
        return MetricRequirements(
            interval_levels=frozenset({self.level}), coordinate=True
        )

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute normalized weighted interval spread.

        Raises:
            ValueError: If the required coordinate is absent.
        """
        if data.coordinate is None:
            raise ValueError("WeightedSpread requires an explicit coordinate.")
        interval = data.interval(self.level)
        order = np.argsort(data.coordinate)
        return wsu(
            interval.lower[order],
            interval.upper[order],
            data.coordinate[order],
        )


@dataclass(frozen=True, slots=True)
class PointContinuousRankedProbabilityScore:
    """Mean point-CRPS across independent scalar scoring units."""

    metric_name: str = "point_crps"

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    @property
    def requirements(self) -> MetricRequirements:
        """Require predictive draws and the shared evaluation grid."""
        return MetricRequirements(predictive_samples=True, evaluation_grid=True)

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute one point-CRPS per unit, then average units equally.

        Raises:
            ValueError: If predictive samples or the grid are absent.
        """
        if data.predictive_samples is None:
            raise ValueError("Point CRPS requires predictive_samples.")
        if data.evaluation_grid is None:
            raise ValueError("Point CRPS requires an evaluation_grid.")
        scores = [
            point_crps(target, samples, data.evaluation_grid.values)
            for target, samples in zip(
                data.targets, data.predictive_samples, strict=True
            )
        ]
        return float(np.mean(scores))


@dataclass(frozen=True, slots=True)
class ContinuousRankedProbabilityScore:
    """Mean expected CRPS under per-unit empirical reference distributions."""

    metric_name: str = "crps"

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        return self.metric_name

    @property
    def requirements(self) -> MetricRequirements:
        """Require predictive/reference draws and the shared grid."""
        return MetricRequirements(
            predictive_samples=True,
            reference_samples=True,
            evaluation_grid=True,
        )

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute expected empirical-reference CRPS and average scoring units.

        Raises:
            ValueError: If either sample matrix or the grid is absent.
        """
        if data.predictive_samples is None:
            raise ValueError("CRPS requires predictive_samples.")
        if data.reference_samples is None:
            raise ValueError("CRPS requires reference_samples.")
        if data.evaluation_grid is None:
            raise ValueError("CRPS requires an evaluation_grid.")
        scores = [
            crps(reference, predictive, data.evaluation_grid.values)
            for reference, predictive in zip(
                data.reference_samples, data.predictive_samples, strict=True
            )
        ]
        return float(np.mean(scores))
