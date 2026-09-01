from __future__ import annotations

import numpy as np
import pytest

from probreg.core.metric_registry import (
    IntervalCoverage,
    MetricInputs,
    RootMeanSquaredError,
    WeightedSpread,
)
from probreg.core.metrics import coverage, rmse, wsu


def test_rmse_adapter_matches_core_function() -> None:
    data = MetricInputs(
        targets=np.array([1.0, 2.0, 3.0]),
        mean=np.array([1.5, 1.5, 3.0]),
    )
    metric = RootMeanSquaredError()

    assert metric(data) == pytest.approx(rmse(data.targets, data.mean))


def test_interval_coverage_adapter_uses_explicit_bounds() -> None:
    data = MetricInputs(
        targets=np.array([0.0, 1.0, 2.0]),
        mean=np.array([0.1, 1.1, 1.9]),
        lower=np.array([-1.0, 0.5, 1.5]),
        upper=np.array([0.5, 1.5, 2.5]),
    )
    metric = IntervalCoverage(level=0.8)

    assert metric(data) == pytest.approx(coverage(data.targets, data.lower, data.upper))


def test_interval_coverage_adapter_can_derive_bounds_from_variance() -> None:
    data = MetricInputs(
        targets=np.array([0.0, 1.0, 2.0]),
        mean=np.array([0.0, 1.0, 2.0]),
        variance=np.array([0.25, 0.25, 0.25]),
    )
    metric = IntervalCoverage(level=0.95)

    assert metric(data) == pytest.approx(1.0)


def test_weighted_spread_adapter_uses_metadata_coordinate() -> None:
    data = MetricInputs(
        targets=np.array([0.0, 0.0, 0.0, 0.0]),
        mean=np.array([0.0, 0.0, 0.0, 0.0]),
        lower=np.array([0.0, 1.0, 1.0, 2.0]),
        upper=np.array([2.0, 3.0, 5.0, 6.0]),
        metadata={"x": np.array([0.0, 1.0, 2.0, 4.0])},
    )
    metric = WeightedSpread()

    assert metric(data) == pytest.approx(
        wsu(data.lower, data.upper, data.metadata["x"])
    )


def test_metric_adapters_reject_missing_required_inputs() -> None:
    data = MetricInputs(
        targets=np.array([0.0]),
        mean=np.array([0.0]),
    )

    with pytest.raises(
        ValueError, match="requires lower/upper bounds or predictive variance"
    ):
        IntervalCoverage()(data)
    with pytest.raises(ValueError, match="requires lower and upper"):
        WeightedSpread()(data)


def test_interval_coverage_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        IntervalCoverage(level=1.2)
