from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from probreg.core.metric_registry import (
    ContinuousRankedProbabilityScore,
    EpochPredictionData,
    EvaluationGrid,
    IntervalCoverage,
    PointContinuousRankedProbabilityScore,
    PredictionInterval,
    RootMeanSquaredError,
    WeightedSpread,
)
from probreg.core.metrics import coverage, crps, point_crps, rmse, wsu


def test_rmse_adapter_matches_core_function() -> None:
    data = EpochPredictionData(
        targets=np.array([1.0, 2.0, 3.0]),
        mean=np.array([1.5, 1.5, 3.0]),
    )

    assert RootMeanSquaredError()(data) == pytest.approx(rmse(data.targets, data.mean))


def test_interval_coverage_selects_explicit_labelled_level() -> None:
    selected = PredictionInterval(
        level=0.8,
        lower=np.array([-1.0, 0.5, 1.5]),
        upper=np.array([0.5, 1.5, 2.5]),
    )
    data = EpochPredictionData(
        targets=np.array([0.0, 1.0, 2.0]),
        mean=np.array([0.1, 1.1, 1.9]),
        variance=np.array([100.0, 100.0, 100.0]),
        intervals=(
            PredictionInterval(0.95, np.full(3, -10.0), np.full(3, 10.0)),
            selected,
        ),
    )

    assert IntervalCoverage(level=0.8)(data) == pytest.approx(
        coverage(data.targets, selected.lower, selected.upper)
    )


def test_weighted_spread_uses_explicit_coordinate_and_interval() -> None:
    interval = PredictionInterval(
        level=0.9,
        lower=np.array([0.0, 1.0, 1.0, 2.0]),
        upper=np.array([2.0, 3.0, 5.0, 6.0]),
    )
    coordinate = np.array([0.0, 1.0, 2.0, 4.0])
    data = EpochPredictionData(
        targets=np.zeros(4),
        mean=np.zeros(4),
        intervals=(interval,),
        coordinate=coordinate,
    )

    assert WeightedSpread(level=0.9)(data) == pytest.approx(
        wsu(interval.lower, interval.upper, coordinate)
    )


def test_point_crps_scores_each_scalar_unit_before_averaging() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 4.0, 61))
    targets = np.array([0.0, 2.0])
    samples = np.array([[-0.5, 0.0, 0.5], [1.0, 2.0, 3.0]])
    data = EpochPredictionData(
        targets=targets,
        mean=np.mean(samples, axis=1),
        predictive_samples=samples,
        evaluation_grid=grid,
    )

    expected = np.mean(
        [point_crps(target, row, grid.values) for target, row in zip(targets, samples)]
    )
    assert PointContinuousRankedProbabilityScore()(data) == pytest.approx(expected)


def test_conditional_crps_scores_paired_distributions_before_averaging() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 4.0, 61))
    reference = np.array([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    predictive = np.array([[-0.5, 0.0, 0.5], [2.0, 3.0, 4.0]])
    data = EpochPredictionData(
        targets=np.array([0.0, 2.0]),
        mean=np.mean(predictive, axis=1),
        predictive_samples=predictive,
        reference_samples=reference,
        evaluation_grid=grid,
    )

    expected = np.mean(
        [crps(true, pred, grid.values) for true, pred in zip(reference, predictive)]
    )
    assert ContinuousRankedProbabilityScore()(data) == pytest.approx(expected)


def test_adapters_declare_materialization_requirements() -> None:
    coverage_requirements = IntervalCoverage(level=0.8).requirements
    spread_requirements = WeightedSpread(level=0.9).requirements
    point_requirements = PointContinuousRankedProbabilityScore().requirements
    conditional_requirements = ContinuousRankedProbabilityScore().requirements

    assert coverage_requirements.interval_levels == frozenset({0.8})
    assert spread_requirements.coordinate
    assert spread_requirements.interval_levels == frozenset({0.9})
    assert point_requirements.predictive_samples
    assert point_requirements.evaluation_grid
    assert conditional_requirements.reference_samples


def test_epoch_prediction_data_rejects_non_scalar_axes_and_invalid_fields() -> None:
    with pytest.raises(ValueError, match="one-dimensional scalar scoring units"):
        EpochPredictionData(targets=np.ones((2, 1)), mean=np.ones((2, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        EpochPredictionData(
            targets=np.array([0.0]),
            mean=np.array([0.0]),
            variance=np.array([-1.0]),
        )
    with pytest.raises(ValueError, match="one row"):
        EpochPredictionData(
            targets=np.array([0.0, 1.0]),
            mean=np.array([0.0, 1.0]),
            predictive_samples=np.ones((1, 3)),
        )


def test_missing_required_typed_fields_fail_clearly() -> None:
    data = EpochPredictionData(targets=np.array([0.0]), mean=np.array([0.0]))

    with pytest.raises(ValueError, match="interval at level"):
        IntervalCoverage()(data)
    with pytest.raises(ValueError, match="explicit coordinate"):
        WeightedSpread()(data)
    with pytest.raises(ValueError, match="predictive_samples"):
        PointContinuousRankedProbabilityScore()(data)


def test_grid_and_intervals_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        EvaluationGrid(np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        PredictionInterval(1.0, np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="must not exceed"):
        PredictionInterval(0.9, np.array([2.0]), np.array([1.0]))


@given(st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=30))
def test_rmse_is_zero_for_equal_finite_vectors(values: list[float]) -> None:
    vector = np.asarray(values)
    data = EpochPredictionData(targets=vector, mean=vector)

    assert RootMeanSquaredError()(data) == pytest.approx(0.0)


def test_crps_adapters_are_non_negative_and_match_reference_expectation() -> None:
    grid = EvaluationGrid(np.linspace(-3.0, 3.0, 101))
    samples = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    data = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        predictive_samples=samples,
        reference_samples=samples.copy(),
        evaluation_grid=grid,
    )

    expected = np.mean(
        [
            np.mean([point_crps(target, row, grid.values) for target in reference])
            for reference, row in zip(samples, samples, strict=True)
        ]
    )
    assert ContinuousRankedProbabilityScore()(data) == pytest.approx(expected)
    assert PointContinuousRankedProbabilityScore()(data) >= 0.0


@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=30))
def test_interval_coverage_is_bounded(values: list[float]) -> None:
    targets = np.asarray(values)
    data = EpochPredictionData(
        targets=targets,
        mean=targets,
        intervals=(
            PredictionInterval(
                0.95,
                targets - 1.0,
                targets + 1.0,
            ),
        ),
    )

    observed = IntervalCoverage()(data)
    assert 0.0 <= observed <= 1.0
