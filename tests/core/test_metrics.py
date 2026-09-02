from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probreg.core.metrics import cdf, coverage, crps, point_crps, rmse, wsu


def test_cdf_and_crps_match_empirical_definitions() -> None:
    samples_true = np.array([0.0, 1.0, 2.0])
    samples_pred = np.array([1.0, 2.0, 3.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0])

    assert cdf(1.0, samples_true) == pytest.approx(2 / 3)
    assert cdf(1.0, np.array([])) == 0.0
    assert crps(samples_true, samples_pred, grid) == pytest.approx(11 / 18)
    assert point_crps(1.0, samples_pred, grid) == pytest.approx(5 / 9)


def test_rmse_and_coverage_are_domain_agnostic() -> None:
    target = np.array([1.0, 2.0, 3.0])
    prediction = np.array([2.0, 2.0, 2.0])

    assert rmse(target, prediction) == pytest.approx(np.sqrt(2 / 3))
    assert coverage(target, [0.5, 2.0, 3.5], [1.0, 2.5, 4.0]) == pytest.approx(2 / 3)


def test_wsu_matches_the_qmodem_test_case_formula() -> None:
    coordinate = np.array([0.0, 1.0, 2.0, 4.0])
    lower = np.array([0.0, 1.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 5.0, 6.0])

    expected = (
        np.dot(
            np.array([(5.0 + 3.0) / 2, (6.0 + 5.0) / 2])
            - np.array([(1.0 + 1.0) / 2, (2.0 + 1.0) / 2]),
            coordinate[1:-1] - coordinate[0],
        )
        / (coordinate[-1] - coordinate[0]) ** 2
    )

    assert wsu(lower, upper, coordinate) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (rmse, ([], [])),
        (coverage, ([1], [2], [1])),
        (wsu, ([0, 0], [1, 1], [0, 1])),
        (wsu, ([0, 0, 0], [1, 1, 1], [0, 2, 1])),
    ],
)
def test_metrics_reject_invalid_inputs(
    function: Callable[..., float], arguments: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError):
        function(*arguments)


def test_crps_matches_average_point_crps_over_reference_samples() -> None:
    first = np.array([-1.0, 0.0, 2.0])
    second = np.array([-0.5, 1.0, 3.0])
    grid = np.linspace(-4.0, 4.0, 101)

    expected = np.mean([point_crps(value, second, grid) for value in first])

    assert crps(first, second, grid) == pytest.approx(expected)
    assert crps(first, second, grid) >= 0.0


def test_point_crps_is_translation_invariant_with_shifted_grid() -> None:
    target = 0.25
    samples = np.array([-1.0, 0.5, 2.0])
    grid = np.linspace(-3.0, 3.0, 121)
    shift = 7.5

    assert point_crps(target, samples, grid) == pytest.approx(
        point_crps(target + shift, samples + shift, grid + shift)
    )


@pytest.mark.parametrize(
    ("samples_true", "samples_pred", "grid"),
    [
        ([], [0.0], [0.0, 1.0]),
        ([0.0], [], [0.0, 1.0]),
        ([0.0], [0.0], [0.0]),
        ([0.0], [0.0], [0.0, 0.0]),
        ([np.nan], [0.0], [0.0, 1.0]),
    ],
)
def test_crps_rejects_malformed_samples_and_grids(
    samples_true: object, samples_pred: object, grid: object
) -> None:
    with pytest.raises(ValueError):
        crps(samples_true, samples_pred, grid)
