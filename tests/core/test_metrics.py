from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probreg.core.metrics import cdf, coverage, crps, point_crps, rmse, wsu


def test_cdf_and_crps_match_qmodem_definitions() -> None:
    samples_true = np.array([0.0, 1.0, 2.0])
    samples_pred = np.array([1.0, 2.0, 3.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0])

    assert cdf(1.0, samples_true) == pytest.approx(2 / 3)
    assert cdf(1.0, np.array([])) == 0.0
    assert crps(samples_true, samples_pred, grid) == pytest.approx(5 / 18)
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
