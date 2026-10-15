from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stagereg.core.metrics import (
    cdf,
    coverage,
    crps,
    expected_sample_crps,
    point_crps,
    rmse,
    sample_crps,
    wsu,
)

_finite_values = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


def test_cdf_and_crps_match_empirical_definitions() -> None:
    samples_true = np.array([0.0, 1.0, 2.0])
    samples_pred = np.array([1.0, 2.0, 3.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0])

    assert cdf(1.0, samples_true) == pytest.approx(2 / 3)
    assert cdf(1.0, np.array([])) == 0.0
    assert crps(samples_true, samples_pred, grid) == pytest.approx(11 / 18)
    assert point_crps(1.0, samples_pred, grid) == pytest.approx(5 / 9)


def _exact_empirical_crps(target: float, samples: np.ndarray) -> float:
    """Integrate ``(F(x) - 1{x >= target})**2`` exactly over its step pieces."""
    points = np.sort(np.append(samples, target))
    total = 0.0
    for left, right in zip(points[:-1], points[1:], strict=True):
        empirical = np.mean(samples <= left)
        step = 1.0 if left >= target else 0.0
        total += (empirical - step) ** 2 * (right - left)
    return total


def test_sample_crps_of_one_sample_is_the_absolute_error() -> None:
    assert sample_crps(1.0, [3.5]) == pytest.approx(2.5)
    assert sample_crps(1.0, [1.0, 2.0, 3.0]) == pytest.approx(5 / 9)


@given(
    target=_finite_values,
    samples=st.lists(_finite_values, min_size=1, max_size=20),
)
def test_sample_crps_is_the_exact_integral_of_the_empirical_cdf(
    target: float, samples: list[float]
) -> None:
    sample_vector = np.asarray(samples)

    expected = _exact_empirical_crps(target, sample_vector)
    assert sample_crps(target, sample_vector) == pytest.approx(expected, abs=1e-6)


@given(
    target=_finite_values,
    samples=st.lists(_finite_values, min_size=1, max_size=20),
    shift=_finite_values,
)
def test_sample_crps_is_non_negative_and_shift_invariant(
    target: float, samples: list[float], shift: float
) -> None:
    sample_vector = np.asarray(samples)
    score = sample_crps(target, sample_vector)

    assert score >= 0.0
    assert sample_crps(target + shift, sample_vector + shift) == pytest.approx(
        score, abs=1e-6
    )


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
        (sample_crps, (0.0, [])),
        (sample_crps, (float("nan"), [1.0])),
        (sample_crps, (0.0, [[1.0]])),
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


@st.composite
def _sample_matrices(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(units, S)`` predictive and ``(units, R)`` reference samples."""
    units = draw(st.integers(1, 4))
    predictive_count = draw(st.integers(1, 8))
    reference_count = draw(st.integers(1, 8))
    predictive = draw(
        st.lists(
            _finite_values,
            min_size=units * predictive_count,
            max_size=units * predictive_count,
        )
    )
    reference = draw(
        st.lists(
            _finite_values,
            min_size=units * reference_count,
            max_size=units * reference_count,
        )
    )
    return (
        np.reshape(predictive, (units, predictive_count)),
        np.reshape(reference, (units, reference_count)),
    )


@given(samples=_sample_matrices())
def test_expected_sample_crps_is_the_mean_sample_crps_over_reference_samples(
    samples: tuple[np.ndarray, np.ndarray],
) -> None:
    predictive, reference = samples

    expected = [
        np.mean([sample_crps(target, row) for target in targets])
        for row, targets in zip(predictive, reference, strict=True)
    ]

    assert expected_sample_crps(predictive, reference) == pytest.approx(
        expected, abs=1e-6
    )


@given(samples=_sample_matrices())
def test_expected_sample_crps_is_non_negative(
    samples: tuple[np.ndarray, np.ndarray],
) -> None:
    assert np.all(expected_sample_crps(*samples) >= 0.0)


@given(
    samples=_sample_matrices(),
    shift=_finite_values,
    seed=st.integers(0, 2**32 - 1),
)
def test_expected_sample_crps_is_permutation_and_translation_invariant(
    samples: tuple[np.ndarray, np.ndarray], shift: float, seed: int
) -> None:
    predictive, reference = samples
    rng = np.random.default_rng(seed)
    score = expected_sample_crps(predictive, reference)

    permuted = expected_sample_crps(
        rng.permuted(predictive, axis=1), rng.permuted(reference, axis=1)
    )
    shifted = expected_sample_crps(predictive + shift, reference + shift)

    assert permuted == pytest.approx(score, abs=1e-6)
    assert shifted == pytest.approx(score, abs=1e-6)


@given(samples=_sample_matrices(), scale=st.floats(1e-3, 1e3))
def test_expected_sample_crps_scales_linearly_under_positive_scaling(
    samples: tuple[np.ndarray, np.ndarray], scale: float
) -> None:
    predictive, reference = samples

    assert expected_sample_crps(scale * predictive, scale * reference) == pytest.approx(
        scale * expected_sample_crps(predictive, reference), rel=1e-6, abs=1e-6
    )


@given(samples=_sample_matrices())
def test_expected_sample_crps_scores_batched_rows_like_single_rows(
    samples: tuple[np.ndarray, np.ndarray],
) -> None:
    predictive, reference = samples

    single_rows = [
        expected_sample_crps(row[None, :], targets[None, :])[0]
        for row, targets in zip(predictive, reference, strict=True)
    ]

    assert expected_sample_crps(predictive, reference) == pytest.approx(single_rows)


def test_expected_sample_crps_of_a_point_reference_is_sample_crps() -> None:
    assert expected_sample_crps([[1.0, 2.0, 3.0]], [[1.0]]) == pytest.approx([5 / 9])


@pytest.mark.parametrize(
    ("predictive_samples", "reference_samples"),
    [
        (np.empty((0, 2)), np.empty((0, 2))),
        ([[]], [[1.0]]),
        ([[1.0]], [[]]),
        ([[np.nan]], [[1.0]]),
        ([[1.0]], [[np.inf]]),
        ([1.0, 2.0], [[1.0]]),
        ([[1.0]], [[[1.0]]]),
        ([[1.0], [2.0]], [[1.0]]),
    ],
)
def test_expected_sample_crps_rejects_malformed_samples(
    predictive_samples: object, reference_samples: object
) -> None:
    with pytest.raises(ValueError):
        expected_sample_crps(predictive_samples, reference_samples)
