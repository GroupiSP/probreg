"""Numerical, NumPy-only metrics for probabilistic regression."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_nonempty_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Internal input validation utility."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return vector


def _matching_vectors(
    *named_values: tuple[str, ArrayLike],
) -> tuple[NDArray[np.float64], ...]:
    vectors = tuple(
        _as_nonempty_vector(values, name=name) for name, values in named_values
    )
    if len({vector.shape for vector in vectors}) != 1:
        names = ", ".join(name for name, _ in named_values)
        raise ValueError(f"{names} must have matching shapes.")
    return vectors


def cdf(x: float, samples: ArrayLike) -> float:
    """Return the empirical cumulative distribution at ``x``."""
    sample_vector = np.asarray(samples, dtype=float)
    if sample_vector.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample_vector.size == 0:
        return 0.0
    return float(np.mean(sample_vector <= x))


def crps(samples_true: ArrayLike, samples_pred: ArrayLike, x_grid: ArrayLike) -> float:
    """Approximate CRPS between empirical true and predicted distributions."""
    grid = _as_nonempty_vector(x_grid, name="x_grid")
    true_cdf = np.array([cdf(x, samples_true) for x in grid])
    predicted_cdf = np.array([cdf(x, samples_pred) for x in grid])
    return float(np.trapezoid((true_cdf - predicted_cdf) ** 2, grid))


def point_crps(y_true: float, samples_pred: ArrayLike, x_grid: ArrayLike) -> float:
    """Approximate CRPS between a point target and empirical predictions."""
    grid = _as_nonempty_vector(x_grid, name="x_grid")
    true_cdf = np.where(grid < y_true, 0.0, 1.0)
    predicted_cdf = np.array([cdf(x, samples_pred) for x in grid])
    return float(np.trapezoid((true_cdf - predicted_cdf) ** 2, grid))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared error."""
    target, prediction = _matching_vectors(("y_true", y_true), ("y_pred", y_pred))
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Return the fraction of targets within inclusive prediction intervals."""
    target, lower_bound, upper_bound = _matching_vectors(
        ("y_true", y_true), ("lower", lower), ("upper", upper)
    )
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower must not exceed upper.")
    return float(np.mean((lower_bound <= target) & (target <= upper_bound)))


def wsu(lower: ArrayLike, upper: ArrayLike, x: ArrayLike) -> float:
    """Return the normalized weighted spread of uncertainty intervals.

    The calculation preserves QMoDeM's established WSU definition while using a
    generic, strictly increasing coordinate rather than battery-specific time.
    """
    lower_bound, upper_bound, coordinate = _matching_vectors(
        ("lower", lower), ("upper", upper), ("x", x)
    )
    if coordinate.size < 3:
        raise ValueError("x, lower, and upper must contain at least three values.")
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower must not exceed upper.")
    if np.any(np.diff(coordinate) <= 0):
        raise ValueError("x must be strictly increasing.")

    interval_widths = (upper_bound[2:] + upper_bound[1:-1]) / 2 - (
        lower_bound[2:] + lower_bound[1:-1]
    ) / 2
    period = coordinate[-1] - coordinate[0]
    return float(np.dot(interval_widths, coordinate[1:-1] - coordinate[0]) / period**2)
