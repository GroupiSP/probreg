"""Tests for the CMAPSS RUL-curve plots."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytest.importorskip("matplotlib")

_CMAPSS_DIR = Path(__file__).parents[2] / "examples" / "jax" / "cmapss"
if str(_CMAPSS_DIR) not in sys.path:
    sys.path.insert(0, str(_CMAPSS_DIR))

_PLOTS_PATH = _CMAPSS_DIR / "plots.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_plots", _PLOTS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS plots module.")
_PLOTS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PLOTS
_SPEC.loader.exec_module(_PLOTS)

_FEATURE_COLUMNS = ["sensor_a", "sensor_b"]


class _StubGaussian:
    """A minimal stand-in for the composite model's Gaussian output."""

    def __init__(self, loc: np.ndarray, scale: np.ndarray) -> None:
        """Store the predictive mean and standard deviation.

        Args:
            loc: Predictive mean, shape `(n_windows, 1)`.
            scale: Predictive standard deviation, shape `(n_windows, 1)`.
        """
        self.loc = loc
        self.scale = scale


class _StubModel:
    """A composite-model stand-in predicting a constant-scale Gaussian."""

    def __init__(self, *, scale: float = 4.0) -> None:
        """Fix the predictive scale and start recording the inputs seen.

        Args:
            scale: The constant predictive standard deviation to return.
        """
        self.scale = scale
        self.seen_inputs: list[np.ndarray] = []

    def __call__(self, inputs: Any) -> _StubGaussian:
        """Predict a Gaussian whose mean counts up over the batch.

        Args:
            inputs: Windowed sensor readings, shape `(n_windows,
                window_length, n_features)`.

        Returns:
            A Gaussian with `loc = [0, 1, ..., n_windows - 1]` and the
            fixed scale.
        """
        array = np.asarray(inputs)
        self.seen_inputs.append(array)
        n_windows = array.shape[0]
        loc = np.arange(n_windows, dtype=float).reshape(-1, 1)
        return _StubGaussian(loc=loc, scale=np.full((n_windows, 1), self.scale))


def _standardized_trajectories(lifetimes: dict[int, int]) -> pd.DataFrame:
    """Build synthetic standardized trajectories with the given lifetimes."""
    rows = []
    for unit_id, n_cycles in lifetimes.items():
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "unit_id": unit_id,
                    "time_cycles": cycle,
                    "sensor_a": 0.01 * unit_id + 0.1 * cycle,
                    "sensor_b": -0.02 * unit_id - 0.05 * cycle,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def captured_show(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record calls to the interactive show, instead of displaying anything."""
    calls: list[bool] = []
    monkeypatch.setattr(_PLOTS.plt, "show", lambda: calls.append(True))
    return calls


def test_rul_curve_shows_truth_prediction_and_band(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 30, 2: 12})
    window_length = 4
    model = _StubModel()

    figure = _PLOTS.plot_validation_rul_curve(
        trajectories,
        _FEATURE_COLUMNS,
        model,
        window_length=window_length,
    )

    assert captured_show == [True]
    axes = figure.axes[0]
    lines = {line.get_label(): line for line in axes.get_lines()}
    assert len(lines) == 2
    truth = lines[_PLOTS._TRUE_RUL_LABEL]
    predicted = lines[_PLOTS._PREDICTED_MEAN_LABEL]
    # The shortest-lifetime unit is unit 2, of 12 cycles.
    np.testing.assert_allclose(truth.get_xdata(), np.arange(1, 13))
    np.testing.assert_allclose(truth.get_ydata(), np.arange(11, -1, -1))
    # The prediction starts one full window into the unit's life.
    predicted_x = np.asarray(predicted.get_xdata())
    assert predicted_x[0] == float(window_length)
    np.testing.assert_allclose(predicted_x, np.arange(window_length, 13))
    np.testing.assert_allclose(predicted.get_ydata(), np.arange(len(predicted_x)))
    # A single filled band accompanies the predicted mean.
    assert len(axes.collections) == 1
    assert axes.get_xlabel().lower().startswith("time cycles")
    assert "rul" in axes.get_ylabel().lower()
    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend_labels == [
        _PLOTS._TRUE_RUL_LABEL,
        _PLOTS._PREDICTED_MEAN_LABEL,
        _PLOTS._INTERVAL_LABEL,
    ]


def test_band_is_the_analytic_95_percent_predictive_interval(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 10})
    window_length = 3
    model = _StubModel(scale=100.0)

    figure = _PLOTS.plot_validation_rul_curve(
        trajectories, _FEATURE_COLUMNS, model, window_length=window_length
    )

    band = figure.axes[0].collections[0]
    vertices = band.get_paths()[0].vertices
    n_windows = 10 - window_length + 1
    loc = np.arange(n_windows, dtype=float)
    expected_lower = loc - 1.96 * 100.0
    expected_upper = loc + 1.96 * 100.0
    # The band is left unclipped at zero, so its lower edge stays negative.
    assert vertices[:, 1].min() == pytest.approx(expected_lower.min())
    assert vertices[:, 1].max() == pytest.approx(expected_upper.max())
    assert expected_lower.min() < 0.0


def test_rul_curve_windows_the_unit_with_the_given_features(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 20, 2: 9})
    window_length = 3
    model = _StubModel()

    _PLOTS.plot_validation_rul_curve(
        trajectories, _FEATURE_COLUMNS, model, window_length=window_length
    )

    (inputs,) = model.seen_inputs
    expected = _PLOTS.build_unit_windows(
        trajectories, _FEATURE_COLUMNS, unit_id=2, window_length=window_length
    ).windows
    np.testing.assert_allclose(inputs, expected)


def test_rul_curve_saves_to_the_given_path(
    tmp_path: Path, captured_show: list[bool]
) -> None:
    trajectories = _standardized_trajectories({1: 15})
    save_path = tmp_path / "rul_curve.png"

    _PLOTS.plot_validation_rul_curve(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        window_length=5,
        save_path=save_path,
    )

    assert save_path.is_file()
    assert save_path.stat().st_size > 0
    assert captured_show == []


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_cycles=st.integers(min_value=2, max_value=40),
    window_length=st.integers(min_value=1, max_value=40),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
def test_band_is_always_the_predicted_mean_plus_minus_1_96_scale(
    monkeypatch: pytest.MonkeyPatch,
    n_cycles: int,
    window_length: int,
    scale: float,
) -> None:
    monkeypatch.setattr(_PLOTS.plt, "show", lambda: None)
    window_length = 1 + (window_length - 1) % n_cycles
    trajectories = _standardized_trajectories({1: n_cycles})

    figure = _PLOTS.plot_validation_rul_curve(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(scale=scale),
        window_length=window_length,
    )

    axes = figure.axes[0]
    predicted = axes.get_lines()[1]
    predicted_y = np.asarray(predicted.get_ydata(), dtype=float)
    # One prediction per cycle from the first full window onwards, drawn
    # against a cycle axis that starts exactly one window length in.
    assert predicted_y.shape == (n_cycles - window_length + 1,)
    assert np.asarray(predicted.get_xdata())[0] == float(window_length)
    # The band is symmetric about the predicted mean and exactly 1.96
    # scales wide on each side, whatever the scale.
    vertices = axes.collections[0].get_paths()[0].vertices
    assert vertices[:, 1].max() == pytest.approx(
        predicted_y.max() + 1.96 * scale, rel=1e-6
    )
    assert vertices[:, 1].min() == pytest.approx(
        predicted_y.min() - 1.96 * scale, rel=1e-6
    )
    _PLOTS.plt.close(figure)
