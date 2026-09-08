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


def _spanning_units(shortest: int, median: int, longest: int) -> Any:
    """Name the three units a figure should draw, in column order."""
    return _PLOTS.LifetimeSpanningUnits(
        shortest=shortest, median=median, longest=longest
    )


@pytest.fixture
def captured_show(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record calls to the interactive show, instead of displaying anything."""
    calls: list[bool] = []
    monkeypatch.setattr(_PLOTS.plt, "show", lambda: calls.append(True))
    return calls


def _lines_by_label(axes: Any) -> dict[str, Any]:
    """Index an axes' lines by their legend label."""
    return {line.get_label(): line for line in axes.get_lines()}


def test_rul_curves_show_truth_prediction_and_band_per_column(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 30, 2: 12, 3: 20, 4: 25})
    window_length = 4

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(2, 3, 1),
        window_length=window_length,
    )

    assert captured_show == [True]
    assert len(figure.axes) == 3
    # One column per unit, in shortest / median / longest lifetime order:
    # unit 2 (12 cycles), unit 3 (20 cycles), and unit 1 (30 cycles).
    for axes, lifetime in zip(figure.axes, (12, 20, 30), strict=True):
        lines = _lines_by_label(axes)
        assert set(lines) == {
            _PLOTS._TRUE_RUL_LABEL,
            _PLOTS._PREDICTED_MEAN_LABEL,
        }
        truth = lines[_PLOTS._TRUE_RUL_LABEL]
        np.testing.assert_allclose(truth.get_xdata(), np.arange(1, lifetime + 1))
        np.testing.assert_allclose(truth.get_ydata(), np.arange(lifetime - 1, -1, -1))
        # The prediction starts one full window into the unit's life.
        predicted = lines[_PLOTS._PREDICTED_MEAN_LABEL]
        predicted_x = np.asarray(predicted.get_xdata())
        np.testing.assert_allclose(predicted_x, np.arange(window_length, lifetime + 1))
        np.testing.assert_allclose(predicted.get_ydata(), np.arange(len(predicted_x)))
        # A single filled band accompanies the predicted mean.
        assert len(axes.collections) == 1
        assert axes.get_xlabel().lower().startswith("time cycles")
        assert "rul" in axes.get_ylabel().lower()


def test_rul_curve_columns_have_independent_axis_limits(
    captured_show: list[bool],
) -> None:
    # A long-lived unit alongside a short-lived one: were the limits
    # shared, the short unit's curve would be squashed into a corner.
    trajectories = _standardized_trajectories({1: 12, 2: 40, 3: 120})

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(1, 2, 3),
        window_length=5,
    )

    shortest, median, longest = figure.axes
    assert shortest.get_xlim() != longest.get_xlim()
    assert shortest.get_ylim() != longest.get_ylim()
    assert median.get_xlim() not in (shortest.get_xlim(), longest.get_xlim())
    # Each column's x range covers its own unit's lifetime and no more.
    for axes, lifetime in zip(figure.axes, (12, 40, 120), strict=True):
        assert axes.get_xlim()[1] >= lifetime
        assert axes.get_xlim()[1] < 2 * lifetime


def test_rul_curve_column_titles_name_the_unit_role_and_lifetime(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({7: 30, 8: 12, 9: 20})

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(8, 9, 7),
        window_length=4,
    )

    titles = [axes.get_title() for axes in figure.axes]
    for title, unit_id, role, lifetime in zip(
        titles,
        (8, 9, 7),
        ("shortest", "median", "longest"),
        (12, 20, 30),
        strict=True,
    ):
        assert f"{unit_id}" in title
        assert role in title.lower()
        assert f"{lifetime}" in title
        assert "cycles" in title.lower()


def test_rul_curves_carry_one_shared_figure_legend(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 30, 2: 12, 3: 20})

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(2, 3, 1),
        window_length=4,
    )

    assert all(axes.get_legend() is None for axes in figure.axes)
    (legend,) = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        _PLOTS._TRUE_RUL_LABEL,
        _PLOTS._PREDICTED_MEAN_LABEL,
        _PLOTS._INTERVAL_LABEL,
    ]


def test_band_is_the_analytic_95_percent_predictive_interval(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 10})
    window_length = 3

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(scale=100.0),
        units=_spanning_units(1, 1, 1),
        window_length=window_length,
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


def test_rul_curves_window_each_unit_with_the_given_features(
    captured_show: list[bool],
) -> None:
    trajectories = _standardized_trajectories({1: 20, 2: 9, 3: 14})
    window_length = 3
    model = _StubModel()

    _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        model,
        units=_spanning_units(2, 3, 1),
        window_length=window_length,
    )

    assert len(model.seen_inputs) == 3
    for inputs, unit_id in zip(model.seen_inputs, (2, 3, 1), strict=True):
        expected = _PLOTS.build_unit_rul_curve(
            trajectories,
            _FEATURE_COLUMNS,
            unit_id=unit_id,
            window_length=window_length,
        ).windows
        np.testing.assert_allclose(inputs, expected)


def test_rul_curves_draw_exactly_the_units_the_caller_selected(
    captured_show: list[bool],
) -> None:
    # A deliberately non-spanning trio: were the selection made here rather
    # than by the caller, units 1 and 4 would be drawn instead.
    trajectories = _standardized_trajectories({1: 8, 2: 20, 3: 30, 4: 60})

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(2, 3, 2),
        window_length=4,
    )

    titles = [axes.get_title() for axes in figure.axes]
    for title, unit_id, lifetime in zip(titles, (2, 3, 2), (20, 30, 20), strict=True):
        assert f"Unit {unit_id}" in title
        assert f"{lifetime}" in title


def test_rul_curves_save_to_the_given_path(
    tmp_path: Path, captured_show: list[bool]
) -> None:
    trajectories = _standardized_trajectories({1: 15, 2: 20, 3: 25})
    save_path = tmp_path / "rul_curves.png"

    _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(),
        units=_spanning_units(1, 2, 3),
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

    figure = _PLOTS.plot_validation_rul_curves(
        trajectories,
        _FEATURE_COLUMNS,
        _StubModel(scale=scale),
        units=_spanning_units(1, 1, 1),
        window_length=window_length,
    )

    for axes in figure.axes:
        predicted = _lines_by_label(axes)[_PLOTS._PREDICTED_MEAN_LABEL]
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
