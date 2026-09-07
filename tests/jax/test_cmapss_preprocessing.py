"""Tests for the CMAPSS windowing and standardization preprocessing."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PREPROCESSING_PATH = (
    Path(__file__).parents[2] / "examples" / "jax" / "cmapss" / "preprocessing.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cmapss_preprocessing", _PREPROCESSING_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS preprocessing module.")
_PREPROCESSING = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREPROCESSING
_SPEC.loader.exec_module(_PREPROCESSING)


def _synthetic_data() -> pd.DataFrame:
    """Build a small synthetic multi-unit trajectory DataFrame."""
    rows = []
    for unit_id, n_cycles in ((1, 5), (2, 4), (3, 6)):
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "unit_id": unit_id,
                    "time_cycles": cycle,
                    "sensor_a": unit_id * 100.0 + cycle,
                    "sensor_b": unit_id * 10.0 + 0.5 * cycle,
                }
            )
    return pd.DataFrame(rows)


def test_split_by_unit_has_no_cross_unit_leakage() -> None:
    data = _synthetic_data()

    train_df, val_df = _PREPROCESSING.split_by_unit(
        data, test_size=0.34, random_state=0
    )

    train_units = set(train_df["unit_id"].unique())
    val_units = set(val_df["unit_id"].unique())

    assert train_units.isdisjoint(val_units)
    assert train_units | val_units == set(data["unit_id"].unique())
    assert len(train_df) + len(val_df) == len(data)


def test_fit_and_apply_standardization_uses_train_only_statistics() -> None:
    data = _synthetic_data()
    train_df = data[data["unit_id"] != 3].reset_index(drop=True)
    val_df = data[data["unit_id"] == 3].reset_index(drop=True)
    feature_columns = ["sensor_a", "sensor_b"]

    stats = _PREPROCESSING.fit_standardization(train_df, feature_columns)

    expected_mean = train_df[feature_columns].mean().to_numpy()
    expected_std = train_df[feature_columns].std(ddof=0).to_numpy()
    np.testing.assert_allclose(stats.mean, expected_mean)
    np.testing.assert_allclose(stats.scale, expected_std)
    assert stats.feature_columns == feature_columns

    applied_val = _PREPROCESSING.apply_standardization(val_df, stats)

    expected_val = (val_df[feature_columns].to_numpy() - expected_mean) / expected_std
    np.testing.assert_allclose(applied_val[feature_columns].to_numpy(), expected_val)
    # Sanity check: val's own statistics differ from train's, so a bug that
    # standardizes with val statistics would fail the assertion above.
    own_mean = val_df[feature_columns].mean().to_numpy()
    assert not np.allclose(own_mean, expected_mean)
    # Non-feature columns remain untouched.
    assert applied_val["unit_id"].tolist() == val_df["unit_id"].tolist()
    assert applied_val["time_cycles"].tolist() == val_df["time_cycles"].tolist()


def test_fit_standardization_guards_zero_variance_columns() -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "time_cycles": [1, 2, 3],
            "sensor_a": [5.0, 5.0, 5.0],
        }
    )

    stats = _PREPROCESSING.fit_standardization(data, ["sensor_a"])

    assert stats.scale[0] == 1.0


def test_build_windows_shapes_and_targets_for_long_trajectory() -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5],
            "sensor_a": [10.0, 11.0, 12.0, 13.0, 14.0],
            "sensor_b": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    feature_columns = ["sensor_a", "sensor_b"]

    windows, targets = _PREPROCESSING.build_windows(
        data, feature_columns, window_length=3, stride=1
    )

    # 5 cycles, window_length=3, stride=1 -> 3 windows.
    assert windows.shape == (3, 3, 2)
    assert targets.shape == (3,)

    # Unit max cycle is 5. First window covers cycles 1-3, last cycle=3.
    np.testing.assert_allclose(windows[0], [[10.0, 0.1], [11.0, 0.2], [12.0, 0.3]])
    assert targets[0] == 5 - 3

    # Second window covers cycles 2-4, last cycle=4.
    np.testing.assert_allclose(windows[1], [[11.0, 0.2], [12.0, 0.3], [13.0, 0.4]])
    assert targets[1] == 5 - 4

    # Third window covers cycles 3-5, last cycle=5.
    np.testing.assert_allclose(windows[2], [[12.0, 0.3], [13.0, 0.4], [14.0, 0.5]])
    assert targets[2] == 5 - 5


def test_build_windows_unsorted_input_is_sorted_by_time_cycles() -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "time_cycles": [3, 1, 2],
            "sensor_a": [30.0, 10.0, 20.0],
        }
    )

    windows, targets = _PREPROCESSING.build_windows(
        data, ["sensor_a"], window_length=3, stride=1
    )

    assert windows.shape == (1, 3, 1)
    np.testing.assert_allclose(windows[0], [[10.0], [20.0], [30.0]])
    assert targets[0] == 3 - 3


def test_build_windows_pads_short_trajectory_and_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [7, 7],
            "time_cycles": [1, 2],
            "sensor_a": [50.0, 60.0],
        }
    )

    with caplog.at_level(logging.INFO):
        windows, targets = _PREPROCESSING.build_windows(
            data, ["sensor_a"], window_length=3, stride=1
        )

    # Padded to exactly window_length rows -> exactly one window.
    assert windows.shape == (1, 3, 1)
    assert targets.shape == (1,)

    # Padding repeats the first observed cycle's feature values backward.
    np.testing.assert_allclose(windows[0], [[50.0], [50.0], [60.0]])

    # Padding never changes the true RUL of the real last cycle: unit's own
    # max time_cycles is 2, and the window's last real cycle is 2.
    assert targets[0] == 2 - 2

    padded_cycles = 3 - 2
    assert any(
        "7" in record.message and str(padded_cycles) in record.message
        for record in caplog.records
        if record.levelno == logging.INFO
    )


def test_build_windows_concatenates_multiple_units_deterministically() -> None:
    data = _synthetic_data()
    feature_columns = ["sensor_a", "sensor_b"]

    windows, targets = _PREPROCESSING.build_windows(
        data, feature_columns, window_length=3, stride=1
    )

    # unit 1: 5 cycles -> 3 windows; unit 2: 4 cycles -> 2 windows;
    # unit 3: 6 cycles -> 4 windows.
    assert windows.shape == (9, 3, 2)
    assert targets.shape == (9,)


def test_build_last_windows_takes_trailing_cycles_per_unit() -> None:
    data = _synthetic_data()
    feature_columns = ["sensor_a", "sensor_b"]

    windows = _PREPROCESSING.build_last_windows(data, feature_columns, window_length=3)

    # 3 units -> one trailing window each.
    assert windows.shape == (3, 3, 2)
    # unit 1 has cycles 1-5; the trailing window covers cycles 3-5.
    np.testing.assert_allclose(
        windows[0], [[103.0, 11.5], [104.0, 12.0], [105.0, 12.5]]
    )
    # unit 3 has cycles 1-6; the trailing window covers cycles 4-6.
    np.testing.assert_allclose(
        windows[2], [[304.0, 32.0], [305.0, 32.5], [306.0, 33.0]]
    )


def test_build_last_windows_pads_short_trajectory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [7, 7],
            "time_cycles": [1, 2],
            "sensor_a": [50.0, 60.0],
        }
    )

    with caplog.at_level(logging.INFO):
        windows = _PREPROCESSING.build_last_windows(data, ["sensor_a"], window_length=3)

    assert windows.shape == (1, 3, 1)
    np.testing.assert_allclose(windows[0], [[50.0], [50.0], [60.0]])
