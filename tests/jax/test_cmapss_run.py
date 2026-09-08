"""Tests for the CMAPSS end-to-end mean-stage training and evaluation."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_CMAPSS_DIR = Path(__file__).parents[2] / "examples" / "jax" / "cmapss"
if str(_CMAPSS_DIR) not in sys.path:
    sys.path.insert(0, str(_CMAPSS_DIR))

_RUN_PATH = _CMAPSS_DIR / "run.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_run", _RUN_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS run module.")
_RUN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUN
_SPEC.loader.exec_module(_RUN)


def _synthetic_windows(
    rng: np.random.Generator, *, n_windows: int, window_length: int, n_sensors: int
) -> tuple[np.ndarray, np.ndarray]:
    windows = rng.normal(size=(n_windows, window_length, n_sensors))
    targets = rng.uniform(low=1.0, high=100.0, size=n_windows)
    return windows, targets


def test_train_gamma_model_and_evaluate_composite_metrics_end_to_end() -> None:
    rng = np.random.default_rng(0)
    train_windows, train_targets = _synthetic_windows(
        rng, n_windows=64, window_length=30, n_sensors=9
    )
    validation_windows, validation_targets = _synthetic_windows(
        rng, n_windows=16, window_length=30, n_sensors=9
    )
    test_windows, test_rul = _synthetic_windows(
        rng, n_windows=10, window_length=30, n_sensors=9
    )
    config = _RUN.CmapssConfig(
        batch_size=16,
        hidden_channels=4,
        kernel_size=3,
        mean_epochs=3,
        variance_epochs=3,
        predictive_sample_count=32,
        seed=1,
    )

    mean_model, state = _RUN.train_mean_model(
        train_windows, train_targets, validation_windows, validation_targets, config
    )
    variance_model = _RUN.train_gamma_model(state, train_windows, train_targets, config)
    metrics = _RUN.evaluate_composite_metrics(
        mean_model, variance_model, test_windows, test_rul, config
    )

    assert "loss" not in metrics
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["coverage"])
    assert math.isfinite(metrics["point_crps"])
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["point_crps"] >= 0.0


def test_cmapss_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="window_length and batch_size"):
        _RUN.CmapssConfig(window_length=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        _RUN.CmapssConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        _RUN.CmapssConfig(mean_epochs=0)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        _RUN.CmapssConfig(variance_epochs=0)
    with pytest.raises(ValueError, match="predictive_sample_count"):
        _RUN.CmapssConfig(predictive_sample_count=0)


def _synthetic_trajectories(
    rng: np.random.Generator, *, unit_ids: range, n_cycles: int
) -> pd.DataFrame:
    """Build a CMAPSS-shaped DataFrame of equal-length synthetic trajectories.

    Args:
        rng: Random generator drawing the sensor readings.
        unit_ids: Unit IDs to generate one trajectory each for.
        n_cycles: Number of cycles per trajectory.

    Returns:
        A DataFrame with `unit_id`, `time_cycles`, and the example's sensor
        columns, one row per unit and cycle.
    """
    frames = []
    for unit_id in unit_ids:
        frame = pd.DataFrame(
            rng.normal(loc=float(unit_id), size=(n_cycles, len(_RUN._SENSOR_NAMES))),
            columns=_RUN._SENSOR_NAMES,
        )
        frame.insert(0, "unit_id", unit_id)
        frame.insert(1, "time_cycles", np.arange(1, n_cycles + 1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def offline_fd001(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve synthetic FD001 splits in place of the cached NASA archive."""
    rng = np.random.default_rng(7)
    train_data = _synthetic_trajectories(rng, unit_ids=range(1, 11), n_cycles=60)
    test_data = _synthetic_trajectories(rng, unit_ids=range(1, 6), n_cycles=45)
    test_rul = rng.uniform(low=1.0, high=100.0, size=5)
    monkeypatch.setattr(_RUN, "load_fd001_data", lambda: train_data)
    monkeypatch.setattr(_RUN, "load_fd001_test_data", lambda: test_data)
    monkeypatch.setattr(_RUN, "load_fd001_test_rul", lambda: test_rul)


@pytest.fixture
def short_window_config() -> "_RUN.CmapssConfig":
    """Configure a window short enough for the synthetic trajectories."""
    return _RUN.CmapssConfig(window_length=10)


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize("window_length", [5, 10, 30])
def test_prepared_data_windows_and_targets_are_aligned(window_length: int) -> None:
    config = _RUN.CmapssConfig(window_length=window_length)

    prepared = _RUN.prepare_cmapss_windows(config)

    n_sensors = len(_RUN._SENSOR_NAMES)
    assert prepared.train_windows.shape[1:] == (config.window_length, n_sensors)
    assert prepared.train_windows.shape[0] == prepared.train_targets.shape[0]
    assert (
        prepared.validation_windows.shape[0] == (prepared.validation_targets.shape[0])
    )
    assert prepared.test_windows.shape[0] == prepared.test_rul.shape[0]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_is_frozen(short_window_config: "_RUN.CmapssConfig") -> None:
    prepared = _RUN.prepare_cmapss_windows(short_window_config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.train_targets = prepared.train_targets  # type: ignore[misc]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_standardizes_every_subset_with_training_statistics(
    short_window_config: "_RUN.CmapssConfig",
) -> None:
    prepared = _RUN.prepare_cmapss_windows(short_window_config)

    stats = prepared.standardization
    assert stats.feature_columns == _RUN._SENSOR_NAMES
    # Statistics fitted on the training subset only: standardizing that
    # subset with its own statistics leaves it at zero mean and unit scale.
    standardized_train = prepared.train_trajectories[stats.feature_columns].to_numpy()
    assert np.allclose(standardized_train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(standardized_train.std(axis=0, ddof=0), 1.0, atol=1e-8)
    # The validation trajectories carry those same statistics, not their own.
    raw = _RUN.load_fd001_data()
    raw_validation = raw[
        raw["unit_id"].isin(prepared.validation_trajectories["unit_id"].unique())
    ]
    expected = _RUN.apply_standardization(raw_validation, stats)
    assert np.allclose(
        prepared.validation_trajectories[stats.feature_columns].to_numpy(),
        expected[stats.feature_columns].to_numpy(),
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_build_composite_model_matches_the_source_models_predictions(
    batch_size: int,
) -> None:
    window_length, n_sensors = 30, 9
    mean_model = _RUN.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    variance_model = _RUN.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(1))
    inputs = jax.random.normal(
        jax.random.key(2), (batch_size, window_length, n_sensors)
    )

    composite = _RUN.build_composite_model(mean_model, variance_model)
    prediction = composite(inputs)

    assert bool(jnp.allclose(prediction.loc, mean_model(inputs)))
    assert bool(jnp.allclose(prediction.scale, jnp.sqrt(variance_model(inputs).mean())))


def test_build_composite_model_clones_the_source_models() -> None:
    n_sensors = 9
    mean_model = _RUN.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    variance_model = _RUN.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(1))
    original_bias = jnp.asarray(mean_model.output.bias[...])

    composite = _RUN.build_composite_model(mean_model, variance_model)
    composite.mean_model.output.bias[...] = original_bias + 1.0

    assert composite.mean_model is not mean_model
    assert composite.variance_model is not variance_model
    assert bool(jnp.allclose(mean_model.output.bias[...], original_bias))


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize("plot_path_argv", [[], ["--plot-path", "curve.png"]])
def test_main_plots_the_rul_curves_after_printing_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plot_path_argv: list[str],
) -> None:
    if plot_path_argv:
        plot_path_argv = [plot_path_argv[0], str(tmp_path / plot_path_argv[1])]
    monkeypatch.setattr(sys, "argv", ["run.py", *plot_path_argv])
    fast_config = _RUN.CmapssConfig(
        window_length=10,
        batch_size=16,
        hidden_channels=2,
        kernel_size=3,
        mean_epochs=1,
        variance_epochs=1,
        predictive_sample_count=8,
    )
    monkeypatch.setattr(_RUN, "CmapssConfig", lambda: fast_config)
    calls: list[dict[str, object]] = []

    def record_plot(
        trajectories: pd.DataFrame,
        feature_columns: list[str],
        model: object,
        *,
        window_length: int,
        save_path: Path | None,
    ) -> None:
        """Record one plotting call and whatever was printed before it."""
        calls.append(
            {
                "trajectories": trajectories,
                "feature_columns": feature_columns,
                "model": model,
                "window_length": window_length,
                "save_path": save_path,
                "printed": capsys.readouterr().out,
            }
        )

    monkeypatch.setattr(_RUN, "plot_validation_rul_curves", record_plot)

    _RUN.main()

    (call,) = calls
    assert call["feature_columns"] == _RUN._SENSOR_NAMES
    assert call["window_length"] == 10
    assert isinstance(call["model"], _RUN.CompositeGaussianModel)
    expected_path = Path(plot_path_argv[1]) if plot_path_argv else None
    assert call["save_path"] == expected_path
    # The unit plotted is a held-out validation unit, and the metrics are
    # already printed by the time the figure is built.
    trajectories = call["trajectories"]
    assert isinstance(trajectories, pd.DataFrame)
    assert not trajectories.empty
    prepared = _RUN.prepare_cmapss_windows(fast_config)
    plotted_units = set(trajectories["unit_id"].unique())
    assert plotted_units == set(prepared.validation_trajectories["unit_id"].unique())
    assert plotted_units.isdisjoint(prepared.train_trajectories["unit_id"].unique())
    printed = call["printed"]
    assert isinstance(printed, str)
    assert "FD001 test RMSE" in printed
    assert "FD001 test point-CRPS" in printed
