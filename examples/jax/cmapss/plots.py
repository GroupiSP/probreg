"""RUL curves with 95% predictive intervals for the CMAPSS example.

Turns a trained composite Gaussian model and the standardized trajectories
it was trained under into the example's headline figure: for one held-out
validation unit, its true linear RUL across the whole trajectory, the
model's predicted mean, and the 95% predictive interval around that mean.

The predicted curve begins at the unit's first full window, because no
prediction exists before `window_length` cycles of sensor history have
accumulated. The true-RUL line spans the whole trajectory regardless, so
the length of that warm-up stays visible rather than hidden.

This module knows nothing about training or about the archive: it takes
already-standardized trajectories and an already-trained model.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Allow sibling-module imports (`preprocessing`) both when run as a script
# and when loaded from an arbitrary working directory, e.g. via `importlib`
# in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from preprocessing import build_unit_windows

_INTERVAL_Z_SCORE = 1.96
_TRUE_RUL_LABEL = "True linear RUL"
_PREDICTED_MEAN_LABEL = "Predicted mean"
_INTERVAL_LABEL = "95% predictive interval"


class PredictiveDistribution(Protocol):
    """A location-scale predictive distribution over a batch of windows."""

    @property
    def loc(self) -> Any:
        """The predictive mean, shape `(n_windows, 1)`."""
        ...

    @property
    def scale(self) -> Any:
        """The predictive standard deviation, shape `(n_windows, 1)`."""
        ...


class PredictiveModel(Protocol):
    """A model mapping windowed sensor readings to a predictive distribution."""

    def __call__(self, inputs: Any) -> PredictiveDistribution:
        """Predict one distribution per windowed example.

        Args:
            inputs: Windowed sensor readings, shape `(n_windows,
                window_length, n_features)`.

        Returns:
            The predictive distribution over those windows.
        """
        ...


def _shortest_lifetime_unit(trajectories: pd.DataFrame) -> object:
    """Find the unit whose run-to-failure lifetime is the shortest.

    A stand-in for the shortest/median/longest lifetime selection that will
    live in the preprocessing module once more than one unit is plotted.

    Args:
        trajectories: Run-to-failure trajectories with `unit_id` and
            `time_cycles` columns.

    Returns:
        The `unit_id` with the smallest maximum `time_cycles`. Ties are
        broken by the smaller unit ID, so the choice is deterministic.

    Raises:
        ValueError: If `trajectories` holds no rows.
    """
    if trajectories.empty:
        raise ValueError("trajectories must hold at least one unit.")
    lifetimes = trajectories.groupby("unit_id")["time_cycles"].max()
    return lifetimes.sort_index().idxmin()


def plot_validation_rul_curve(
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    model: PredictiveModel,
    *,
    window_length: int = 30,
    save_path: Path | None = None,
) -> Figure:
    """Plot the RUL curve of the shortest-lifetime held-out validation unit.

    That unit is never trained on, so the curve is evidence of
    generalization rather than of memorized fit. The band is
    `loc ± 1.96 * scale`, taken analytically from the model's Gaussian and
    left unclipped at zero, so that a band dipping below zero stays visible
    as evidence of the Gaussian assumption breaking down near end of life.

    Args:
        trajectories: The standardized validation-subset trajectories,
            already scaled with the statistics fitted on the training
            subset only.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        model: The trained composite predictive model.
        window_length: Number of cycles per window, as trained with.
        save_path: Where to save the figure. When omitted, the figure is
            displayed instead.

    Returns:
        The figure holding the plotted curve.
    """
    unit_id = _shortest_lifetime_unit(trajectories)
    unit_trajectory = trajectories[trajectories["unit_id"] == unit_id].sort_values(
        "time_cycles"
    )
    unit_cycles = unit_trajectory["time_cycles"].to_numpy(dtype=float)
    true_linear_rul = unit_cycles[-1] - unit_cycles

    unit = build_unit_windows(
        trajectories,
        feature_columns,
        unit_id=unit_id,
        window_length=window_length,
    )
    prediction = model(np.asarray(unit.windows, dtype=np.float32))
    loc = np.asarray(prediction.loc, dtype=float).reshape(-1)
    scale = np.asarray(prediction.scale, dtype=float).reshape(-1)
    half_width = _INTERVAL_Z_SCORE * scale

    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.plot(unit_cycles, true_linear_rul, label=_TRUE_RUL_LABEL, color="black")
    axes.plot(unit.cycles, loc, label=_PREDICTED_MEAN_LABEL, color="tab:blue")
    axes.fill_between(
        unit.cycles,
        loc - half_width,
        loc + half_width,
        alpha=0.25,
        color="tab:blue",
        label=_INTERVAL_LABEL,
    )
    axes.set_xlabel("Time cycles")
    axes.set_ylabel("RUL (cycles)")
    axes.set_title(f"CMAPSS FD001 validation unit {unit_id} RUL curve")
    axes.legend()
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(Path(save_path))
    else:
        plt.show()
    return figure
