"""RUL curves with 95% predictive intervals for the CMAPSS example.

Turns a trained composite Gaussian model and the standardized trajectories
it was trained under into the example's headline figure: three held-out
validation units side by side — the shortest-, median-, and
longest-lifetime one — each showing its true linear RUL across the whole
trajectory, the model's predicted mean, and the 95% predictive interval
around that mean. Spanning the lifetime range that way shows whether the
model's accuracy and its stated uncertainty behave consistently across
units that fail early and units that survive several times as long.

The predicted curve begins at each unit's first full window, because no
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Allow sibling-module imports (`preprocessing`) both when run as a script
# and when loaded from an arbitrary working directory, e.g. via `importlib`
# in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from preprocessing import build_unit_windows, select_lifetime_spanning_units

_INTERVAL_Z_SCORE = 1.96
_TRUE_RUL_LABEL = "True linear RUL"
_PREDICTED_MEAN_LABEL = "Predicted mean"
_INTERVAL_LABEL = "95% predictive interval"
# The role each column plays in the figure, in column order.
_ROLE_LABELS: tuple[str, ...] = (
    "shortest lifetime",
    "median lifetime",
    "longest lifetime",
)


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


def _draw_unit_rul_curve(
    axes: Axes,
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    model: PredictiveModel,
    *,
    unit_id: object,
    role: str,
    window_length: int,
) -> None:
    """Draw one unit's RUL curve and predictive band onto one column.

    Args:
        axes: The column to draw onto. Its limits are left to autoscale, so
            each column is scaled to its own unit and the band stays
            readable in the short-lived unit.
        trajectories: The standardized trajectories holding `unit_id`.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        model: The trained composite predictive model.
        unit_id: Identifier of the unit to draw.
        role: The role this unit plays in the figure, named in the title.
        window_length: Number of cycles per window, as trained with.
    """
    unit_trajectory = trajectories[trajectories["unit_id"] == unit_id].sort_values(
        "time_cycles"
    )
    unit_cycles = unit_trajectory["time_cycles"].to_numpy(dtype=float)
    lifetime = unit_cycles[-1]
    true_linear_rul = lifetime - unit_cycles

    unit = build_unit_windows(
        trajectories,
        feature_columns,
        unit_id=unit_id,
        window_length=window_length,
    )
    prediction = model(np.asarray(unit.windows, dtype=np.float32))
    loc = np.asarray(prediction.loc, dtype=float).reshape(-1)
    half_width = _INTERVAL_Z_SCORE * np.asarray(prediction.scale, dtype=float).reshape(
        -1
    )

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
    axes.set_title(f"Unit {unit_id} — {role} ({lifetime:g} cycles)")


def plot_validation_rul_curves(
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    model: PredictiveModel,
    *,
    window_length: int = 30,
    save_path: Path | None = None,
) -> Figure:
    """Plot RUL curves spanning the lifetime range of the validation subset.

    The three plotted units are the shortest-, median-, and
    longest-lifetime units of the held-out validation subset. Those units
    are never trained on, so the curves are evidence of generalization
    rather than of memorized fit; a unit's lifetime is only observable at
    all for the run-to-failure train-split trajectories the validation
    subset is carved out of. Each band is `loc ± 1.96 * scale`, taken
    analytically from the model's Gaussian and left unclipped at zero, so
    that a band dipping below zero stays visible as evidence of the
    Gaussian assumption breaking down near end of life.

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
        The figure holding the three columns, in shortest / median /
        longest lifetime order.
    """
    selected = select_lifetime_spanning_units(trajectories)

    # Unshared axes, so each column is scaled to its own unit and the band
    # stays readable in the short-lived one.
    figure, columns = plt.subplots(1, len(_ROLE_LABELS), figsize=(15.0, 4.5))
    plotted_units = (selected.shortest, selected.median, selected.longest)
    for axes, unit_id, role in zip(columns, plotted_units, _ROLE_LABELS, strict=True):
        _draw_unit_rul_curve(
            axes,
            trajectories,
            feature_columns,
            model,
            unit_id=unit_id,
            role=role,
            window_length=window_length,
        )

    # One legend for the whole figure: every column draws the same three
    # elements, so repeating a legend per column would say nothing new.
    handles, labels = columns[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels))
    figure.suptitle("CMAPSS FD001 held-out validation RUL curves")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    if save_path is not None:
        figure.savefig(Path(save_path))
    else:
        plt.show()
    return figure
