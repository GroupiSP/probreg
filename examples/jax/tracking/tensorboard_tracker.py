"""A TensorBoard [`ExperimentTracker`][probreg.core.ExperimentTracker].

This is the module to copy into your own project. It is backend-neutral:
its surface accepts Python floats, strings and Matplotlib figures, never
anything JAX-shaped, so the same tracker serves any `probreg` backend and
its mapping logic is testable with neither `tensorboardX` nor JAX
installed. Converting backend arrays to floats belongs to the run script,
which is already the backend-specific layer.

The writer is a constructor argument. `tensorboardX` is imported only when
the default writer is built, which is what keeps this module importable
without the optional dependency.

Swapping TensorBoard for MLflow, Weights & Biases or Aim means rewriting
this file and nothing else: the run script is typed against
[`ExperimentTracker`][probreg.core.ExperimentTracker].
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from probreg.core.naming import flatten_parameters


class SummaryWriter(Protocol):
    """The subset of TensorBoard's summary-writer surface used here."""

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int | None = None
    ) -> None: ...

    def add_figure(
        self, tag: str, figure: Any, global_step: int | None = None
    ) -> None: ...

    def add_text(
        self, tag: str, text_string: str, global_step: int | None = None
    ) -> None: ...

    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float],
        name: str | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def create_summary_writer(logdir: str | Path) -> SummaryWriter:
    """Build a `tensorboardX` summary writer for one run's directory.

    Args:
        logdir: Directory the run's event files are written to.

    Returns:
        A summary writer writing to `logdir`.

    Raises:
        ImportError: If `tensorboardX` is not installed.
    """
    from tensorboardX import SummaryWriter as TensorBoardXSummaryWriter

    return TensorBoardXSummaryWriter(logdir=str(logdir))


def _to_hparam_value(value: Any) -> bool | int | float | str:
    """Convert a parameter value to one the HParams plugin can store.

    TensorBoard's HParams plugin stores only scalars and strings, so any
    other value is stringified.

    Args:
        value: A leaf value of a parameter mapping.

    Returns:
        `value` unchanged if it is a scalar or string, else its string form.
    """
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _is_figure(value: Any) -> bool:
    """Report whether a value is a Matplotlib figure.

    Matplotlib is imported here rather than at module level so that a
    tracker logging only scalars and text needs neither the import nor
    the dependency.

    Args:
        value: The value to classify.

    Returns:
        `True` if Matplotlib is installed and `value` is one of its
        figures, `False` otherwise.
    """
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return False
    return isinstance(value, Figure)


class TensorBoardTracker:
    """Record a run's parameters, metrics and artifacts to TensorBoard.

    Metrics become scalar summaries, parameters go through the HParams plugin so that
    runs are comparable in a sortable table, and artifacts are dispatched on their type.
    """

    def __init__(
        self, logdir: str | Path, *, writer: SummaryWriter | None = None
    ) -> None:
        """Initialize the tracker.

        Args:
            logdir: Directory this run's event files are written to.
                Ignored when `writer` is given.
            writer: The summary writer receiving every summary. Defaults
                to a `tensorboardX` writer on `logdir`; pass one to
                redirect the summaries, as the tests do.
        """
        self._writer = create_summary_writer(logdir) if writer is None else writer

    def log_params(self, values: Mapping[str, Any]) -> None:
        """Record a run's parameters through the HParams plugin.

        Each leaf becomes one HParams entry keyed by its parameter path,
        e.g. ``data/train/samples``.

        Args:
            values: The run's parameters, a nested mapping of bare
                ``snake_case`` leaf keys.

        Returns:
            None.

        Raises:
            ValueError: If a key, at any depth, is empty or contains ``/``.
        """
        # `name="."` keeps the HParams session in this run's own directory.
        # Left to its own default, `tensorboardX` opens a second writer on a
        # time-named subdirectory, which TensorBoard then reads as a separate
        # run: one run with the scalars and no hyperparameters, another with
        # the hyperparameters and no metric columns to sort by.
        hparams = {
            path: _to_hparam_value(value)
            for path, value in flatten_parameters(values).items()
        }
        self._writer.add_hparams(hparams, {}, name=".")

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        """Record one scalar summary per metric at the given step.

        Args:
            values: Metric values keyed by the tag to record them under.
                Anything convertible with `float` is accepted, including
                the zero-dimensional arrays a backend's metrics arrive as,
                so no backend type is ever imported here.
            step: The step the metrics belong to, TensorBoard's x axis.

        Returns:
            None.
        """
        for tag, value in values.items():
            self._writer.add_scalar(tag, float(value), step)

    def log_artifact(self, name: str, value: Any) -> None:
        """Record an artifact, dispatching on its type.

        Args:
            name: The tag to record the artifact under.
            value: A string, recorded as a text summary, or a Matplotlib
                figure, recorded as an image summary.

        Returns:
            None.

        Raises:
            TypeError: If `value` is neither a string nor a Matplotlib
                figure.
        """
        if isinstance(value, str):
            self._writer.add_text(name, value)
        elif _is_figure(value):
            self._writer.add_figure(name, value)
        else:
            raise TypeError(
                f"cannot log artifact {name!r} of type "
                f"{type(value).__name__!r}: expected a string or a "
                "Matplotlib figure."
            )

    def flush(self) -> None:
        """Flush pending summaries to disk.

        Returns:
            None.
        """
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the writer, ending the run.

        Returns:
            None.
        """
        self._writer.close()
