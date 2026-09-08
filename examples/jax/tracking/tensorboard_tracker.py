"""A TensorBoard :class:`~probreg.core.tracking.ExperimentTracker`.

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
:class:`~probreg.core.tracking.ExperimentTracker`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

PARAMETER_SEPARATOR = "/"
"""Separator joining the key path of a nested parameter into a flat name."""


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
        self, hparam_dict: dict[str, Any], metric_dict: dict[str, float]
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


def flatten_parameters(
    values: Mapping[str, Any], *, separator: str = PARAMETER_SEPARATOR
) -> dict[str, bool | int | float | str]:
    """Flatten a nested parameter mapping into HParams-compatible entries.

    TensorBoard's HParams plugin accepts only flat scalar and string
    values, so nested mappings are joined into one key per leaf and any
    value the plugin cannot carry is stringified. Keys may not contain the
    separator: allowing them would let two distinct nested key paths
    collapse onto one flat name and silently overwrite each other.

    Args:
        values: The parameters to flatten, possibly nested.
        separator: String joining the segments of a nested key path.

    Returns:
        One entry per leaf, keyed by its separator-joined key path.

    Raises:
        ValueError: If a key contains the separator.
    """

    def walk(
        mapping: Mapping[str, Any], prefix: str
    ) -> Iterator[tuple[str, bool | int | float | str]]:
        for key, value in mapping.items():
            if separator in key:
                raise ValueError(
                    f"parameter key {key!r} may not contain {separator!r}: "
                    "it would collide with a nested key path."
                )
            name = f"{prefix}{key}"
            if isinstance(value, Mapping):
                yield from walk(value, f"{name}{separator}")
            elif isinstance(value, bool | int | float | str):
                yield name, value
            else:
                yield name, str(value)

    return dict(walk(values, ""))


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

        Args:
            values: The run's parameters, possibly nested. Nested keys
                are flattened onto one HParams entry per leaf.

        Returns:
            None.

        Raises:
            ValueError: If a key contains
                :data:`PARAMETER_SEPARATOR`.
        """
        self._writer.add_hparams(flatten_parameters(values), {})

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        """Record one scalar summary per metric at the given step.

        Args:
            values: Metric values keyed by the tag to record them under.
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
