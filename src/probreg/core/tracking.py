"""Training-event and experiment-tracking protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from probreg.core.naming import Split, metric_tag
from probreg.core.types import TrainingState


@dataclass(frozen=True)
class Decision:
    """The measurement a decision event judged.

    Attributes:
        metric: Bare name of the monitored metric.
        value: The metric's value at the step the decision was made.
    """

    metric: str
    value: float


@dataclass(frozen=True)
class TrainingEvent:
    """A structured event emitted during staged training.

    Attributes:
        name: The point in the stage's lifecycle the event marks, such as
            ``epoch_end`` or ``best_model``.
        stage: Name of the stage that emitted the event.
        split: Split the event concerns: the split its metrics were
            measured on, or, for a decision event, the split of the metric
            the decision was made on.
        iteration: Outer iteration of the staged workflow.
        step: Step of the stage the event belongs to.
        metrics: Metrics measured at this point, keyed by bare metric
            name. Empty for a decision event.
        state: The live training state.
        decision: The measurement a decision event judged, or ``None``
            for an event that reports measurements.
    """

    name: str
    stage: str
    split: Split
    iteration: int
    step: int
    metrics: Mapping[str, float]
    state: TrainingState
    decision: Decision | None = None


class EventSink(Protocol):
    """Consumes training events."""

    def on_event(self, event: TrainingEvent) -> None: ...


class ExperimentTracker(Protocol):
    """Records parameters, metrics, and artifacts for an experiment."""

    def log_params(self, values: Mapping[str, Any]) -> None:
        """Record a run's hyperparameters.

        Args:
            values: A nested mapping whose leaf keys are bare
                ``snake_case`` names. A split is a level of nesting of its
                own (``{"data": {"train": {"samples": ...}}}``), never part
                of a leaf key (``train_samples``). A tracker that stores
                flat names flattens the mapping with
                [`flatten_parameters`][probreg.core.flatten_parameters] into
                parameter paths such as ``data/train/samples``.

        Returns:
            None.
        """
        ...

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        """Record metric values at a step.

        Args:
            values: Metric values keyed by the tag to record them under.
            step: The step the metrics belong to.

        Returns:
            None.
        """
        ...

    def log_artifact(self, name: str, value: Any) -> None:
        """Record an artifact under a name.

        Args:
            name: The name to record the artifact under.
            value: The artifact to record.

        Returns:
            None.
        """
        ...


@dataclass(frozen=True)
class TrackerEventSink:
    """An [`EventSink`][probreg.core.EventSink] forwarding event metrics to a tracker.

    Each metric is logged under its metric tag, built from the event's
    stage, the event's split and the metric name, with the event's own
    step. A decision event carries no metrics, so it adds no series.

    Only
    [`ExperimentTracker.log_metrics`][probreg.core.ExperimentTracker.log_metrics]
    is called. Hyperparameters and artifacts stay caller-driven, since
    neither arrives on a training event.

    Attributes:
        tracker: The experiment tracker that receives the metrics.
    """

    tracker: ExperimentTracker

    def on_event(self, event: TrainingEvent) -> None:
        """Log an event's metrics to the tracker under their metric tags.

        Args:
            event: The training event whose metrics to forward.

        Returns:
            None.
        """
        if not event.metrics:
            return
        self.tracker.log_metrics(
            {
                metric_tag(event.stage, event.split, name): value
                for name, value in event.metrics.items()
            },
            step=event.step,
        )
