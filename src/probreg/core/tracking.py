"""Training-event and experiment-tracking protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from probreg.core.types import TrainingState


@dataclass(frozen=True)
class TrainingEvent:
    """A structured event emitted during staged training."""

    name: str
    stage: str
    iteration: int
    step: int
    metrics: Mapping[str, float]
    state: TrainingState
    payload: Mapping[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    """Consumes training events."""

    def on_event(self, event: TrainingEvent) -> None: ...


class ExperimentTracker(Protocol):
    """Records parameters, metrics, and artifacts for an experiment."""

    def log_params(self, values: Mapping[str, Any]) -> None: ...

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None: ...

    def log_artifact(self, name: str, value: Any) -> None: ...


DEFAULT_EVENT_PREFIXES: Mapping[str, str] = MappingProxyType(
    {"epoch_end": "train/", "validation_end": "validation/"}
)
"""The event-prefix mapping :class:`TrackerEventSink` uses by default."""


@dataclass(frozen=True)
class TrackerEventSink:
    """An :class:`EventSink` that forwards event metrics to a tracker.

    Metrics are logged under the tag ``<stage>/<event prefix><metric>``,
    with the event's own step. The stage segment is unconditional: the two
    stages of a staged run each emit ``epoch_end`` against their own epoch
    counter, so a stage-blind tag would overwrite one stage's curve with
    the other's.

    Only :meth:`ExperimentTracker.log_metrics` is called. Hyperparameters
    and artifacts stay caller-driven, since neither arrives on a training
    event.

    Attributes:
        tracker: The experiment tracker that receives the metrics.
        event_prefixes: Prefix per event name, applied after the stage
            segment. Events absent from the mapping forward their metrics
            verbatim under the stage segment rather than being dropped, so
            give every event whose curves must stay separate a distinct
            prefix.
    """

    tracker: ExperimentTracker
    event_prefixes: Mapping[str, str] = DEFAULT_EVENT_PREFIXES

    def on_event(self, event: TrainingEvent) -> None:
        """Log an event's metrics to the tracker under namespaced tags.

        Args:
            event: The training event whose metrics to forward.

        Returns:
            None.
        """
        prefix = self.event_prefixes.get(event.name, "")
        self.tracker.log_metrics(
            {
                f"{event.stage}/{prefix}{name}": value
                for name, value in event.metrics.items()
            },
            step=event.step,
        )
