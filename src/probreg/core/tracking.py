"""Training-event and experiment-tracking protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
