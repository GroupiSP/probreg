from __future__ import annotations

from typing import Any

from probreg.core.tracking import EventSink, ExperimentTracker, TrainingEvent
from probreg.core.types import TrainingState


class InMemoryTracker:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: dict[str, Any] = {}

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def log_params(self, values: dict[str, Any]) -> None:
        self.params.update(values)

    def log_metrics(self, values: dict[str, float], *, step: int) -> None:
        self.metrics.append((values, step))

    def log_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value


def test_tracker_protocols_record_structured_training_data() -> None:
    tracker = InMemoryTracker()
    sink: EventSink = tracker
    experiment_tracker: ExperimentTracker = tracker
    event = TrainingEvent(
        name="epoch_end",
        stage="mean",
        iteration=0,
        step=3,
        metrics={"loss": 0.1},
        state=TrainingState(stage="mean"),
    )

    sink.on_event(event)
    experiment_tracker.log_params({"learning_rate": 0.01})
    experiment_tracker.log_metrics({"loss": 0.1}, step=3)
    experiment_tracker.log_artifact("checkpoint", "mean-best")

    assert tracker.events == [event]
    assert tracker.params == {"learning_rate": 0.01}
    assert tracker.metrics == [({"loss": 0.1}, 3)]
    assert tracker.artifacts == {"checkpoint": "mean-best"}
