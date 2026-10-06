"""Tests for the training-event and experiment-tracking contracts.

`TrackerEventSink` is exercised only at its real seam, a `run_supervised`
call, so those tests need the optional JAX backend. Core stays
backend-neutral: the shared `supervised_run` driver imports the backend
lazily and skips when it is absent, so the rest of this module still runs
under a bare `pytest tests/core`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from probreg.core.naming import Split, parse_metric_tag
from probreg.core.tracking import (
    EventSink,
    ExperimentTracker,
    TrackerEventSink,
    TrainingEvent,
)
from probreg.core.types import StageResult, TrainingState

Tracker = Callable[[], Any]
Run = Callable[..., StageResult]


def test_tracker_protocols_record_structured_training_data(
    in_memory_tracker: Tracker,
) -> None:
    tracker = in_memory_tracker()
    sink: EventSink = tracker
    experiment_tracker: ExperimentTracker = tracker
    event = TrainingEvent(
        name="epoch_end",
        stage="mean",
        split=Split.TRAIN,
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


@given(name=st.text(min_size=1), split=st.sampled_from(Split))
def test_tracker_event_sink_tags_any_event_without_registration(
    in_memory_tracker: Tracker, name: str, split: Split
) -> None:
    tracker = in_memory_tracker()

    TrackerEventSink(tracker).on_event(
        TrainingEvent(
            name=name,
            stage="mean",
            split=split,
            iteration=0,
            step=2,
            metrics={"loss": 0.5},
            state=TrainingState(stage="mean"),
        )
    )

    assert tracker.metrics == [({f"mean/{split}/loss": 0.5}, 2)]


def test_tracker_event_sink_tags_a_real_run_by_stage_and_split(
    in_memory_tracker: Tracker, supervised_run: Run
) -> None:
    tracker = in_memory_tracker()
    sink: EventSink = TrackerEventSink(tracker)

    result = supervised_run(sink, stage="mean")

    assert tracker.tags == {"mean/train/loss", "mean/validation/loss"}
    assert set(result.state.metric_history) == tracker.tags
    assert tracker.params == {}
    assert tracker.artifacts == {}


@given(epochs=st.integers(min_value=1, max_value=4), validate=st.booleans())
@settings(deadline=None, max_examples=6)
def test_tracker_event_sink_tags_every_metric_with_its_stage_and_split(
    in_memory_tracker: Tracker, supervised_run: Run, epochs: int, validate: bool
) -> None:
    observer = in_memory_tracker()
    tracker = in_memory_tracker()

    result = supervised_run(
        observer, TrackerEventSink(tracker), epochs=epochs, validate=validate
    )

    assert tracker.tags == {
        f"{event.stage}/{event.split}/{metric}"
        for event in observer.events
        for metric in event.metrics
    }
    assert set(result.state.metric_history) == tracker.tags
    for tag in tracker.tags:
        assert parse_metric_tag(tag).stage == "supervised"


@given(epochs=st.integers(min_value=1, max_value=4))
@settings(deadline=None, max_examples=4)
def test_tracker_event_sink_preserves_every_emitted_metric(
    in_memory_tracker: Tracker, supervised_run: Run, epochs: int
) -> None:
    observer = in_memory_tracker()
    tracker = in_memory_tracker()

    supervised_run(observer, TrackerEventSink(tracker), epochs=epochs)

    emitted = sum(len(event.metrics) for event in observer.events)
    forwarded = sum(len(values) for values, _ in tracker.metrics)
    assert forwarded == emitted


@given(epochs=st.integers(min_value=1, max_value=4))
@settings(deadline=None, max_examples=4)
def test_tracker_event_sink_records_the_step_of_each_event(
    in_memory_tracker: Tracker, supervised_run: Run, epochs: int
) -> None:
    observer = in_memory_tracker()
    tracker = in_memory_tracker()

    supervised_run(observer, TrackerEventSink(tracker), epochs=epochs)

    assert [step for _, step in tracker.metrics] == [
        event.step for event in observer.events
    ]


@given(
    stages=st.lists(
        st.sampled_from(["mean", "variance", "joint"]),
        min_size=2,
        max_size=2,
        unique=True,
    )
)
@settings(deadline=None, max_examples=3)
def test_tracker_event_sink_tags_are_injective_in_stage_split_and_metric(
    in_memory_tracker: Tracker, supervised_run: Run, stages: list[str]
) -> None:
    tracker = in_memory_tracker()

    for stage in stages:
        supervised_run(TrackerEventSink(tracker), epochs=1, stage=stage)

    tags = [tag for values, _ in tracker.metrics for tag in values]
    assert len(set(tags)) == len(stages) * len(Split)
    assert len(tags) == len(set(tags))
