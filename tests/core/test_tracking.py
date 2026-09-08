"""Tests for the training-event and experiment-tracking contracts.

`TrackerEventSink` is exercised only at its real seam, a `run_supervised`
call, so those tests need the optional JAX backend. Core stays
backend-neutral: the backend is imported inside the driver below and the
tests that need it are marked, so the rest of this module still runs
under a bare `pytest tests/core`.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from probreg.core.tracking import (
    EventSink,
    ExperimentTracker,
    TrackerEventSink,
    TrainingEvent,
)
from probreg.core.types import TrainingState

requires_jax_backend = pytest.mark.skipif(
    importlib.util.find_spec("flax") is None,
    reason="The JAX backend is required to drive a real training run.",
)


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


def train_with_sinks(*sinks: EventSink, epochs: int = 3) -> None:
    """Drive a real validated training run through the given event sinks.

    A one-parameter linear model is fitted on a single constant batch, with
    held-out validation enabled so that both `epoch_end` and
    `validation_end` events are emitted under the `"mean"` stage. The
    validation metric prefix is cleared so that the tags observed by the
    tracker come only from the sink under test.

    Args:
        *sinks: The event sinks to attach to the run.
        epochs: The number of epochs to train for.

    Returns:
        None.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from probreg.core.types import Batch
    from probreg.jax import (
        HeldOutValidation,
        create_optimizer,
        initialize_training_state,
        run_supervised,
    )

    class LinearModel(nnx.Module):
        def __init__(self) -> None:
            self.linear = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

        def __call__(self, inputs: Any) -> Any:
            return self.linear(inputs)

    def squared_error(
        model: LinearModel,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: Any,
        training: bool,
    ) -> Any:
        del sample_weight, key, training
        return jnp.mean(jnp.square(model(inputs) - targets))

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del epoch
        target = 2.0 if split == "train" else 1.0
        return [Batch(inputs=jnp.array([[1.0]]), targets=jnp.array([[target]]))]

    model = LinearModel()
    optimizer = create_optimizer(model, optax.sgd(0.1))
    state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=epochs,
        validation=HeldOutValidation(
            model=model, loader=loader, loss=squared_error, metric_prefix=""
        ),
        stage="mean",
        event_sinks=list(sinks),
    )


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


def test_tracker_event_sink_satisfies_the_event_sink_protocol() -> None:
    sink: EventSink = TrackerEventSink(InMemoryTracker())

    assert callable(sink.on_event)


@requires_jax_backend
def test_tracker_event_sink_namespaces_a_real_run_by_stage_and_event() -> None:
    tracker = InMemoryTracker()

    train_with_sinks(TrackerEventSink(tracker))

    tags = {tag for values, _ in tracker.metrics for tag in values}
    assert "mean/train/loss" in tags
    assert "mean/validation/loss" in tags
    assert all(tag.startswith("mean/") for tag in tags)
    assert tracker.params == {}
    assert tracker.artifacts == {}


@requires_jax_backend
def test_tracker_event_sink_forwards_metrics_of_unmapped_events() -> None:
    tracker = InMemoryTracker()

    train_with_sinks(TrackerEventSink(tracker, event_prefixes={}))

    tags = {tag for values, _ in tracker.metrics for tag in values}
    assert tags == {"mean/loss"}


@requires_jax_backend
@given(epochs=st.integers(min_value=1, max_value=4))
@settings(deadline=None, max_examples=4)
def test_tracker_event_sink_preserves_every_emitted_metric(epochs: int) -> None:
    observer = InMemoryTracker()
    tracker = InMemoryTracker()

    train_with_sinks(observer, TrackerEventSink(tracker), epochs=epochs)

    emitted = sum(len(event.metrics) for event in observer.events)
    forwarded = sum(len(values) for values, _ in tracker.metrics)
    assert forwarded == emitted


@requires_jax_backend
@given(epochs=st.integers(min_value=1, max_value=4))
@settings(deadline=None, max_examples=4)
def test_tracker_event_sink_records_the_step_of_each_event(epochs: int) -> None:
    observer = InMemoryTracker()
    tracker = InMemoryTracker()

    train_with_sinks(observer, TrackerEventSink(tracker), epochs=epochs)

    assert [step for _, step in tracker.metrics] == [
        event.step for event in observer.events if event.metrics
    ]


@given(
    prefixes=st.dictionaries(
        st.sampled_from(["epoch_end", "validation_end", "best_model", "early_stop"]),
        st.sampled_from(["a/", "b/", "c/", "d/"]),
        min_size=1,
        max_size=4,
    )
)
def test_tracker_event_sink_tags_are_injective_in_stage_event_and_metric(
    prefixes: dict[str, str],
) -> None:
    assume(len(set(prefixes.values())) == len(prefixes))
    tracker = InMemoryTracker()
    sink = TrackerEventSink(tracker, event_prefixes=prefixes)
    keys = [
        (stage, name, metric)
        for stage in ("mean", "variance")
        for name in prefixes
        for metric in ("loss", "rmse")
    ]

    for stage, name, metric in keys:
        sink.on_event(
            TrainingEvent(
                name=name,
                stage=stage,
                iteration=0,
                step=0,
                metrics={metric: 0.1},
                state=TrainingState(stage=stage),
            )
        )

    tags = [tag for values, _ in tracker.metrics for tag in values]
    assert len(set(tags)) == len(keys)
