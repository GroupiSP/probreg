"""The metric-naming scheme, observed from real training runs.

A metric has one bare metric name, and every recorded series one metric
tag, ``stage/split/metric``. These tests drive a real `run_supervised` call
or a real staged mean→variance run, and observe it the way a user would:
through an in-memory tracker behind a `TrackerEventSink`, a recording sink,
`state.metric_history` and the stage result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from probreg.core.checkpoints import Checkpoint
from probreg.core.early_stopping import EarlyStopper
from probreg.core.naming import Split, parse_metric_tag
from probreg.core.tracking import EventSink, TrackerEventSink, TrainingEvent
from probreg.core.types import Batch, StageResult, TrainingState
from probreg.jax import (
    HeldOutValidation,
    create_optimizer,
    initialize_training_state,
    run_supervised,
)
from probreg.jax.distributions import GammaHead
from probreg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
)

DECISION_EVENTS = {"best_model", "early_stop"}


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


class InMemoryTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int]] = []

    def log_params(self, values: dict[str, Any]) -> None:
        del values

    def log_metrics(self, values: dict[str, float], *, step: int) -> None:
        self.metrics.append((dict(values), step))

    def log_artifact(self, name: str, value: Any) -> None:
        del name, value

    @property
    def tags(self) -> set[str]:
        return {tag for values, _ in self.metrics for tag in values}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.values: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.values[key] = checkpoint

    def load(self, key: str) -> Checkpoint:
        return self.values[key]

    def exists(self, key: str) -> bool:
        return key in self.values


def squared_error(
    model: nnx.Module,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del sample_weight, key, training
    return jnp.mean(jnp.square(model(inputs) - targets))


def loader(*, split: str, epoch: int) -> list[Batch]:
    del epoch
    inputs = jnp.array([[-1.0], [0.0], [1.0]])
    offset = 0.0 if split == "train" else 0.5
    return [Batch(inputs=inputs, targets=2.0 * inputs + offset)]


class Observers:
    """A recording sink and a tracker sink observing the same run."""

    def __init__(self) -> None:
        self.recorder = RecordingSink()
        self.tracker = InMemoryTracker()

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        return (self.recorder, TrackerEventSink(self.tracker))


def run_single_stage(
    observers: Observers,
    *,
    epochs: int = 3,
    learning_rate: float = 0.1,
    validate: bool = True,
    early_stopper: EarlyStopper | None = None,
    stage: str | None = None,
) -> StageResult:
    model = LinearModel(rngs=nnx.Rngs(0))
    optimizer = create_optimizer(model, optax.sgd(learning_rate))
    state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
    stage_kwargs = {} if stage is None else {"stage": stage}
    return run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=epochs,
        validation=(
            HeldOutValidation(model=model, loader=loader, loss=squared_error)
            if validate
            else None
        ),
        early_stopper=early_stopper,
        event_sinks=observers.sinks,
        **stage_kwargs,
    )


def run_staged(
    observers: Observers,
    *,
    early_stopper: EarlyStopper | None = None,
    validate: bool = True,
    learning_rate: float = 0.1,
) -> tuple[TrainingState, StageResult, StageResult]:
    mean_model = LinearModel(rngs=nnx.Rngs(0))
    state = TrainingState(rng_state=jax.random.key(1))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.sgd(learning_rate)),
        train_loader=loader,
        options=SupervisedStageOptions(
            epochs=4,
            validation=(
                HeldOutValidation(model=mean_model, loader=loader, loss=squared_error)
                if validate
                else None
            ),
            early_stopper=early_stopper,
            event_sinks=observers.sinks,
            checkpoint_store=MemoryCheckpointStore(),
            checkpoint_key="mean-best",
        ),
    )
    mean_stage.prepare(state)
    mean_result = mean_stage.train(state)

    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
        source_loader=loader,
        options=SupervisedStageOptions(epochs=3, event_sinks=observers.sinks),
        validation_factory=(
            (
                lambda residuals: HeldOutValidation(
                    model=variance_model,
                    loader=residuals,
                    loss=variance_stage.loss,
                )
            )
            if validate
            else None
        ),
    )
    variance_stage.prepare(state)
    variance_result = variance_stage.train(state)
    return state, mean_result, variance_result


def expected_tags(stage: str, events: Sequence[TrainingEvent]) -> set[str]:
    return {
        f"{stage}/{event.split}/{metric}"
        for event in events
        if event.stage == stage
        for metric in event.metrics
    }


def test_tracker_tags_are_stage_split_metric_and_match_the_history() -> None:
    observers = Observers()

    result = run_single_stage(observers, stage="mean")

    assert observers.tracker.tags == {"mean/train/loss", "mean/validation/loss"}
    assert set(result.state.metric_history) == observers.tracker.tags


def test_single_stage_run_with_the_default_stage_records_supervised_tags() -> None:
    observers = Observers()

    result = run_single_stage(observers, validate=False)

    assert observers.tracker.tags == {"supervised/train/loss"}
    assert set(result.state.metric_history) == {"supervised/train/loss"}
    assert len(result.state.metric_history["supervised/train/loss"]) == 3


def test_events_carry_the_split_they_concern() -> None:
    observers = Observers()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    run_single_stage(observers, learning_rate=0.0, epochs=4, early_stopper=stopper)

    splits = {event.name: event.split for event in observers.recorder.events}
    assert splits == {
        "epoch_end": Split.TRAIN,
        "validation_end": Split.VALIDATION,
        "best_model": Split.VALIDATION,
        "early_stop": Split.VALIDATION,
    }


@pytest.mark.parametrize("validate", [False, True])
def test_decision_events_name_the_train_split_when_the_stopper_reads_it(
    validate: bool,
) -> None:
    observers = Observers()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0, source=Split.TRAIN)

    result = run_single_stage(
        observers,
        learning_rate=0.0,
        epochs=4,
        validate=validate,
        early_stopper=stopper,
    )

    decisions = [
        event for event in observers.recorder.events if event.name in DECISION_EVENTS
    ]
    assert [event.name for event in decisions] == ["best_model", "early_stop"]
    assert all(event.split is Split.TRAIN for event in decisions)
    history = result.state.metric_history["supervised/train/loss"]
    assert [event.payload for event in decisions] == [
        {"metric": "loss", "value": history[0]},
        {"metric": "loss", "value": history[1]},
    ]


def test_decision_events_carry_no_metrics_and_add_no_tracker_series() -> None:
    observers = Observers()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    result = run_single_stage(
        observers, learning_rate=0.0, epochs=4, early_stopper=stopper
    )

    decisions = [
        event for event in observers.recorder.events if event.name in DECISION_EVENTS
    ]
    assert {event.name for event in decisions} == DECISION_EVENTS
    assert all(event.metrics == {} for event in decisions)
    history = result.state.metric_history["supervised/validation/loss"]
    assert [event.payload for event in decisions] == [
        {"metric": "loss", "value": history[0]},
        {"metric": "loss", "value": history[1]},
    ]
    measured = sum(len(event.metrics) for event in observers.recorder.events)
    assert sum(len(values) for values, _ in observers.tracker.metrics) == measured
    assert observers.tracker.tags == {
        "supervised/train/loss",
        "supervised/validation/loss",
    }


@given(epochs=st.integers(min_value=1, max_value=4), validate=st.booleans())
@settings(deadline=None, max_examples=6)
def test_every_metric_of_every_event_is_tagged_with_its_stage_and_split(
    epochs: int, validate: bool
) -> None:
    observers = Observers()

    result = run_single_stage(observers, epochs=epochs, validate=validate)

    assert observers.tracker.tags == expected_tags(
        "supervised", observers.recorder.events
    )
    assert set(result.state.metric_history) == observers.tracker.tags
    for tag in observers.tracker.tags:
        assert parse_metric_tag(tag).stage == "supervised"


def test_stage_result_and_event_metrics_are_bare() -> None:
    observers = Observers()

    result = run_single_stage(observers)

    assert set(result.metrics) == {"loss"}
    assert all(
        set(event.metrics) == {"loss"}
        for event in observers.recorder.events
        if event.name not in DECISION_EVENTS
    )


def test_the_two_stages_of_a_staged_run_never_share_a_tag() -> None:
    observers = Observers()

    state, _, _ = run_staged(observers)

    mean_tags = expected_tags("mean", observers.recorder.events)
    variance_tags = expected_tags("variance", observers.recorder.events)
    assert mean_tags == {"mean/train/loss", "mean/validation/loss"}
    assert variance_tags == {"variance/train/loss", "variance/validation/loss"}
    assert observers.tracker.tags == mean_tags | variance_tags
    assert set(state.metric_history) == observers.tracker.tags


@pytest.mark.parametrize("validate", [False, True])
def test_staged_run_without_a_stopper_returns_bare_train_only_metrics(
    validate: bool,
) -> None:
    state, mean_result, _ = run_staged(Observers(), validate=validate)

    assert set(mean_result.metrics) == {"loss"}
    assert mean_result.loss == state.metric_history["mean/train/loss"][-1]


@given(source=st.sampled_from(Split), validate=st.booleans())
@settings(deadline=None, max_examples=4)
def test_staged_restore_returns_bare_train_only_metrics(
    source: Split, validate: bool
) -> None:
    # A zero learning rate makes epoch 0 the best, so the stopper halts at
    # epoch 1 and the stage restores the epoch-0 checkpoint.
    validate = validate or source is Split.VALIDATION
    stopper = EarlyStopper(metric="loss", mode="min", patience=0, source=source)

    state, mean_result, _ = run_staged(
        Observers(), early_stopper=stopper, validate=validate, learning_rate=0.0
    )

    restored_history = state.metric_history["mean/train/loss"]
    assert len(restored_history) == 1
    assert set(mean_result.metrics) == {"loss"}
    assert mean_result.loss == restored_history[-1]
