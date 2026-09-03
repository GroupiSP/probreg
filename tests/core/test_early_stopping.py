from __future__ import annotations

import math

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from probreg.core.early_stopping import (
    EarlyStopper,
    EarlyStoppingState,
    MetricSource,
    OptimizationMode,
)


def test_minimizing_metric_tracks_best_value_and_stops_after_patience() -> None:
    stopper = EarlyStopper(
        metric="loss", mode=OptimizationMode.MIN, patience=1, min_delta=0.1
    )

    assert stopper.observe(1.0, epoch=0).improved
    assert not stopper.observe(0.95, epoch=1).should_stop
    decision = stopper.observe(0.94, epoch=2)

    assert decision.should_stop
    assert decision.state.best_score == 1.0
    assert decision.state.best_epoch == 0
    assert decision.state.non_improving_epochs == 2


def test_maximizing_metric_accepts_improvements_above_min_delta() -> None:
    stopper = EarlyStopper(metric="accuracy", mode="max", patience=0, min_delta=0.05)

    assert stopper.observe(0.5, epoch=0).improved
    assert stopper.observe(0.56, epoch=1).improved
    assert stopper.state.best_score == 0.56
    assert stopper.state.source is MetricSource.VALIDATION


def test_expects_validation_and_monitored_metric_name_reflect_configuration() -> None:
    validation_stopper = EarlyStopper(metric="validation_loss", mode="min", patience=0)
    training_stopper = EarlyStopper(
        metric="loss", mode="min", patience=0, source=MetricSource.TRAINING
    )

    assert validation_stopper.expects_validation() is True
    assert validation_stopper.monitored_metric_name() == "validation_loss"
    assert training_stopper.expects_validation() is False
    assert training_stopper.monitored_metric_name() == "loss"


def test_stopper_can_be_restored_from_checkpointable_state() -> None:
    original = EarlyStopper(
        metric="loss",
        mode="min",
        patience=2,
        source=MetricSource.TRAINING,
    )
    original.observe(1.0, epoch=0)
    state = original.observe(1.1, epoch=1).state
    restored = EarlyStopper(
        metric="loss",
        mode="min",
        patience=2,
        source="training",
        state=state,
    )

    decision = restored.observe(1.2, epoch=2)

    assert decision.should_stop is False
    assert decision.state.non_improving_epochs == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": "", "mode": "min", "patience": 1}, "non-empty"),
        ({"metric": "loss", "mode": "min", "patience": -1}, "non-negative"),
        (
            {"metric": "loss", "mode": "min", "patience": 1, "min_delta": math.nan},
            "finite",
        ),
    ],
)
def test_stopper_rejects_invalid_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        EarlyStopper(**kwargs)  # type: ignore[arg-type]


def test_stopper_rejects_invalid_observations_and_observations_after_stop() -> None:
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    with pytest.raises(ValueError, match="finite"):
        stopper.observe(math.inf, epoch=0)

    stopper.observe(1.0, epoch=0)
    assert stopper.observe(1.0, epoch=1).should_stop
    with pytest.raises(RuntimeError, match="after early stopping"):
        stopper.observe(1.0, epoch=2)


@given(
    patience=st.integers(min_value=0),
    extra_epochs=st.integers(min_value=1),
)
@example(patience=1, extra_epochs=98)
def test_state_rejects_unstopped_exhausted_patience(
    patience: int,
    extra_epochs: int,
) -> None:
    with pytest.raises(ValueError, match="stopped"):
        EarlyStoppingState(
            metric="loss",
            mode=OptimizationMode.MIN,
            min_delta=0.0,
            patience=patience,
            best_score=1.0,
            best_epoch=0,
            non_improving_epochs=patience + extra_epochs,
            stopped=False,
        )


@given(
    mode=st.sampled_from(tuple(OptimizationMode)),
    patience=st.integers(min_value=0, max_value=20),
    min_delta=st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    initial=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_constant_observations_stop_exactly_after_patience(
    mode: OptimizationMode,
    patience: int,
    min_delta: float,
    initial: float,
) -> None:
    stopper = EarlyStopper(
        metric="metric",
        mode=mode,
        patience=patience,
        min_delta=min_delta,
    )

    assert stopper.observe(initial, epoch=0).improved
    for epoch in range(1, patience + 1):
        assert not stopper.observe(initial, epoch=epoch).should_stop

    decision = stopper.observe(initial, epoch=patience + 1)

    assert decision.should_stop
    assert decision.state.best_score == initial
    assert decision.state.best_epoch == 0
