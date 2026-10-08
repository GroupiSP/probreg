"""The epoch loop drives any per-batch step, independently of an NNX model.

Every test steps a plain Python double, no `nnx` model, optimizer or loss in
sight, which is how the posterior stage will drive an inference method's
``update(batch, key)``.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probreg.core.checkpoints import InMemoryCheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.metric_registry import EpochPredictionData, RootMeanSquaredError
from probreg.core.naming import Split
from probreg.core.types import Batch, TrainingState
from probreg.jax.epoch_loop import StateSnapshot, run_epoch_loop
from probreg.jax.metrics import MetricSuite


class ScriptedStep:
    """A per-batch step returning a scripted loss and recording what it saw."""

    def __init__(self, losses: list[float]) -> None:
        self._losses = iter(losses)
        self.batches: list[Batch] = []
        self.keys: list[jax.Array] = []

    def __call__(self, batch: Batch, key: jax.Array, /) -> Mapping[str, jax.Array]:
        self.batches.append(batch)
        self.keys.append(key)
        return {"loss": jnp.asarray(next(self._losses))}


def _loader(n_batches: int):
    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        return [
            Batch(inputs=jnp.array([[float(epoch)]]), metadata={"index": index})
            for index in range(n_batches)
        ]

    return loader


def _no_snapshot() -> StateSnapshot:
    raise AssertionError("no checkpoint was expected.")


def _state() -> TrainingState:
    return TrainingState(rng_state=jax.random.key(0))


@settings(deadline=None, max_examples=20)
@given(
    epochs=st.integers(min_value=1, max_value=3),
    n_batches=st.integers(min_value=1, max_value=3),
    data=st.data(),
)
def test_train_loss_per_epoch_is_the_mean_of_the_step_losses(
    epochs: int, n_batches: int, data: st.DataObject
) -> None:
    losses = data.draw(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, width=32),
            min_size=epochs * n_batches,
            max_size=epochs * n_batches,
        )
    )
    step = ScriptedStep(losses)
    state = _state()

    result = run_epoch_loop(
        step=step,
        snapshot_state=_no_snapshot,
        train_loader=_loader(n_batches),
        state=state,
        epochs=epochs,
        stage="posterior",
    )

    expected = [
        sum(losses[e * n_batches : (e + 1) * n_batches]) / n_batches
        for e in range(epochs)
    ]
    assert state.metric_history["posterior/train/loss"] == pytest.approx(
        expected, rel=1e-5, abs=1e-5
    )
    assert result.loss == pytest.approx(expected[-1], rel=1e-5, abs=1e-5)
    assert state.active_stage == "posterior"
    assert len(step.batches) == epochs * n_batches
    distinct_keys = {tuple(jax.random.key_data(key).tolist()) for key in step.keys}
    assert len(distinct_keys) == len(step.keys)


def test_best_checkpoint_holds_the_snapshot_taken_at_the_last_improvement() -> None:
    # Losses per epoch 3, 1, 2, 2, 0: the best is epoch 1, and patience 1 stops
    # after epoch 3, so the final 0 is never reached.
    snapshots: list[int] = []

    def snapshot_state() -> StateSnapshot:
        snapshots.append(len(snapshots))
        return StateSnapshot(parameters={"taken": snapshots[-1]}, optimizer_state="opt")

    store = InMemoryCheckpointStore()
    state = _state()

    result = run_epoch_loop(
        step=ScriptedStep([3.0, 1.0, 2.0, 2.0, 0.0]),
        snapshot_state=snapshot_state,
        train_loader=_loader(1),
        state=state,
        epochs=5,
        stage="posterior",
        early_stopper=EarlyStopper(
            metric="loss", mode="min", patience=1, source=Split.TRAIN
        ),
        checkpoint_store=store,
    )

    checkpoint = store.load("posterior/best")
    assert checkpoint.epoch == 1
    assert checkpoint.parameters == {"taken": 1}
    assert checkpoint.optimizer_state == "opt"
    assert snapshots == [0, 1]
    assert state.metric_history["posterior/train/loss"] == [3.0, 1.0, 2.0, 2.0]
    assert result.loss == 2.0


def test_epoch_metrics_reduce_what_the_prediction_collector_returns() -> None:
    collector_keys: list[jax.Array] = []

    def collect(batch: Batch, key: jax.Array, /) -> EpochPredictionData:
        collector_keys.append(key)
        return EpochPredictionData(targets=np.array([1.0]), mean=np.array([3.0]))

    step = ScriptedStep([0.0, 0.0])
    state = _state()

    run_epoch_loop(
        step=step,
        snapshot_state=_no_snapshot,
        train_loader=_loader(2),
        state=state,
        epochs=1,
        stage="posterior",
        metrics=MetricSuite(epoch=(RootMeanSquaredError(),), predictor=_unused),
        epoch_predictions=collect,
    )

    assert state.metric_history["posterior/train/rmse"] == [2.0]
    step_keys = {tuple(jax.random.key_data(key).tolist()) for key in step.keys}
    assert len(collector_keys) == 2
    assert all(
        tuple(jax.random.key_data(key).tolist()) not in step_keys
        for key in collector_keys
    )


def test_epoch_metrics_without_a_prediction_collector_are_refused() -> None:
    state = _state()

    with pytest.raises(ValueError, match="epoch prediction collector"):
        run_epoch_loop(
            step=ScriptedStep([0.0]),
            snapshot_state=_no_snapshot,
            train_loader=_loader(1),
            state=state,
            epochs=1,
            stage="posterior",
            metrics=MetricSuite(epoch=(RootMeanSquaredError(),), predictor=_unused),
        )
    assert state.active_stage is None
    assert state.metric_history == {}


def _unused(*args: object) -> EpochPredictionData:
    raise AssertionError("the suite predictor is not used by the loop.")
