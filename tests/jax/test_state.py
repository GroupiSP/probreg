from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probreg.core.checkpoints import Checkpoint
from probreg.core.types import ParameterRole, StageState, TrainingState
from probreg.jax.state import (
    create_optimizer,
    freeze_training_state,
    restore_checkpoint,
    snapshot,
)


class ScalarModel(nnx.Module):
    def __init__(self, value: float) -> None:
        self.value = nnx.Param(jnp.asarray(value))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.value[...] * inputs


class OtherModel(nnx.Module):
    def __init__(self, value: float) -> None:
        self.other = nnx.Param(jnp.asarray(value))


def test_snapshot_is_independent_and_checkpoint_restores_live_objects() -> None:
    model = ScalarModel(1.0)
    optimizer = create_optimizer(model, optax.sgd(0.1))
    state = TrainingState(
        model_components={"mean_model": model},
        parameter_roles={"mean_model": ParameterRole.MEAN},
        optimizer_states={"mean_optimizer": optimizer},
        rng_state=jax.random.key(3),
        lifecycle_state=StageState.MEAN_READY,
        stage="mean",
        metric_history={"mean_training_loss": [1.0]},
    )
    checkpoint = Checkpoint(
        state=freeze_training_state(state),
        epoch=0,
        parameters=snapshot(model),
        optimizer_state=snapshot(optimizer),
        rng_state=state.rng_state,
    )

    model.value[...] = 5.0
    optimizer.step[...] = 4
    state.lifecycle_state = StageState.VARIANCE_READY
    state.metric_history["mean_training_loss"].append(2.0)

    restore_checkpoint(
        checkpoint,
        state=state,
        model=model,
        optimizer=optimizer,
        model_name="mean_model",
        optimizer_name="mean_optimizer",
    )

    assert model.value[...] == 1.0
    assert int(optimizer.step.get_value()) == 0
    assert state.lifecycle_state is StageState.MEAN_READY
    assert state.active_stage == "mean"
    assert state.model_components == {"mean_model": model}
    assert state.optimizer_states == {"mean_optimizer": optimizer}
    assert state.metric_history == {"mean_training_loss": [1.0]}


def test_restore_checkpoint_rejects_incompatible_objects_before_mutation() -> None:
    saved_model = ScalarModel(1.0)
    saved_optimizer = create_optimizer(saved_model, optax.sgd(0.1))
    saved_state = TrainingState(rng_state=jax.random.key(0))
    checkpoint = Checkpoint(
        state=freeze_training_state(saved_state),
        epoch=0,
        parameters=snapshot(saved_model),
        optimizer_state=snapshot(saved_optimizer),
        rng_state=saved_state.rng_state,
    )
    incompatible_model = OtherModel(5.0)
    incompatible_optimizer = create_optimizer(incompatible_model, optax.adam(0.1))
    live_state = TrainingState(rng_state=jax.random.key(1))

    with pytest.raises(ValueError, match="model graph"):
        restore_checkpoint(
            checkpoint,
            state=live_state,
            model=incompatible_model,
            optimizer=incompatible_optimizer,
        )

    assert incompatible_model.other[...] == 5.0


def test_restore_checkpoint_validates_optimizer_before_mutating_model() -> None:
    saved_model = ScalarModel(1.0)
    saved_optimizer = create_optimizer(saved_model, optax.sgd(0.1))
    saved_state = TrainingState(rng_state=jax.random.key(0))
    checkpoint = Checkpoint(
        state=freeze_training_state(saved_state),
        epoch=0,
        parameters=snapshot(saved_model),
        optimizer_state=snapshot(saved_optimizer),
        rng_state=saved_state.rng_state,
    )
    live_model = ScalarModel(5.0)
    incompatible_optimizer = create_optimizer(live_model, optax.adam(0.1))
    live_state = TrainingState(rng_state=jax.random.key(1))

    with pytest.raises(ValueError, match="optimizer state"):
        restore_checkpoint(
            checkpoint,
            state=live_state,
            model=live_model,
            optimizer=incompatible_optimizer,
        )

    assert live_model.value[...] == 5.0
