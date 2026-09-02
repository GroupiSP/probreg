from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from probreg.core.types import (
    Batch,
    CheckpointRef,
    ParameterRole,
    StageState,
    TrainingState,
)


def test_batch_is_immutable() -> None:
    batch = Batch(inputs=[1, 2], targets=[3], metadata={"split": "train"})

    with pytest.raises(FrozenInstanceError):
        batch.targets = [4]  # type: ignore[misc]

    assert batch.metadata == {"split": "train"}


def test_training_states_do_not_share_mutable_defaults() -> None:
    first = TrainingState()
    second = TrainingState()
    first.parameter_roles["mean"] = ParameterRole.MEAN
    first.checkpoint_registry["best"] = CheckpointRef("best-model")

    assert second.parameter_roles == {}
    assert second.checkpoint_registry == {}


def test_training_state_separates_lifecycle_from_active_stage() -> None:
    state = TrainingState()

    state.lifecycle_state = StageState.INITIALIZED
    state.active_stage = "mean"

    assert state.lifecycle_state is StageState.INITIALIZED
    assert state.active_stage == "mean"


def test_training_state_stage_alias_preserves_active_stage_compatibility() -> None:
    state = TrainingState()

    state.stage = "supervised"

    assert state.active_stage == "supervised"
    assert state.stage == "supervised"
