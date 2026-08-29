from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from probreg.core.types import Batch, CheckpointRef, ParameterRole, TrainingState


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
