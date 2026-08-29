from __future__ import annotations

import pytest

from probreg.core.stages import StageState, validate_transition
from probreg.core.types import (
    CheckpointRef,
    StageResult,
    TrainingState,
    ValidationResult,
)


class ExampleStage:
    name = "mean"
    requires = frozenset()
    produces = frozenset({"mean"})

    def prepare(self, state: TrainingState) -> None:
        state.stage = self.name

    def train(self, state: TrainingState) -> StageResult:
        return StageResult(state=state, loss=0.0)

    def validate(self, state: TrainingState) -> ValidationResult:
        return ValidationResult(passed=state.stage == self.name)

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        del state
        return CheckpointRef("mean-best")


@pytest.mark.parametrize(
    ("current", "next_state"),
    list(zip(tuple(StageState)[:-1], tuple(StageState)[1:], strict=True)),
)
def test_validate_transition_accepts_adjacent_lifecycle_states(
    current: StageState, next_state: StageState
) -> None:
    validate_transition(current, next_state)


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (StageState.NEW, StageState.MEAN_READY),
        (StageState.MEAN_READY, StageState.MEAN_READY),
        (StageState.COMPLETED, StageState.NEW),
    ],
)
def test_validate_transition_rejects_invalid_lifecycle_states(
    current: StageState, next_state: StageState
) -> None:
    with pytest.raises(ValueError, match="Cannot transition"):
        validate_transition(current, next_state)


def test_training_stage_contract_has_a_complete_lifecycle() -> None:
    stage = ExampleStage()
    state = TrainingState()

    stage.prepare(state)

    assert stage.train(state).loss == 0.0
    assert stage.validate(state).passed
    assert stage.select_checkpoint(state).key == "mean-best"
