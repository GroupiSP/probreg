"""Training-stage lifecycle contracts and transition validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from probreg.core.types import (
    CheckpointRef,
    StageResult,
    TrainingState,
    ValidationResult,
)


class StageState(StrEnum):
    """The ordered lifecycle of a staged probabilistic-regression workflow."""

    NEW = "new"
    INITIALIZED = "initialized"
    MEAN_READY = "mean_ready"
    VARIANCE_READY = "variance_ready"
    POSTERIOR_READY = "posterior_ready"
    COMPLETED = "completed"


def validate_transition(current: StageState, next_state: StageState) -> None:
    """Raise ``ValueError`` unless ``next_state`` directly follows ``current``."""
    lifecycle = tuple(StageState)
    current_index = lifecycle.index(current)
    expected_state = (
        lifecycle[current_index + 1] if current_index + 1 < len(lifecycle) else None
    )
    if next_state is not expected_state:
        expected = (
            expected_state.value if expected_state is not None else "no further state"
        )
        raise ValueError(
            f"Cannot transition from {current.value!r} to {next_state.value!r}; "
            f"expected {expected!r}."
        )


class TrainingStage(Protocol):
    """A named unit of training with declared prerequisites and outputs."""

    name: str
    requires: frozenset[str]
    produces: frozenset[str]

    def prepare(self, state: TrainingState) -> None: ...

    def train(self, state: TrainingState) -> StageResult: ...

    def validate(self, state: TrainingState) -> ValidationResult: ...

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef: ...
