"""Checkpoint value objects and persistence protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from probreg.core.types import TrainingState


@dataclass(frozen=True)
class Checkpoint:
    """The complete state necessary to resume staged training."""

    state: TrainingState
    epoch: int
    parameters: Any | None = None
    optimizer_state: Any | None = None
    rng_state: Any | None = None
    early_stopping_state: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CheckpointStore(Protocol):
    """Persists checkpoints under opaque keys."""

    def save(self, key: str, checkpoint: Checkpoint) -> None: ...

    def load(self, key: str) -> Checkpoint: ...

    def exists(self, key: str) -> bool: ...
