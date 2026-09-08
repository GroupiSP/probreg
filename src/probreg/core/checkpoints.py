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


class InMemoryCheckpointStore:
    """A :class:`CheckpointStore` that keeps checkpoints in process memory.

    Checkpoints live only for the lifetime of the store, which makes it the
    natural choice for examples, tests, and short runs that only need the
    best checkpoint of the current process restored at the end. Saving
    under an existing key replaces the checkpoint held there.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Store a checkpoint under a key, replacing any checkpoint there.

        Args:
            key: The opaque key to store the checkpoint under.
            checkpoint: The checkpoint to store.

        Returns:
            None.
        """
        self._checkpoints[key] = checkpoint

    def load(self, key: str) -> Checkpoint:
        """Return the checkpoint stored under a key.

        Args:
            key: The key the checkpoint was saved under.

        Returns:
            The stored checkpoint.

        Raises:
            KeyError: If no checkpoint has been saved under ``key``.
        """
        return self._checkpoints[key]

    def exists(self, key: str) -> bool:
        """Report whether a checkpoint is stored under a key.

        Args:
            key: The key to look up.

        Returns:
            ``True`` if a checkpoint is stored under ``key``.
        """
        return key in self._checkpoints
