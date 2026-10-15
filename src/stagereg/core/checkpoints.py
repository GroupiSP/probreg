"""Checkpoint value objects and persistence protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from stagereg.core.types import TrainingState

STAGE_COMPLETE_METADATA_KEY = "stage_complete"
"""Checkpoint metadata key marking a stage's finalized checkpoint.

A stage saves its finalized checkpoint with this key set to ``True`` in its
[`metadata`][stagereg.core.Checkpoint], next to the name of the stage. Its best
checkpoints, saved on every improvement during training, do not carry it.
"""

STAGE_METADATA_KEY = "stage"
"""Checkpoint metadata key naming the stage that wrote the checkpoint."""


@dataclass(frozen=True)
class Checkpoint:
    """The complete state necessary to resume staged training.

    Attributes:
        state: Snapshot of the [`TrainingState`][stagereg.core.TrainingState] at
            the time the checkpoint was saved. Its ``rng_state`` is the random
            key a restore resumes training with.
        epoch: Epoch of the active stage at which the checkpoint was saved.
        parameters: Snapshot of the model's parameters, in the backend's own
            format. Defaults to ``None``.
        optimizer_state: Snapshot of the optimizer's state, in the backend's own
            format. Defaults to ``None``.
        early_stopping_state: State of the early stopper, such as an
            [`EarlyStoppingState`][stagereg.core.EarlyStoppingState], so that
            patience and the best value carry over on resume. Defaults to
            ``None``.
        metadata: Information about the checkpoint, such as the stage that saved
            it. Defaults to an empty mapping.
    """

    state: TrainingState
    epoch: int
    parameters: Any | None = None
    optimizer_state: Any | None = None
    early_stopping_state: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CheckpointStore(Protocol):
    """Persists checkpoints under opaque keys."""

    def save(self, key: str, checkpoint: Checkpoint) -> None: ...

    def load(self, key: str) -> Checkpoint: ...

    def exists(self, key: str) -> bool: ...


class InMemoryCheckpointStore:
    """A [`CheckpointStore`][stagereg.core.CheckpointStore] kept in process memory.

    Checkpoints live only for the lifetime of the store, which makes it the natural
    choice for examples, tests, and short runs that only need the best checkpoint of the
    current process restored at the end. Saving under an existing key replaces the
    checkpoint held there.
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


def is_finalized(checkpoint: Checkpoint) -> bool:
    """Report whether a checkpoint is a stage's finalized checkpoint.

    Args:
        checkpoint: The checkpoint to inspect.

    Returns:
        ``True`` if its metadata maps
        [`STAGE_COMPLETE_METADATA_KEY`][stagereg.core.STAGE_COMPLETE_METADATA_KEY]
        to ``True``.
    """
    return checkpoint.metadata.get(STAGE_COMPLETE_METADATA_KEY) is True


PersistencePredicate = Callable[[str, Checkpoint], bool]
"""Decides, from its key and value, whether a save reaches the persistent store."""


def finalized_only(key: str, checkpoint: Checkpoint) -> bool:
    """Persist only finalized checkpoints.

    Args:
        key: The key the checkpoint is saved under.
        checkpoint: The checkpoint being saved.

    Returns:
        Whether ``checkpoint`` [is finalized][stagereg.core.is_finalized].
    """
    del key
    return is_finalized(checkpoint)


def always(key: str, checkpoint: Checkpoint) -> bool:
    """Persist every save.

    Args:
        key: The key the checkpoint is saved under.
        checkpoint: The checkpoint being saved.

    Returns:
        ``True``.
    """
    del key, checkpoint
    return True


def every_nth_save(n: int) -> PersistencePredicate:
    """Return a predicate that persists every ``n``-th save under each key.

    The predicate counts saves per key, so the ``n``-th, ``2n``-th, ... save
    under a key is persisted. The count lives in the returned predicate: give
    each [`CachedCheckpointStore`][stagereg.core.CachedCheckpointStore] its own.

    Args:
        n: The period, in saves under one key.

    Returns:
        A stateful persistence predicate.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}.")
    counts: dict[str, int] = {}

    def persist(key: str, checkpoint: Checkpoint) -> bool:
        del checkpoint
        counts[key] = counts.get(key, 0) + 1
        return counts[key] % n == 0

    return persist


class CachedCheckpointStore:
    """A [`CheckpointStore`][stagereg.core.CheckpointStore] that persists selectively.

    Every save lands in an in-memory cache, and only the saves ``persist_when``
    selects are forwarded to the wrapped ``persistent`` store. Early stopping can
    then save a best checkpoint on every improvement without each one reaching a
    slow or remote store. Reads try the cache first and fall back to the
    persistent store, so a fresh cache still loads what an earlier process
    persisted.

    Args:
        persistent: The store selected saves are forwarded to.
        persist_when: Called with the key and checkpoint of every save; the save is
            forwarded when it returns ``True``. Defaults to
            [`finalized_only`][stagereg.core.finalized_only].
    """

    def __init__(
        self,
        persistent: CheckpointStore,
        persist_when: PersistencePredicate = finalized_only,
    ) -> None:
        self._persistent = persistent
        self._persist_when = persist_when
        self._cache = InMemoryCheckpointStore()

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Cache a checkpoint, and persist it if the predicate selects it.

        Args:
            key: The opaque key to store the checkpoint under.
            checkpoint: The checkpoint to store.

        Returns:
            None.
        """
        self._cache.save(key, checkpoint)
        if self._persist_when(key, checkpoint):
            self._persistent.save(key, checkpoint)

    def load(self, key: str) -> Checkpoint:
        """Return the checkpoint under a key, from the cache or else the store.

        Args:
            key: The key the checkpoint was saved under.

        Returns:
            The cached checkpoint, or the persisted one if none is cached.

        Raises:
            KeyError: If neither the cache nor the persistent store holds ``key``.
        """
        if self._cache.exists(key):
            return self._cache.load(key)
        if not self._persistent.exists(key):
            raise KeyError(key)
        return self._persistent.load(key)

    def exists(self, key: str) -> bool:
        """Report whether the cache or the persistent store holds a key.

        Args:
            key: The key to look up.

        Returns:
            ``True`` if a checkpoint is cached or persisted under ``key``.
        """
        return self._cache.exists(key) or self._persistent.exists(key)
