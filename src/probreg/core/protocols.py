"""Backend-neutral interfaces for data loading and optimization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from probreg.core.types import (
    Batch,
    PyTree,
    StageResult,
    TrainingState,
    ValidationResult,
)


class Dataset(Protocol):
    """An indexable collection of raw examples."""

    def __len__(self) -> int: ...

    def get(self, index: int) -> Mapping[str, Any]: ...


class LoaderFactory(Protocol):
    """Builds a batch loader for a split and epoch."""

    def __call__(self, *, split: str, epoch: int) -> Iterable[Batch]: ...


class Optimizer(Protocol):
    """A functional optimizer with explicit, checkpointable state."""

    def init(self, parameters: PyTree) -> Any: ...

    def update(
        self, gradients: PyTree, state: Any, parameters: PyTree
    ) -> tuple[PyTree, Any]: ...


class Step(Protocol):
    """Executes one training or evaluation step."""

    def __call__(
        self, batch: Batch, state: TrainingState, *, key: Any, training: bool
    ) -> StageResult: ...


class ValidationStrategy(Protocol):
    """Evaluates a trained state according to a caller-defined policy."""

    def __call__(self, state: TrainingState, *, epoch: int) -> ValidationResult: ...
