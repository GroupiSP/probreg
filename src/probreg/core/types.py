"""Backend-neutral value objects shared by probabilistic-regression stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Backends supply concrete array and tree implementations in their adapters.
Array = Any
PyTree = Any


class ParameterRole(StrEnum):
    """The responsibility of a trainable model component."""

    MEAN = "mean"
    VARIANCE = "variance"
    POSTERIOR = "posterior"
    AUXILIARY = "auxiliary"


class StageState(StrEnum):
    """The ordered lifecycle of a staged probabilistic-regression workflow."""

    NEW = "new"
    INITIALIZED = "initialized"
    MEAN_READY = "mean_ready"
    VARIANCE_READY = "variance_ready"
    POSTERIOR_READY = "posterior_ready"
    COMPLETED = "completed"


@dataclass(frozen=True)
class Batch:
    """A batch of model inputs, optional targets, weights, and metadata."""

    inputs: PyTree
    targets: PyTree | None = None
    sample_weight: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointRef:
    """An opaque reference to a persisted checkpoint."""

    key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """State shared across one or more explicit training stages."""

    model_components: dict[str, Any] = field(default_factory=dict)
    parameter_roles: dict[str, ParameterRole] = field(default_factory=dict)
    frozen_components: frozenset[str] = field(default_factory=frozenset)
    optimizer_states: dict[str, Any] = field(default_factory=dict)
    posterior_state: Any | None = None
    rng_state: Any | None = None
    lifecycle_state: StageState = StageState.NEW
    active_stage: str | None = None
    outer_iteration: int = 0
    data_fingerprint: str | None = None
    checkpoint_registry: dict[str, CheckpointRef] = field(default_factory=dict)
    metric_history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def stage(self) -> str | None:
        """Return the active stage label kept for compatibility.

        Returns:
            The active training-stage label, or ``None`` outside a runner.
        """
        return self.active_stage

    @stage.setter
    def stage(self, value: str | None) -> None:
        """Set the active stage label kept for compatibility.

        Args:
            value: The active training-stage label, or ``None``.
        """
        self.active_stage = value

    def register_component(self, name: str, component: Any) -> None:
        """Register a model component under ``name``.

        Args:
            name: The key under which ``component`` is stored in
                ``model_components``.
            component: The model component to register.

        Raises:
            ValueError: If ``name`` is already bound to a different object.
        """
        registered = self.model_components.get(name)
        if name in self.model_components and registered is not component:
            raise ValueError(f"model component {name!r} is already registered.")
        self.model_components[name] = component

    def register_optimizer(self, name: str, optimizer: Any) -> None:
        """Register an optimizer state under ``name``.

        Args:
            name: The key under which ``optimizer`` is stored in
                ``optimizer_states``.
            optimizer: The optimizer state to register.

        Raises:
            ValueError: If ``name`` is already bound to a different object.
        """
        registered = self.optimizer_states.get(name)
        if name in self.optimizer_states and registered is not optimizer:
            raise ValueError(f"optimizer state {name!r} is already registered.")
        self.optimizer_states[name] = optimizer

    def record_metric(self, name: str, value: float) -> None:
        """Append ``value`` to the metric history recorded under ``name``.

        Args:
            name: The metric name whose history is appended to.
            value: The observed metric value to append.
        """
        self.metric_history.setdefault(name, []).append(value)


@dataclass(frozen=True)
class StageResult:
    """The outcome of executing a training stage."""

    state: TrainingState
    metrics: Mapping[str, float] = field(default_factory=dict)
    loss: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating a training stage."""

    passed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    message: str | None = None
