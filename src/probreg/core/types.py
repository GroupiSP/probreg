"""Backend-neutral value objects shared by probabilistic-regression stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Backends supply concrete array and tree implementations in their adapters.
Array = Any
"""An array of the active backend, such as a `jax.Array`.

Typed as `Any` because `probreg.core` imports no backend.
"""

PyTree = Any
"""A nested container of arrays of the active backend, such as model parameters.

Typed as `Any` because `probreg.core` imports no backend.
"""


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
    """A batch of model inputs, optional targets, weights, and metadata.

    Attributes:
        inputs: Model inputs, an array or nested container of arrays with a leading
            batch dimension.
        targets: Regression targets aligned with ``inputs``. Defaults to ``None``,
            for batches that are only predicted on.
        sample_weight: Weights with a leading batch dimension, one per row, scaling
            each row's contribution to the loss. Defaults to ``None``, which weights
            every row equally.
        metadata: Caller-supplied information about the batch, carried along
            unchanged. Defaults to an empty mapping.
    """

    inputs: PyTree
    targets: PyTree | None = None
    sample_weight: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointRef:
    """An opaque reference to a persisted checkpoint.

    Attributes:
        key: Key the checkpoint is stored under in a
            [`CheckpointStore`][probreg.core.CheckpointStore]. It identifies the
            checkpoint and carries no meaning of its own.
        metadata: Information about the checkpoint, such as the stage that saved
            it. Defaults to an empty mapping.
    """

    key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """State shared across one or more explicit training stages.

    Attributes:
        model_components: Model objects keyed by component name, as registered
            with [`register_component`][probreg.core.TrainingState.register_component].
            Defaults to an empty mapping.
        parameter_roles: The [`ParameterRole`][probreg.core.ParameterRole] of each
            registered component, keyed by component name, so a stage can check
            the prerequisites an earlier stage left behind. Defaults to an empty
            mapping.
        frozen_components: Names of components that are no longer trained, such
            as a fitted mean model while a later stage trains the variance.
            Defaults to an empty set.
        optimizer_states: Optimizer states keyed by name, as registered with
            [`register_optimizer`][probreg.core.TrainingState.register_optimizer].
            Defaults to an empty mapping.
        posterior_state: State of the posterior-approximation stage, set once
            ``lifecycle_state`` reaches
            [`StageState.POSTERIOR_READY`][probreg.core.StageState]. Defaults
            to ``None``.
        rng_state: Random key threaded through training; seed it to make a run
            reproducible. Defaults to ``None``.
        lifecycle_state: Position of the workflow in the ordered
            [`StageState`][probreg.core.StageState] lifecycle. Defaults to
            [`StageState.NEW`][probreg.core.StageState].
        stage: Label of the active stage, the value behind the
            [`active_stage`][probreg.core.TrainingState.active_stage] property.
            Defaults to ``None`` outside a runner.
        outer_iteration: Zero-based outer iteration: the count of passes through
            the run's sequence of stages. Defaults to ``0``.
        data_fingerprint: Identifier of the dataset the state was trained on,
            meant to detect resuming against different data. Set by the user and
            not interpreted by the library. Defaults to ``None``.
        checkpoint_registry: Named references to checkpoints a stage registered,
            such as its best checkpoint. Defaults to an empty mapping.
        metric_history: Recorded values of each metric in order, keyed by metric
            tag (``stage/split/metric``), the same tags the experiment tracker
            uses. Defaults to an empty mapping.
    """

    model_components: dict[str, Any] = field(default_factory=dict)
    parameter_roles: dict[str, ParameterRole] = field(default_factory=dict)
    frozen_components: frozenset[str] = field(default_factory=frozenset)
    optimizer_states: dict[str, Any] = field(default_factory=dict)
    posterior_state: Any | None = None
    rng_state: Any | None = None
    lifecycle_state: StageState = StageState.NEW
    stage: str | None = None
    outer_iteration: int = 0
    data_fingerprint: str | None = None
    checkpoint_registry: dict[str, CheckpointRef] = field(default_factory=dict)
    metric_history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def active_stage(self) -> str | None:
        """Return the explicit active training-stage label.

        Returns:
            The active training-stage label, or ``None`` outside a runner.
        """
        return self.stage

    @active_stage.setter
    def active_stage(self, value: str | None) -> None:
        """Set the explicit active training-stage label.

        Args:
            value: The active training-stage label, or ``None``.
        """
        self.stage = value

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
    """The outcome of executing a training stage.

    Attributes:
        state: Training state the stage left behind.
        metrics: Final metric values of the stage, keyed by metric name. Defaults
            to an empty mapping.
        loss: Loss of the stage's last epoch on the ``train`` split. Defaults to
            ``None`` when the stage reports none.
    """

    state: TrainingState
    metrics: Mapping[str, float] = field(default_factory=dict)
    loss: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating a training stage.

    Attributes:
        passed: Whether the stage's invariants hold.
        metrics: Metric values measured during validation, keyed by metric name.
            Defaults to an empty mapping.
        message: Explanation of the outcome, typically of a failure. Defaults to
            ``None``.
    """

    passed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    message: str | None = None
