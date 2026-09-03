"""JAX objectives for explicit mean and Gamma variance training stages."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.checkpoints import CheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
from probreg.core.protocols import LoaderFactory, ValidationStrategy
from probreg.core.tracking import EventSink
from probreg.core.types import (
    Batch,
    CheckpointRef,
    ParameterRole,
    StageResult,
    StageState,
    TrainingState,
    ValidationResult,
)
from probreg.jax.evaluation import SupervisedLoss
from probreg.jax.losses import make_supervised_loss
from probreg.jax.metrics import MetricSuite
from probreg.jax.supervised import run_supervised


def materialize_residual_loader(
    mean_model: nnx.Module,
    source_loader: LoaderFactory,
    *,
    splits: Sequence[str] = ("train", "validation"),
    source_epoch: int = 0,
) -> LoaderFactory:
    """Materialize detached squared residual targets for fixed data splits.

    The source loader is consumed exactly once per configured split. The
    returned loader replays the cached batches for every later epoch, making
    the Step 1 mean predictions and source data a fixed Step 2 training
    snapshot.

    Args:
        mean_model: Trained deterministic mean model.
        source_loader: Loader providing inputs and original regression targets.
        splits: Split names to materialize. Defaults to training and validation.
        source_epoch: Source-loader epoch used for the one-time snapshot.

    Returns:
        A loader factory serving cached batches whose targets are detached
        squared residuals.

    Raises:
        ValueError: If splits are empty, duplicated, unknown at replay time, or
            produce no batches; if a source batch lacks targets; or if mean
            predictions and targets have different shapes.
    """
    split_names = tuple(splits)
    if not split_names:
        raise ValueError("splits must contain at least one split name.")
    if any(not name for name in split_names):
        raise ValueError("split names must not be empty.")
    if len(set(split_names)) != len(split_names):
        raise ValueError("split names must be unique.")

    prediction_model = nnx.clone(mean_model)
    prediction_model.eval()
    cached: dict[str, tuple[Batch, ...]] = {}

    for split in split_names:
        residual_batches = tuple(
            _materialize_residual_batch(prediction_model, batch)
            for batch in source_loader(split=split, epoch=source_epoch)
        )
        if not residual_batches:
            raise ValueError(f"source split {split!r} must provide at least one batch.")
        cached[split] = residual_batches

    def residual_loader(*, split: str, epoch: int) -> tuple[Batch, ...]:
        del epoch
        if split not in cached:
            raise ValueError(f"residual split {split!r} was not materialized.")
        return cached[split]

    return residual_loader


def _materialize_residual_batch(mean_model: nnx.Module, batch: Batch) -> Batch:
    """Return one batch with detached squared residual targets."""
    if batch.targets is None:
        raise ValueError("source batches must provide targets.")
    predictions = mean_model(batch.inputs)
    if predictions.shape != batch.targets.shape:
        raise ValueError("mean predictions and targets must have matching shapes.")
    residuals = jax.lax.stop_gradient(jnp.square(batch.targets - predictions))
    return Batch(
        inputs=batch.inputs,
        targets=residuals,
        sample_weight=batch.sample_weight,
        metadata=batch.metadata,
    )


@dataclass(frozen=True)
class SupervisedStageOptions:
    """Shared epoch-runner options for concrete supervised stages.

    Attributes:
        epochs: Maximum number of training epochs.
        validation: Optional validation strategy.
        early_stopper: Optional early-stopping policy.
        event_sinks: Event consumers notified by the runner.
        checkpoint_store: Optional best-checkpoint store.
        checkpoint_key: Explicit caller-selected best-checkpoint key.
        metrics: Batch and epoch metric registrations.
    """

    epochs: int
    validation: ValidationStrategy | None = None
    early_stopper: EarlyStopper | None = None
    event_sinks: Sequence[EventSink] = ()
    checkpoint_store: CheckpointStore | None = None
    checkpoint_key: str = "best"
    metrics: MetricSuite = field(default_factory=MetricSuite)


@dataclass
class MeanStage:
    """Concrete Step 1 stage training a deterministic mean model with MSE."""

    model: nnx.Module
    optimizer: nnx.Optimizer
    train_loader: LoaderFactory
    options: SupervisedStageOptions
    model_name: str = "mean_model"
    optimizer_name: str = "mean_optimizer"
    loss: SupervisedLoss = field(
        default_factory=lambda: make_supervised_loss(SquaredErrorLoss())
    )
    name: str = field(default="mean", init=False)
    requires: frozenset[str] = field(default_factory=frozenset, init=False)
    produces: frozenset[str] = field(
        default_factory=lambda: frozenset({"mean"}),
        init=False,
    )

    def prepare(self, state: TrainingState) -> None:
        """Register Step 1 ownership and initialize the staged lifecycle.

        Args:
            state: Shared staged training state.

        Raises:
            ValueError: If the lifecycle or component ownership is invalid.
        """
        if state.lifecycle_state not in (StageState.NEW, StageState.INITIALIZED):
            raise ValueError("mean stage requires NEW or INITIALIZED lifecycle state.")
        _validate_named_registration(
            state.model_components,
            self.model_name,
            self.model,
            kind="model component",
        )
        _validate_named_registration(
            state.optimizer_states,
            self.optimizer_name,
            self.optimizer,
            kind="optimizer state",
        )
        _validate_parameter_role(state, self.model_name, ParameterRole.MEAN)
        state.register_component(self.model_name, self.model)
        state.register_optimizer(self.optimizer_name, self.optimizer)
        state.parameter_roles[self.model_name] = ParameterRole.MEAN
        state.lifecycle_state = StageState.INITIALIZED
        state.active_stage = self.name

    def train(self, state: TrainingState) -> StageResult:
        """Train the mean model and transition the workflow to ``MEAN_READY``.

        Args:
            state: Prepared staged training state.

        Returns:
            The supervised runner result.

        Raises:
            ValueError: If the stage was not prepared or training is non-finite.
        """
        if state.lifecycle_state is not StageState.INITIALIZED:
            raise ValueError("mean stage must be prepared before training.")
        result = run_supervised(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            loss=self.loss,
            state=state,
            epochs=self.options.epochs,
            validation=self.options.validation,
            early_stopper=self.options.early_stopper,
            event_sinks=self.options.event_sinks,
            checkpoint_store=self.options.checkpoint_store,
            checkpoint_key=self.options.checkpoint_key,
            stage=self.name,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            metrics=self.options.metrics,
        )
        if result.loss is None or not math.isfinite(result.loss):
            raise ValueError("mean stage produced a non-finite final loss.")
        state.lifecycle_state = StageState.MEAN_READY
        return result

    def validate(self, state: TrainingState) -> ValidationResult:
        """Validate mean-stage lifecycle and ownership invariants.

        Args:
            state: Shared staged training state.

        Returns:
            A validation result describing whether Step 1 is ready.
        """
        passed = (
            state.lifecycle_state is StageState.MEAN_READY
            and state.model_components.get(self.model_name) is self.model
            and state.optimizer_states.get(self.optimizer_name) is self.optimizer
            and state.parameter_roles.get(self.model_name) is ParameterRole.MEAN
        )
        return ValidationResult(
            passed=passed,
            message=None if passed else "mean stage invariants are not satisfied.",
        )

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        """Return the explicitly configured mean checkpoint reference.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the configured best checkpoint.

        Raises:
            ValueError: If no configured checkpoint exists.
        """
        del state
        store = self.options.checkpoint_store
        key = self.options.checkpoint_key
        if store is None or not store.exists(key):
            raise ValueError(f"checkpoint {key!r} is not available.")
        return CheckpointRef(key=key, metadata={"stage": self.name})


@dataclass
class GammaVarianceStage:
    """Concrete Step 2 stage fitting Gamma-distributed squared residuals."""

    model: nnx.Module
    optimizer: nnx.Optimizer
    source_loader: LoaderFactory
    options: SupervisedStageOptions
    mean_model_name: str = "mean_model"
    model_name: str = "variance_model"
    optimizer_name: str = "variance_optimizer"
    splits: Sequence[str] = ("train", "validation")
    source_epoch: int = 0
    validation_factory: Callable[[LoaderFactory], ValidationStrategy] | None = None
    loss: SupervisedLoss = field(
        default_factory=lambda: make_supervised_loss(
            NegativeLogLikelihoodLoss(target_transform=add_epsilon())
        )
    )
    name: str = field(default="variance", init=False)
    requires: frozenset[str] = field(
        default_factory=lambda: frozenset({"mean"}),
        init=False,
    )
    produces: frozenset[str] = field(
        default_factory=lambda: frozenset({"variance"}),
        init=False,
    )
    _residual_loader: LoaderFactory | None = field(default=None, init=False, repr=False)
    _validation: ValidationStrategy | None = field(default=None, init=False, repr=False)

    def prepare(self, state: TrainingState) -> None:
        """Validate Step 1 output and materialize fixed residual targets.

        Args:
            state: Shared state whose mean component is ready.

        Raises:
            TypeError: If the registered mean component is not an NNX module.
            ValueError: If lifecycle, ownership, or registrations are invalid.
        """
        if state.lifecycle_state is not StageState.MEAN_READY:
            raise ValueError("variance stage requires MEAN_READY lifecycle state.")
        if self.mean_model_name not in state.model_components:
            raise ValueError(
                f"mean model component {self.mean_model_name!r} is not registered."
            )
        mean_model = state.model_components[self.mean_model_name]
        if not isinstance(mean_model, nnx.Module):
            raise TypeError("registered mean model must be an NNX module.")
        if state.parameter_roles.get(self.mean_model_name) is not ParameterRole.MEAN:
            raise ValueError("registered mean model must have the MEAN parameter role.")
        _validate_named_registration(
            state.model_components,
            self.model_name,
            self.model,
            kind="model component",
        )
        _validate_named_registration(
            state.optimizer_states,
            self.optimizer_name,
            self.optimizer,
            kind="optimizer state",
        )
        _validate_parameter_role(state, self.model_name, ParameterRole.VARIANCE)

        residual_loader = materialize_residual_loader(
            mean_model,
            self.source_loader,
            splits=self.splits,
            source_epoch=self.source_epoch,
        )
        validation = (
            self.validation_factory(residual_loader)
            if self.validation_factory is not None
            else self.options.validation
        )

        state.register_component(self.model_name, self.model)
        state.register_optimizer(self.optimizer_name, self.optimizer)
        state.parameter_roles[self.model_name] = ParameterRole.VARIANCE
        state.frozen_components = state.frozen_components | {self.mean_model_name}
        state.active_stage = self.name
        self._residual_loader = residual_loader
        self._validation = validation

    def train(self, state: TrainingState) -> StageResult:
        """Train the variance model and transition to ``VARIANCE_READY``.

        Args:
            state: Prepared state retaining a frozen mean component.

        Returns:
            The supervised runner result.

        Raises:
            ValueError: If preparation is incomplete or training is non-finite.
        """
        if state.lifecycle_state is not StageState.MEAN_READY:
            raise ValueError("variance stage requires MEAN_READY lifecycle state.")
        if self._residual_loader is None:
            raise ValueError("variance stage must be prepared before training.")
        result = run_supervised(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self._residual_loader,
            loss=self.loss,
            state=state,
            epochs=self.options.epochs,
            validation=self._validation,
            early_stopper=self.options.early_stopper,
            event_sinks=self.options.event_sinks,
            checkpoint_store=self.options.checkpoint_store,
            checkpoint_key=self.options.checkpoint_key,
            stage=self.name,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            metrics=self.options.metrics,
        )
        if result.loss is None or not math.isfinite(result.loss):
            raise ValueError("variance stage produced a non-finite final loss.")
        state.lifecycle_state = StageState.VARIANCE_READY
        return result

    def validate(self, state: TrainingState) -> ValidationResult:
        """Validate variance-stage lifecycle, ownership, and freezing.

        Args:
            state: Shared staged training state.

        Returns:
            A validation result describing whether Step 2 is ready.
        """
        passed = (
            state.lifecycle_state is StageState.VARIANCE_READY
            and state.model_components.get(self.model_name) is self.model
            and state.optimizer_states.get(self.optimizer_name) is self.optimizer
            and state.parameter_roles.get(self.model_name) is ParameterRole.VARIANCE
            and self.mean_model_name in state.frozen_components
        )
        return ValidationResult(
            passed=passed,
            message=None if passed else "variance stage invariants are not satisfied.",
        )

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        """Return the explicitly configured variance checkpoint reference.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the configured best checkpoint.

        Raises:
            ValueError: If no configured checkpoint exists.
        """
        del state
        store = self.options.checkpoint_store
        key = self.options.checkpoint_key
        if store is None or not store.exists(key):
            raise ValueError(f"checkpoint {key!r} is not available.")
        return CheckpointRef(key=key, metadata={"stage": self.name})


def _validate_named_registration(
    registry: dict[str, Any],
    name: str,
    value: Any,
    *,
    kind: str,
) -> None:
    """Reject a conflicting named object without mutating the registry."""
    if name in registry and registry[name] is not value:
        raise ValueError(f"{kind} {name!r} is already registered.")


def _validate_parameter_role(
    state: TrainingState,
    component_name: str,
    role: ParameterRole,
) -> None:
    """Reject conflicting component ownership without mutating state."""
    registered = state.parameter_roles.get(component_name)
    if registered is not None and registered is not role:
        raise ValueError(
            f"model component {component_name!r} already has role {registered.value!r}."
        )
