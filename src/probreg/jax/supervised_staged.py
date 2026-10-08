"""Explicit supervised stages for mean and Gamma variance training."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
from probreg.core.naming import Split, parse_metric_tag
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
from probreg.jax.state import freeze_training_state, restore_checkpoint, snapshot
from probreg.jax.supervised import resolve_checkpoint_key, run_supervised

_STAGE_METADATA_KEY = "stage"
"""Checkpoint metadata key naming the stage that wrote the checkpoint."""

_STAGE_COMPLETE_METADATA_KEY = "stage_complete"
"""Checkpoint metadata key marking a stage's finalized checkpoint."""


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
        checkpoint_key: The checkpoint key the stage saves its best checkpoint
            under. Defaults to ``None``, which resolves to the stage-scoped
            key ``f"{stage}/best"`` (``mean/best``, ``variance/best``), the
            same default as [`run_supervised`][probreg.jax.run_supervised],
            so stages sharing one ``checkpoint_store`` never overwrite each
            other's checkpoints.
        metrics: Batch and epoch metric registrations.
    """

    epochs: int
    validation: ValidationStrategy | None = None
    early_stopper: EarlyStopper | None = None
    event_sinks: Sequence[EventSink] = ()
    checkpoint_store: CheckpointStore | None = None
    checkpoint_key: str | None = None
    metrics: MetricSuite = field(default_factory=MetricSuite)


@dataclass
class MeanStage:
    """Concrete Step 1 stage training a deterministic mean model with MSE.

    Attributes:
        model: NNX model producing deterministic mean predictions.
        optimizer: NNX optimizer bound to ``model``.
        train_loader: Factory producing original regression batches.
        options: Shared supervised-runner options.
        model_name: State registry name for the mean model.
        optimizer_name: State registry name for the mean optimizer.
        loss: Scalar supervised loss used to train the mean model.
    """

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

        With an early stopper and a checkpoint store configured, the stage then
        restores its best checkpoint, so the live mean model holds the best
        epoch's weights, and saves it again under the same key as a finalized
        ``MEAN_READY`` checkpoint with metadata
        ``{"stage": "mean", "stage_complete": True}``.

        Args:
            state: Prepared staged training state.

        Returns:
            The supervised runner result, or the restored epoch's training
            metrics when the best checkpoint was restored.

        Raises:
            ValueError: If the stage was not prepared or training is non-finite.
        """
        if state.lifecycle_state is not StageState.INITIALIZED:
            raise ValueError("mean stage must be prepared before training.")
        return _train_stage(
            self,
            state,
            train_loader=self.train_loader,
            validation=self.options.validation,
            ready=StageState.MEAN_READY,
        )

    def restore(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore the mean stage's finalized checkpoint into ``state``.

        The live model and optimizer take the checkpoint's weights in place and
        are registered under ``model_name`` and ``optimizer_name``. The restore
        is clean-slate, like
        [`restore_checkpoint`][probreg.jax.restore_checkpoint]: other registered
        components and optimizers are dropped. Afterwards the state passes
        [`validate`][probreg.jax.MeanStage.validate], so the variance stage can
        be prepared or restored next.

        Args:
            state: Training state to restore into, typically a fresh one.
            checkpoint: The mean stage's finalized checkpoint.

        Raises:
            ValueError: If ``checkpoint`` is not finalized by this stage, that
                is its lifecycle state is not ``MEAN_READY`` or its metadata
                lacks ``"stage": "mean"`` and ``"stage_complete": True``; this
                is checked before ``state`` or the live objects change. Also
                raised, as by
                [`restore_checkpoint`][probreg.jax.restore_checkpoint], if the
                live model or optimizer is incompatible with the checkpoint,
                which may leave them and ``state`` partially restored.
            TypeError: As by
                [`restore_checkpoint`][probreg.jax.restore_checkpoint], if the
                checkpoint does not hold NNX snapshots or a JAX random key.
        """
        _require_finalized(checkpoint, stage=self.name, ready=StageState.MEAN_READY)
        self._restore_live(state, checkpoint)

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a mean checkpoint into the live model and mark Step 1 ready."""
        restore_checkpoint(
            checkpoint,
            state=state,
            model=self.model,
            optimizer=self.optimizer,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
        )
        state.lifecycle_state = StageState.MEAN_READY
        state.active_stage = self.name

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
        """Return a reference to the mean stage's best checkpoint.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the checkpoint key the stage saves under: the
            configured ``checkpoint_key``, or ``f"{stage}/best"`` when none
            was configured.

        Raises:
            ValueError: If no checkpoint exists under that key.
        """
        del state
        return _select_checkpoint(self)


@dataclass
class GammaVarianceStage:
    """Concrete Step 2 stage fitting Gamma-distributed squared residuals.

    Attributes:
        model: NNX model producing Gamma residual predictions.
        optimizer: NNX optimizer bound to ``model``.
        source_loader: Factory producing original regression batches.
        options: Shared supervised-runner options.
        mean_model_name: Registry name of the prepared mean model.
        mean_optimizer_name: Registry name of the mean optimizer, kept
            registered when the stage restores its best checkpoint.
        model_name: State registry name for the variance model.
        optimizer_name: State registry name for the variance optimizer.
        splits: Source split names materialized as residual batches.
        source_epoch: Source-loader epoch used for residual materialization.
        validation_factory: Optional factory adapting the residual loader into
            a validation strategy.
        loss: Scalar supervised loss used to train the variance model.
    """

    model: nnx.Module
    optimizer: nnx.Optimizer
    source_loader: LoaderFactory
    options: SupervisedStageOptions
    mean_model_name: str = "mean_model"
    mean_optimizer_name: str = "mean_optimizer"
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

        With an early stopper and a checkpoint store configured, the stage then
        restores its best checkpoint, so the live variance model holds the best
        epoch's weights, while the mean model and, if registered, the mean
        optimizer stay registered. It saves the checkpoint again under the same
        key as a finalized ``VARIANCE_READY`` checkpoint with metadata
        ``{"stage": "variance", "stage_complete": True}``.

        Args:
            state: Prepared state retaining a frozen mean component.

        Returns:
            The supervised runner result, or the restored epoch's training
            metrics when the best checkpoint was restored.

        Raises:
            ValueError: If preparation is incomplete or training is non-finite.
        """
        if state.lifecycle_state is not StageState.MEAN_READY:
            raise ValueError("variance stage requires MEAN_READY lifecycle state.")
        if self._residual_loader is None:
            raise ValueError("variance stage must be prepared before training.")
        return _train_stage(
            self,
            state,
            train_loader=self._residual_loader,
            validation=self._validation,
            ready=StageState.VARIANCE_READY,
        )

    def restore(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore the variance stage's finalized checkpoint into ``state``.

        Call it after the mean stage's
        [`restore`][probreg.jax.MeanStage.restore]. The live variance model and
        optimizer take the checkpoint's weights in place and are registered
        under ``model_name`` and ``optimizer_name``. The mean model stays
        registered under ``mean_model_name``, and the mean optimizer under
        ``mean_optimizer_name`` if it was registered before. Afterwards the
        state passes [`validate`][probreg.jax.GammaVarianceStage.validate].

        Args:
            state: Training state the mean stage has been restored into.
            checkpoint: The variance stage's finalized checkpoint.

        Raises:
            ValueError: If ``checkpoint`` is not finalized by this stage, that
                is its lifecycle state is not ``VARIANCE_READY`` or its
                metadata lacks ``"stage": "variance"`` and ``"stage_complete":
                True``, or if no mean model is registered under
                ``mean_model_name``; both are checked before ``state`` or the
                live objects change. Also raised, as by
                [`restore_checkpoint`][probreg.jax.restore_checkpoint], if the
                live model or optimizer is incompatible with the checkpoint,
                which may leave them and ``state`` partially restored.
            TypeError: As by
                [`restore_checkpoint`][probreg.jax.restore_checkpoint], if the
                checkpoint does not hold NNX snapshots or a JAX random key.
        """
        _require_finalized(checkpoint, stage=self.name, ready=StageState.VARIANCE_READY)
        if self.mean_model_name not in state.model_components:
            raise ValueError(
                f"mean model component {self.mean_model_name!r} is not registered."
            )
        self._restore_live(state, checkpoint)

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a variance checkpoint, keeping the mean registrations live.

        The variance checkpoint snapshots only the variance model and optimizer,
        and ``restore_checkpoint`` restores clean-slate, so the mean model and,
        if registered, the mean optimizer are carried across the restore.
        """
        mean_model = state.model_components[self.mean_model_name]
        mean_optimizer = state.optimizer_states.get(self.mean_optimizer_name)
        restore_checkpoint(
            checkpoint,
            state=state,
            model=self.model,
            optimizer=self.optimizer,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
        )
        state.register_component(self.mean_model_name, mean_model)
        if mean_optimizer is not None:
            state.register_optimizer(self.mean_optimizer_name, mean_optimizer)
        state.lifecycle_state = StageState.VARIANCE_READY
        state.active_stage = self.name

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
        """Return a reference to the variance stage's best checkpoint.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the checkpoint key the stage saves under: the
            configured ``checkpoint_key``, or ``f"{stage}/best"`` when none
            was configured.

        Raises:
            ValueError: If no checkpoint exists under that key.
        """
        del state
        return _select_checkpoint(self)


class _CheckpointingStage(Protocol):
    """A concrete supervised stage, as the shared training helpers see it.

    Attributes:
        name: Stage name, scoping metric tags and the default checkpoint key.
        model: The stage's live model.
        optimizer: The stage's live optimizer, bound to ``model``.
        options: The stage's supervised-runner options.
        model_name: State registry name for ``model``.
        optimizer_name: State registry name for ``optimizer``.
        loss: Scalar supervised loss used to train ``model``.
    """

    name: str
    model: nnx.Module
    optimizer: nnx.Optimizer
    options: SupervisedStageOptions
    model_name: str
    optimizer_name: str
    loss: SupervisedLoss

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a checkpoint into ``state`` and mark the stage ready."""
        ...


def _checkpoint_key(stage: _CheckpointingStage) -> str:
    """Return the checkpoint key ``stage`` saves its best checkpoint under."""
    return resolve_checkpoint_key(stage.options.checkpoint_key, stage.name)


def _train_stage(
    stage: _CheckpointingStage,
    state: TrainingState,
    *,
    train_loader: LoaderFactory,
    validation: ValidationStrategy | None,
    ready: StageState,
) -> StageResult:
    """Run a stage's supervised training, mark it ready and finalize its best.

    Args:
        stage: The prepared stage to train.
        state: State the stage has been prepared on.
        train_loader: Factory producing the stage's training batches.
        validation: The stage's validation strategy, if any.
        ready: The lifecycle state the stage reaches once trained.

    Returns:
        The supervised runner result, or the restored epoch's training metrics
        when the best checkpoint was restored.

    Raises:
        ValueError: If training produced a non-finite final loss.
    """
    options = stage.options
    result = run_supervised(
        model=stage.model,
        optimizer=stage.optimizer,
        train_loader=train_loader,
        loss=stage.loss,
        state=state,
        epochs=options.epochs,
        validation=validation,
        early_stopper=options.early_stopper,
        event_sinks=options.event_sinks,
        checkpoint_store=options.checkpoint_store,
        checkpoint_key=_checkpoint_key(stage),
        stage=stage.name,
        model_name=stage.model_name,
        optimizer_name=stage.optimizer_name,
        metrics=options.metrics,
    )
    if result.loss is None or not math.isfinite(result.loss):
        raise ValueError(f"{stage.name} stage produced a non-finite final loss.")
    state.lifecycle_state = ready
    return _restore_and_finalize_best_checkpoint(stage, state, result)


def _select_checkpoint(stage: _CheckpointingStage) -> CheckpointRef:
    """Return a reference to the checkpoint ``stage`` saves its best under.

    Args:
        stage: The stage whose checkpoint to reference.

    Returns:
        Reference to the stage's checkpoint key, with the stage-name metadata.

    Raises:
        ValueError: If no checkpoint exists under that key.
    """
    store = stage.options.checkpoint_store
    key = _checkpoint_key(stage)
    if store is None or not store.exists(key):
        raise ValueError(f"checkpoint {key!r} is not available.")
    return CheckpointRef(key=key, metadata={_STAGE_METADATA_KEY: stage.name})


def _restore_and_finalize_best_checkpoint(
    stage: _CheckpointingStage,
    state: TrainingState,
    result: StageResult,
) -> StageResult:
    """Restore a stage's best checkpoint and save it again as finalized.

    The checkpoint is not finalized yet, so it is restored through the stage's
    unchecked restore path rather than its public ``restore``. Without an
    early stopper, a checkpoint store, or a saved best checkpoint, ``result``
    is returned unchanged.

    Args:
        stage: The stage that has just trained.
        state: State the stage has just trained and marked ready.
        result: The supervised runner result of the stage.

    Returns:
        ``result`` if nothing was restored, otherwise a result reporting the
        restored epoch's training metrics.
    """
    options = stage.options
    store = options.checkpoint_store
    key = _checkpoint_key(stage)
    if options.early_stopper is None or store is None or not store.exists(key):
        return result

    checkpoint = store.load(key)
    stage._restore_live(state, checkpoint)
    finalized = Checkpoint(
        state=freeze_training_state(state),
        epoch=checkpoint.epoch,
        parameters=snapshot(stage.model),
        optimizer_state=snapshot(stage.optimizer),
        rng_state=state.rng_state,
        early_stopping_state=checkpoint.early_stopping_state,
        metadata={
            **checkpoint.metadata,
            _STAGE_METADATA_KEY: stage.name,
            _STAGE_COMPLETE_METADATA_KEY: True,
        },
    )
    store.save(key, finalized)
    metrics = _latest_training_metrics(state, stage.name)
    return StageResult(state=state, metrics=metrics, loss=metrics["loss"])


def _require_finalized(
    checkpoint: Checkpoint,
    *,
    stage: str,
    ready: StageState,
) -> None:
    """Reject a checkpoint that is not the named stage's finalized checkpoint.

    Args:
        checkpoint: The checkpoint a stage is asked to restore.
        stage: Name of the stage that must have finalized ``checkpoint``.
        ready: The lifecycle state ``stage`` finalizes its checkpoint in.

    Raises:
        ValueError: If the checkpoint's lifecycle state is not ``ready``, or
            its metadata does not name ``stage`` and mark it complete.
    """
    if (
        checkpoint.state.lifecycle_state is not ready
        or checkpoint.metadata.get(_STAGE_METADATA_KEY) != stage
        or checkpoint.metadata.get(_STAGE_COMPLETE_METADATA_KEY) is not True
    ):
        raise ValueError(
            f"checkpoint is not a finalized {stage!r} checkpoint: expected "
            f"lifecycle state {ready.value!r} and metadata "
            f"{{{_STAGE_METADATA_KEY!r}: {stage!r}, "
            f"{_STAGE_COMPLETE_METADATA_KEY!r}: True}}."
        )


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


def _latest_training_metrics(
    state: TrainingState,
    stage_name: str,
) -> dict[str, float]:
    """Return the latest stage training metrics from persisted history.

    History keys that are not metric tags, such as ones a caller recorded
    directly, are skipped.

    Args:
        state: State whose ``metric_history`` is keyed by metric tags.
        stage_name: Stage whose training metrics to return.

    Returns:
        The last recorded value of every training metric of the stage,
        keyed by bare metric name.

    Raises:
        ValueError: If the stage recorded no training loss.
    """
    metrics = {}
    for tag, values in state.metric_history.items():
        try:
            parsed = parse_metric_tag(tag)
        except ValueError:
            continue
        if parsed.stage == stage_name and parsed.split is Split.TRAIN and values:
            metrics[parsed.metric] = values[-1]
    if "loss" not in metrics:
        raise ValueError("selected checkpoint does not contain a training loss.")
    return metrics
