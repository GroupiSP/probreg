"""A single-model NNX supervised training runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
from flax import nnx

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.protocols import LoaderFactory, ValidationStrategy
from probreg.core.tracking import EventSink, TrainingEvent
from probreg.core.types import StageResult, TrainingState
from probreg.jax.evaluation import SupervisedLoss
from probreg.jax.metrics import (
    BatchMetricSpec,
    MetricSuite,
    collect_step_metrics,
    initialize_batch_metric_values,
    maybe_collect_epoch_metric_inputs,
    reduce_metric_suite,
)
from probreg.jax.rng import split_key
from probreg.jax.state import freeze_training_state, snapshot


def make_train_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX/Optax supervised training step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.

    Returns:
        A JIT-compiled function ``train_step(model, optimizer, inputs,
        targets, sample_weight, key)`` that performs one gradient update
        in place and returns a mapping containing ``"loss"`` plus
        registered batch metric values. Loss is evaluated on the
        pre-update training-state model, while batch metrics are computed
        on the same pre-update parameters with ``training=False`` to
        avoid a second state-mutating training-mode forward pass.
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
    ) -> Mapping[str, jax.Array]:
        def loss_fn(current_model: nnx.Module) -> jax.Array:
            return loss(
                current_model,
                inputs,
                targets,
                sample_weight,
                key,
                True,
            )

        loss_value, gradients = nnx.value_and_grad(loss_fn)(model)
        values: dict[str, jax.Array] = {"loss": loss_value}
        for spec in metrics:
            values[spec.name] = spec.metric(
                model,
                inputs,
                targets,
                sample_weight,
                key,
                False,
            )
        optimizer.update(model, gradients)
        return values

    return train_step


def run_supervised(
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_loader: LoaderFactory,
    loss: SupervisedLoss,
    state: TrainingState,
    epochs: int,
    validation: ValidationStrategy | None = None,
    early_stopper: EarlyStopper | None = None,
    event_sinks: Sequence[EventSink] = (),
    checkpoint_store: CheckpointStore | None = None,
    checkpoint_key: str = "best",
    stage: str = "supervised",
    metrics: MetricSuite = MetricSuite(),
) -> StageResult:
    """Train a single NNX model for a fixed or early-stopped number of epochs.

    Args:
        model: The NNX module to train, mutated in place.
        optimizer: The NNX optimizer used to update ``model``, mutated
            in place.
        train_loader: Factory producing the training batch loader for a
            given split and epoch.
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.
        state: The training state to update in place across epochs.
        epochs: The maximum number of epochs to run.
        validation: An optional strategy used to evaluate ``state``
            after each epoch. Required if ``early_stopper`` monitors a
            validation metric.
        early_stopper: An optional policy that stops training early
            based on a monitored training or validation metric.
        event_sinks: Sinks notified of epoch, validation, best-model,
            and early-stop events.
        checkpoint_store: An optional store used to persist the best
            checkpoint when ``early_stopper`` reports an improvement.
        checkpoint_key: The key under which the best checkpoint is
            saved. Defaults to ``"best"``.
        stage: The stage name recorded on ``state`` and emitted events.
            Defaults to ``"supervised"``.
        metrics: Registered batch/epoch metrics for training.

    Returns:
        A :class:`StageResult` with the final ``state``, the last
        recorded training metrics, and the final training loss.

    Raises:
        ValueError: If ``epochs`` is not positive, if ``early_stopper``
            monitors a validation metric without a ``validation``
            strategy, or if the monitored metric is not produced by
            training or validation.
        TypeError: If ``state.rng_state`` is not a JAX random key.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not isinstance(state.rng_state, jax.Array):
        raise TypeError("state.rng_state must be a JAX random key.")
    if early_stopper and early_stopper.expects_validation() and validation is None:
        raise ValueError("validation metric monitoring requires a validation strategy.")

    _initialize_run_state(state, stage=stage, model=model, optimizer=optimizer)
    metric_suite = metrics
    train_step = make_train_step(loss, metrics=metric_suite.batch)
    latest_metrics: Mapping[str, float] = {}

    for epoch in range(epochs):
        epoch_metrics = _run_training_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_step=train_step,
            state=state,
            epoch=epoch,
            metrics=metric_suite,
        )
        latest_metrics = epoch_metrics
        _record_metrics(state, "training", epoch_metrics)
        _emit(event_sinks, "epoch_end", stage, epoch, epoch_metrics, state)

        validation_metrics = _run_validation_epoch(
            validation=validation,
            state=state,
            stage=stage,
            epoch=epoch,
            event_sinks=event_sinks,
        )
        if _should_stop_early(
            early_stopper=early_stopper,
            training_metrics=epoch_metrics,
            validation_metrics=validation_metrics,
            checkpoint_store=checkpoint_store,
            checkpoint_key=checkpoint_key,
            state=state,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            stage=stage,
            event_sinks=event_sinks,
        ):
            break

    return StageResult(state=state, metrics=latest_metrics, loss=latest_metrics["loss"])


def _initialize_run_state(
    state: TrainingState,
    *,
    stage: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> None:
    """Register mutable training components on the live state.

    Args:
        state: The training state mutated across epochs.
        stage: Stage name to record on ``state``.
        model: Model trained by the runner.
        optimizer: Optimizer updating ``model``.
    """
    state.register_component("model", model)
    state.register_optimizer("optimizer", optimizer)
    state.stage = stage


def _run_training_epoch(
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_loader: LoaderFactory,
    train_step: Callable[..., Mapping[str, jax.Array]],
    state: TrainingState,
    epoch: int,
    metrics: MetricSuite,
) -> dict[str, float]:
    """Run one training epoch and reduce all registered metrics.

    Args:
        model: The NNX model being trained.
        optimizer: The NNX optimizer updating ``model``.
        train_loader: Factory producing training batches.
        train_step: JIT-compiled training step.
        state: The live training state containing the RNG key.
        epoch: Epoch index passed to ``train_loader``.
        metrics: Registered batch and epoch metrics.

    Returns:
        Reduced epoch metrics containing ``"loss"`` and any registered metrics.
    """
    losses: list[float] = []
    batch_metric_values = initialize_batch_metric_values(metrics.batch)
    epoch_metric_parts = [] if metrics.epoch else None

    for batch in train_loader(split="train", epoch=epoch):
        state.rng_state, batch_key = split_key(state.rng_state)
        maybe_collect_epoch_metric_inputs(
            epoch_metric_parts,
            suite=metrics,
            model=model,
            batch=batch,
        )
        step_output = train_step(
            model,
            optimizer,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            batch_key,
        )
        collect_step_metrics(
            step_output,
            metrics=metrics.batch,
            losses=losses,
            batch_metric_values=batch_metric_values,
            context="train step",
        )

    return reduce_metric_suite(
        suite=metrics,
        losses=losses,
        batch_metric_values=batch_metric_values,
        epoch_metric_parts=epoch_metric_parts,
    )


def _run_validation_epoch(
    *,
    validation: ValidationStrategy | None,
    state: TrainingState,
    stage: str,
    epoch: int,
    event_sinks: Sequence[EventSink],
) -> Mapping[str, float]:
    """Run validation for one epoch when configured.

    Args:
        validation: Optional validation strategy.
        state: The live training state.
        stage: Stage name used in emitted events.
        epoch: Epoch index being validated.
        event_sinks: Event sinks notified on validation completion.

    Returns:
        Validation metrics, or an empty mapping when validation is disabled.
    """
    if validation is None:
        return {}

    validation_result = validation(state, epoch=epoch)
    validation_metrics = validation_result.metrics
    _record_metrics(state, "", validation_metrics)
    _emit(event_sinks, "validation_end", stage, epoch, validation_metrics, state)
    return validation_metrics


def _select_monitored_metrics(
    *,
    early_stopper: EarlyStopper | None,
    training_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> Mapping[str, float]:
    """Return the metric namespace observed by the early stopper.

    Args:
        early_stopper: Optional early-stopping policy.
        training_metrics: Metrics produced by the training epoch.
        validation_metrics: Metrics produced by validation.

    Returns:
        The metric mapping to observe for early stopping.
    """
    if early_stopper is None or not early_stopper.expects_validation():
        return training_metrics
    return validation_metrics


def _should_stop_early(
    *,
    early_stopper: EarlyStopper | None,
    training_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    checkpoint_store: CheckpointStore | None,
    checkpoint_key: str,
    state: TrainingState,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    epoch: int,
    stage: str,
    event_sinks: Sequence[EventSink],
) -> bool:
    """Observe metrics with the early stopper and emit side effects.

    Args:
        early_stopper: Optional early-stopping policy.
        training_metrics: Metrics produced by the training epoch.
        validation_metrics: Metrics produced by validation.
        checkpoint_store: Optional checkpoint store for best-model snapshots.
        checkpoint_key: Checkpoint key used for best-model snapshots.
        state: The live training state.
        model: The live model.
        optimizer: The live optimizer.
        epoch: Epoch index being observed.
        stage: Stage name used in emitted events.
        event_sinks: Event sinks notified on improvement or early stop.

    Returns:
        ``True`` when training should stop early, otherwise ``False``.

    Raises:
        ValueError: If the monitored metric is missing from the selected metric
            mapping.
    """
    if early_stopper is None:
        return False

    monitored_metrics = _select_monitored_metrics(
        early_stopper=early_stopper,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
    )
    metric_name = early_stopper.monitored_metric_name()
    if metric_name not in monitored_metrics:
        raise ValueError(f"monitored metric {metric_name!r} was not produced.")

    decision = early_stopper.observe(monitored_metrics[metric_name], epoch=epoch)
    if decision.improved:
        _save_checkpoint(
            checkpoint_store,
            checkpoint_key,
            state,
            model,
            optimizer,
            epoch,
            decision.state,
        )
        _emit(event_sinks, "best_model", stage, epoch, monitored_metrics, state)
    if decision.should_stop:
        _emit(event_sinks, "early_stop", stage, epoch, monitored_metrics, state)
    return decision.should_stop


def _record_metrics(
    state: TrainingState, prefix: str, metrics: Mapping[str, float]
) -> None:
    """Append metric values to ``state.metric_history`` in place.

    Args:
        state: The training state whose ``metric_history`` is updated.
        prefix: A prefix joined to each metric name with an underscore,
            or an empty string to leave names unprefixed.
        metrics: Mapping of metric name to the value observed this
            epoch.
    """
    for name, value in metrics.items():
        metric_name = f"{prefix}_{name}" if prefix else name
        state.record_metric(metric_name, value)


def _emit(
    sinks: Sequence[EventSink],
    name: str,
    stage: str,
    epoch: int,
    metrics: Mapping[str, float],
    state: TrainingState,
) -> None:
    """Build a training event and dispatch it to every sink.

    Args:
        sinks: The event sinks to notify.
        name: The event name, e.g. ``"epoch_end"`` or ``"early_stop"``.
        stage: The stage name associated with the event.
        epoch: The epoch at which the event occurred.
        metrics: The metrics associated with the event.
        state: The training state associated with the event.
    """
    event = TrainingEvent(
        name=name,
        stage=stage,
        iteration=state.outer_iteration,
        step=epoch,
        metrics=metrics,
        state=state,
    )
    for sink in sinks:
        sink.on_event(event)


def _save_checkpoint(
    store: CheckpointStore | None,
    key: str,
    state: TrainingState,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    epoch: int,
    early_stopping_state: object,
) -> None:
    """Persist a best-model checkpoint if a store was configured.

    Args:
        store: The checkpoint store to write to, or ``None`` to skip
            checkpointing entirely.
        key: The key under which the checkpoint is saved.
        state: The current training state, frozen into an independent
            snapshot before being embedded in the checkpoint.
        model: The model to snapshot for the checkpoint.
        optimizer: The optimizer to snapshot for the checkpoint.
        epoch: The epoch at which the improvement was observed.
        early_stopping_state: The early-stopping state at the time of
            the improvement.
    """
    if store is None:
        return
    store.save(
        key,
        Checkpoint(
            state=freeze_training_state(state),
            epoch=epoch,
            parameters=snapshot(model),
            optimizer_state=snapshot(optimizer),
            rng_state=state.rng_state,
            early_stopping_state=early_stopping_state,
        ),
    )
