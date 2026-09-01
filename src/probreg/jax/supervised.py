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
    _finite_float,
    _metric_mean,
    merge_metric_inputs,
    resolve_metric_inputs,
)
from probreg.jax.rng import split_key
from probreg.jax.state import freeze_training_state, snapshot


def make_train_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., jax.Array | Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX/Optax supervised training step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.

    Returns:
        A JIT-compiled function ``train_step(model, optimizer, inputs,
        targets, sample_weight, key)`` that performs one gradient update
        in place. When ``metrics`` is empty it returns the scalar loss.
        Otherwise it returns a mapping containing ``"loss"`` plus
        registered batch metric values. Loss and metric values are
        evaluated on the pre-update model state for consistency.
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
    ) -> jax.Array | Mapping[str, jax.Array]:
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
        if not metrics:
            optimizer.update(model, gradients)
            return loss_value
        values: dict[str, jax.Array] = {"loss": loss_value}
        for spec in metrics:
            values[spec.name] = spec.metric(
                model,
                inputs,
                targets,
                sample_weight,
                key,
                True,
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

    state.register_component("model", model)
    state.register_optimizer("optimizer", optimizer)
    state.stage = stage
    metric_suite = metrics
    train_step = make_train_step(loss, metrics=metric_suite.batch)
    latest_metrics: Mapping[str, float] = {}

    for epoch in range(epochs):
        train_losses: list[float] = []
        train_batch_metric_values: dict[str, list[float]] = {
            spec.name: [] for spec in metric_suite.batch
        }
        epoch_metric_parts = [] if metric_suite.epoch else None
        for batch in train_loader(split="train", epoch=epoch):
            state.rng_state, batch_key = split_key(state.rng_state)
            if epoch_metric_parts is not None:
                epoch_metric_parts.append(
                    resolve_metric_inputs(suite=metric_suite, model=model, batch=batch)
                )
            step_output = train_step(
                model,
                optimizer,
                batch.inputs,
                batch.targets,
                batch.sample_weight,
                batch_key,
            )
            if isinstance(step_output, Mapping):
                train_losses.append(float(step_output["loss"]))
                for spec in metric_suite.batch:
                    if spec.name not in step_output:
                        raise ValueError(
                            f"train step did not produce registered metric {spec.name!r}."
                        )
                    train_batch_metric_values[spec.name].append(
                        float(step_output[spec.name])
                    )
            else:
                if metric_suite.batch:
                    raise ValueError(
                        "train_step must return metric mappings when batch metrics are registered."
                    )
                train_losses.append(float(step_output))

        epoch_metrics: dict[str, float] = {
            "loss": _finite_float("loss", _metric_mean(train_losses))
        }
        for spec in metric_suite.batch:
            epoch_metrics[spec.name] = _finite_float(
                spec.name, spec.reduce(train_batch_metric_values[spec.name])
            )
        if epoch_metric_parts is not None:
            merged = merge_metric_inputs(epoch_metric_parts)
            for epoch_metric in metric_suite.epoch:
                epoch_metrics[epoch_metric.name] = _finite_float(
                    epoch_metric.name,
                    float(epoch_metric(merged)),
                )

        latest_metrics = epoch_metrics
        _record_metrics(state, "training", epoch_metrics)
        _emit(event_sinks, "epoch_end", stage, epoch, epoch_metrics, state)

        validation_metrics: Mapping[str, float] = {}
        if validation is not None:
            validation_result = validation(state, epoch=epoch)
            validation_metrics = validation_result.metrics
            _record_metrics(state, "", validation_metrics)
            _emit(
                event_sinks, "validation_end", stage, epoch, validation_metrics, state
            )

        monitored_metrics = (
            epoch_metrics
            if early_stopper is None or not early_stopper.expects_validation()
            else validation_metrics
        )
        if early_stopper is not None:
            metric_name = early_stopper.monitored_metric_name()
            if metric_name not in monitored_metrics:
                raise ValueError(f"monitored metric {metric_name!r} was not produced.")
            decision = early_stopper.observe(
                monitored_metrics[metric_name], epoch=epoch
            )
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
                break

    return StageResult(state=state, metrics=latest_metrics, loss=latest_metrics["loss"])


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
