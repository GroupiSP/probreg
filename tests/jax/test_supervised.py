from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probreg.core.checkpoints import Checkpoint
from probreg.core.early_stopping import EarlyStopper, MetricSource
from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.core.metric_registry import (
    EpochPredictionData,
    EvaluationGrid,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
)
from probreg.core.tracking import TrainingEvent
from probreg.core.types import Batch, StageResult, TrainingState, ValidationResult
from probreg.jax import (
    BatchMetricSpec,
    GaussianHead,
    GaussianPredictor,
    HeldOutValidation,
    MetricSuite,
    PredictionRequirements,
    create_optimizer,
    evaluate_loader,
    initialize_training_state,
    make_evaluation_step,
    make_supervised_loss,
    make_train_step,
    run_supervised,
    split_key,
)


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


def linear_predictor(
    model: nnx.Module,
    batch: Batch,
    requirements: PredictionRequirements,
    key: jax.Array,
    /,
) -> EpochPredictionData:
    del requirements, key
    if batch.targets is None:
        raise ValueError("batch.targets must be provided for epoch metrics.")
    return EpochPredictionData(
        targets=np.asarray(jax.device_get(batch.targets), dtype=float).reshape(-1),
        mean=np.asarray(jax.device_get(model(batch.inputs)), dtype=float).reshape(-1),
    )


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.values: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.values[key] = checkpoint

    def load(self, key: str) -> Checkpoint:
        return self.values[key]

    def exists(self, key: str) -> bool:
        return key in self.values


class EventCollector:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)


def squared_error(
    model: LinearModel,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del key, training
    errors = jnp.square(model(inputs) - targets)
    if sample_weight is not None:
        errors = errors * sample_weight
    return jnp.mean(errors)


def mean_absolute_error(
    model: LinearModel,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del key, training
    errors = jnp.abs(model(inputs) - targets)
    if sample_weight is not None:
        errors = errors * sample_weight
    return jnp.mean(errors)


def training_flag_metric(
    model: LinearModel,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del model, inputs, targets, sample_weight, key
    return jnp.asarray(1.0 if training else 0.0)


def loader(*, split: str, epoch: int) -> Iterable[Batch]:
    del epoch
    target = 2.0 if split == "train" else 1.0
    return [Batch(inputs=jnp.array([[1.0]]), targets=jnp.array([[target]]))]


def make_components(
    learning_rate: float = 0.1,
) -> tuple[LinearModel, nnx.Optimizer, TrainingState]:
    model = LinearModel(rngs=nnx.Rngs(0))
    optimizer = create_optimizer(model, optax.sgd(learning_rate))
    state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
    return model, optimizer, state


def test_fixed_epoch_training_without_validation_updates_parameters_and_rng() -> None:
    model, optimizer, state = make_components()
    initial_key = state.rng_state

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=3,
    )

    assert len(result.state.metric_history["training_loss"]) == 3
    assert result.loss is not None
    assert not bool(jnp.array_equal(initial_key, result.state.rng_state))
    assert int(optimizer.step.get_value()) == 3


def test_run_supervised_registers_named_model_and_optimizer() -> None:
    model, optimizer, state = make_components()
    state.model_components.clear()
    state.optimizer_states.clear()

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=1,
        stage="mean",
        model_name="mean_model",
        optimizer_name="mean_optimizer",
    )

    assert result.state.model_components == {"mean_model": model}
    assert result.state.optimizer_states == {"mean_optimizer": optimizer}
    assert result.state.active_stage == "mean"


def test_run_supervised_default_registration_names_remain_compatible() -> None:
    model, optimizer, state = make_components()

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=1,
    )

    assert result.state.model_components == {"model": model}
    assert result.state.optimizer_states == {"optimizer": optimizer}


def test_make_train_step_without_registered_metrics_returns_loss_mapping() -> None:
    model, optimizer, _ = make_components()
    train_step = make_train_step(squared_error)

    output = train_step(
        model,
        optimizer,
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
        None,
        jax.random.key(0),
    )

    assert set(output) == {"loss"}


def test_make_train_step_evaluates_batch_metrics_in_inference_mode() -> None:
    model, optimizer, _ = make_components()
    train_step = make_train_step(
        squared_error,
        metrics=(BatchMetricSpec(name="training_flag", metric=training_flag_metric),),
    )

    output = train_step(
        model,
        optimizer,
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
        None,
        jax.random.key(0),
    )

    assert output["training_flag"] == pytest.approx(0.0)


def test_make_evaluation_step_without_registered_metrics_returns_loss_mapping() -> None:
    model, _, state = make_components()
    evaluation_step = make_evaluation_step(squared_error)

    metrics, next_key = evaluate_loader(
        model,
        loader(split="train", epoch=0),
        key=state.rng_state,
        evaluation_step=evaluation_step,
    )

    assert set(metrics) == {"loss"}
    assert not bool(jnp.array_equal(next_key, state.rng_state))


def test_training_metric_stopping_saves_best_checkpoint_and_events() -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    store = MemoryCheckpointStore()
    events = EventCollector()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=MetricSource.TRAINING,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=4,
        early_stopper=stopper,
        event_sinks=[events],
        checkpoint_store=store,
        checkpoint_key="mean-best",
    )

    assert len(result.state.metric_history["training_loss"]) == 2
    assert store.exists("mean-best")
    checkpoint = store.load("mean-best")
    assert checkpoint.epoch == 0
    assert checkpoint.early_stopping_state.best_epoch == 0
    assert checkpoint.early_stopping_state.stopped is False
    assert [event.name for event in events.events] == [
        "epoch_end",
        "best_model",
        "epoch_end",
        "early_stop",
    ]


def test_best_checkpoint_state_is_frozen_and_unaffected_by_later_epochs() -> None:
    # Learning rate equal to zero is ensured that the checkpoint is saved after
    # the first epoch, which allows to test if the checkpoint remains unaffected by later epochs.
    model, optimizer, state = make_components(learning_rate=0.0)
    store = MemoryCheckpointStore()
    # High patience so training keeps running (and keeps mutating ``state``)
    # for several epochs after the one-and-only improvement is checkpointed.
    stopper = EarlyStopper(
        metric="loss", mode="min", patience=5, source=MetricSource.TRAINING
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=3,
        early_stopper=stopper,
        checkpoint_store=store,
    )

    checkpoint = store.load("best")

    # The checkpoint was saved after epoch 0; its frozen state must only
    # contain that single observation, even though the live state keeps
    # accumulating history for all 3 epochs.
    assert checkpoint.state.metric_history["training_loss"] == [
        result.state.metric_history["training_loss"][0]
    ]
    assert len(result.state.metric_history["training_loss"]) == 3

    # Mutating the live training state after the fact (as further training,
    # or a resumed run, would) must not leak into the already-saved
    # checkpoint's state.
    result.state.metric_history["training_loss"].append(999.0)
    result.state.checkpoint_registry["late"] = object()

    assert checkpoint.state.metric_history["training_loss"] == [
        result.state.metric_history["training_loss"][0]
    ]
    assert "late" not in checkpoint.state.checkpoint_registry

    # The checkpoint's snapshot must also be independent of the live model
    # and optimizer, which the training loop keeps mutating in place.
    assert checkpoint.state.model_components == {}
    assert checkpoint.state.optimizer_states == {}


def test_held_out_validation_drives_validation_metric_stopping() -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    validation = HeldOutValidation(model=model, loader=loader, loss=squared_error)
    stopper = EarlyStopper(metric="validation_loss", mode="min", patience=0)

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=4,
        validation=validation,
        early_stopper=stopper,
    )

    assert len(result.state.metric_history["validation_loss"]) == 2


def test_custom_fold_validation_strategy_is_accepted() -> None:
    model, optimizer, state = make_components(learning_rate=0.0)

    def fold_validation(
        current_state: TrainingState, *, epoch: int
    ) -> ValidationResult:
        del current_state
        return ValidationResult(
            passed=True,
            metrics={"fold_1_loss": float(epoch)},
            message="fold-1",
        )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=2,
        validation=fold_validation,
    )

    assert result.state.metric_history["fold_1_loss"] == [0.0, 1.0]


def test_validation_stopping_requires_a_validation_strategy() -> None:
    model, optimizer, state = make_components()
    stopper = EarlyStopper(metric="validation_loss", mode="min", patience=1)

    with pytest.raises(ValueError, match="requires a validation strategy"):
        run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            loss=squared_error,
            state=state,
            epochs=1,
            early_stopper=stopper,
        )


def test_split_key_is_reproducible() -> None:
    key = jax.random.key(42)

    next_key, operation_key = split_key(key)
    duplicate_next_key, duplicate_operation_key = split_key(key)

    assert bool(jnp.array_equal(next_key, duplicate_next_key))
    assert bool(jnp.array_equal(operation_key, duplicate_operation_key))


def test_run_supervised_records_registered_batch_and_epoch_metrics() -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    events = EventCollector()
    metric_suite = MetricSuite(
        batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
        epoch=(RootMeanSquaredError(),),
        predictor=linear_predictor,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=2,
        metrics=metric_suite,
        event_sinks=[events],
    )

    assert set(result.metrics) == {"loss", "mae", "rmse"}
    assert len(result.state.metric_history["training_loss"]) == 2
    assert len(result.state.metric_history["training_mae"]) == 2
    assert len(result.state.metric_history["training_rmse"]) == 2
    assert all(
        {"loss", "mae", "rmse"} <= set(event.metrics)
        for event in events.events
        if event.name == "epoch_end"
    )


def test_held_out_validation_prefixes_registered_metrics() -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    validation = HeldOutValidation(
        model=model,
        loader=loader,
        loss=squared_error,
        metrics=MetricSuite(
            batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
            epoch=(RootMeanSquaredError(),),
            predictor=linear_predictor,
        ),
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss=squared_error,
        state=state,
        epochs=2,
        validation=validation,
    )

    assert len(result.state.metric_history["validation_loss"]) == 2
    assert len(result.state.metric_history["validation_mae"]) == 2
    assert len(result.state.metric_history["validation_rmse"]) == 2


def test_metric_suite_rejects_reserved_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="reserved"):
        MetricSuite(batch=(BatchMetricSpec(name="loss", metric=mean_absolute_error),))

    with pytest.raises(ValueError, match="duplicate"):
        MetricSuite(
            batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
            epoch=(RootMeanSquaredError(metric_name="mae"),),
        )


def test_epoch_metrics_require_explicit_predictor() -> None:
    with pytest.raises(ValueError, match="explicit suite predictor"):
        MetricSuite(epoch=(RootMeanSquaredError(),))


def test_sampled_epoch_metrics_preserve_loss_batch_metric_and_rng_trajectory() -> None:
    def random_key_metric(
        model: nnx.Module,
        inputs: jax.Array,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del model, inputs, targets, sample_weight, training
        return jax.random.uniform(key)

    batch_metric = BatchMetricSpec(name="key_probe", metric=random_key_metric)
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    base_suite = MetricSuite(batch=(batch_metric,))
    sampled_suite = MetricSuite(
        batch=(batch_metric,),
        epoch=(PointContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
        predictive_sample_count=8,
        evaluation_grid=grid,
    )

    def run(metrics: MetricSuite) -> tuple[StageResult, GaussianHead]:
        model = GaussianHead(1, 1, rngs=nnx.Rngs(7))
        optimizer = create_optimizer(model, optax.sgd(0.01))
        state = initialize_training_state(model, optimizer, rng_key=jax.random.key(11))
        result = run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            loss=make_supervised_loss(NegativeLogLikelihoodLoss()),
            state=state,
            epochs=2,
            metrics=metrics,
        )
        return result, model

    base_result, base_model = run(base_suite)
    sampled_result, sampled_model = run(sampled_suite)

    assert base_result.metrics["loss"] == pytest.approx(sampled_result.metrics["loss"])
    assert base_result.metrics["key_probe"] == pytest.approx(
        sampled_result.metrics["key_probe"]
    )
    assert bool(
        jnp.array_equal(base_result.state.rng_state, sampled_result.state.rng_state)
    )
    for base_leaf, sampled_leaf in zip(
        jax.tree.leaves(nnx.state(base_model)),
        jax.tree.leaves(nnx.state(sampled_model)),
        strict=True,
    ):
        assert jnp.array_equal(base_leaf, sampled_leaf)
