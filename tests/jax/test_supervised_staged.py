from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probreg.core.checkpoints import Checkpoint
from probreg.core.early_stopping import EarlyStopper, MetricSource
from probreg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
from probreg.core.protocols import LoaderFactory, ValidationStrategy
from probreg.core.tracking import TrainingEvent
from probreg.core.types import (
    Batch,
    ParameterRole,
    StageState,
    TrainingState,
    ValidationResult,
)
from probreg.jax.distributions import GammaHead
from probreg.jax.losses import make_supervised_loss
from probreg.jax.state import create_optimizer, restore_checkpoint
from probreg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
    materialize_residual_loader,
)


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


class SqueezedLinearModel(LinearModel):
    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs).squeeze(-1)


class LogPrediction:
    batch_shape = (2, 1)
    event_shape = ()

    def __init__(self, values: jax.Array) -> None:
        self.values = values

    def log_prob(self, targets: jax.Array) -> jax.Array:
        return jnp.log(targets) + self.values

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        del key
        return jnp.zeros(sample_shape + self.batch_shape)

    def mean(self) -> jax.Array:
        return self.values

    def variance(self) -> jax.Array:
        return jnp.ones_like(self.values)


class LogPredictionModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> LogPrediction:
        return LogPrediction(jnp.zeros_like(inputs))


class DropoutMeanModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.dropout(inputs)


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


def test_mean_squared_error_loss_supports_weights_and_reduction() -> None:
    model = LinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_supervised_loss(SquaredErrorLoss(), reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([[1.0], [5.0]]),
        jnp.array([2.0, 3.0]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_supports_weights_and_reduction() -> None:
    loss = make_supervised_loss(
        NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6)),
        reduction=jnp.sum,
    )

    value = loss(
        LogPredictionModel(),
        jnp.ones((2, 1)),
        jnp.array([[0.0], [1.0]]),
        jnp.array([2.0, 3.0]),
        jax.random.key(0),
        True,
    )

    expected = -2.0 * jnp.log(1e-6) - 3.0 * jnp.log(1.000001)
    assert value == pytest.approx(float(expected))


def test_supervised_loss_rejects_mismatched_sample_weight_batch() -> None:
    loss = make_supervised_loss(SquaredErrorLoss())

    with pytest.raises(ValueError, match="matching batch sizes"):
        loss(
            LinearModel(rngs=nnx.Rngs(0)),
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
            jnp.ones((3,)),
            jax.random.key(0),
            True,
        )


def test_supervised_loss_squeezes_column_weights_for_vector_losses() -> None:
    model = SqueezedLinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_supervised_loss(SquaredErrorLoss(), reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([1.0, 5.0]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_has_finite_gradient_at_zero() -> None:
    loss = make_supervised_loss(
        NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6))
    )

    def evaluate(target: jax.Array) -> jax.Array:
        return loss(
            LogPredictionModel(),
            jnp.ones((1, 1)),
            target,
            None,
            jax.random.key(0),
            True,
        )

    gradient = jax.grad(evaluate)(jnp.zeros((1, 1)))

    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_materialize_residual_loader_caches_exact_detached_residuals() -> None:
    model = LinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    calls: list[tuple[str, int]] = []
    sample_weight = jnp.array([[0.5], [1.0]])
    metadata = {"source": "xsin"}

    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        calls.append((split, epoch))
        return [
            Batch(
                inputs=jnp.array([[1.0], [2.0]]),
                targets=jnp.array([[3.0], [2.0]]),
                sample_weight=sample_weight,
                metadata=metadata,
            )
        ]

    residual_loader = materialize_residual_loader(
        model,
        source_loader,
        splits=("train",),
        source_epoch=7,
    )
    first = residual_loader(split="train", epoch=0)
    second = residual_loader(split="train", epoch=99)

    assert calls == [("train", 7)]
    assert first is second
    assert jnp.array_equal(first[0].targets, jnp.array([[1.0], [4.0]]))
    assert first[0].sample_weight is sample_weight
    assert first[0].metadata is metadata


def test_materialized_residual_targets_stop_source_target_gradients() -> None:
    model = LinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 0.0
    model.linear.bias[...] = 0.0

    def residual_sum(targets: jax.Array) -> jax.Array:
        def source_loader(*, split: str, epoch: int) -> list[Batch]:
            del split, epoch
            return [Batch(inputs=jnp.ones_like(targets), targets=targets)]

        loader = materialize_residual_loader(
            model,
            source_loader,
            splits=("train",),
        )
        return jnp.sum(loader(split="train", epoch=0)[0].targets)

    gradient = jax.grad(residual_sum)(jnp.array([[2.0]]))

    assert jnp.array_equal(gradient, jnp.zeros((1, 1)))


def test_materialize_residual_loader_uses_inference_clone() -> None:
    model = DropoutMeanModel(rngs=nnx.Rngs(0))
    model.train()

    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        values = jnp.ones((2, 1))
        return [Batch(inputs=values, targets=values)]

    loader = materialize_residual_loader(
        model,
        source_loader,
        splits=("train",),
    )

    assert model.dropout.deterministic is False
    assert jnp.array_equal(
        loader(split="train", epoch=0)[0].targets,
        jnp.zeros((2, 1)),
    )


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ((), "at least one"),
        (("",), "must not be empty"),
        (("train", "train"), "unique"),
    ],
)
def test_materialize_residual_loader_rejects_invalid_splits(
    splits: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_residual_loader(
            LinearModel(rngs=nnx.Rngs(0)),
            lambda **kwargs: [],
            splits=splits,
        )


def test_materialize_residual_loader_rejects_empty_source_split() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        materialize_residual_loader(
            LinearModel(rngs=nnx.Rngs(0)),
            lambda **kwargs: [],
            splits=("train",),
        )


def test_materialize_residual_loader_requires_targets() -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=jnp.ones((1, 1)))]

    with pytest.raises(ValueError, match="provide targets"):
        materialize_residual_loader(
            LinearModel(rngs=nnx.Rngs(0)),
            source_loader,
            splits=("train",),
        )


def test_materialize_residual_loader_requires_matching_shapes() -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [
            Batch(
                inputs=jnp.ones((2, 1)),
                targets=jnp.ones((2,)),
            )
        ]

    with pytest.raises(ValueError, match="matching shapes"):
        materialize_residual_loader(
            LinearModel(rngs=nnx.Rngs(0)),
            source_loader,
            splits=("train",),
        )


def test_materialized_residual_loader_rejects_unknown_split() -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=jnp.ones((1, 1)), targets=jnp.ones((1, 1)))]

    loader = materialize_residual_loader(
        LinearModel(rngs=nnx.Rngs(0)),
        source_loader,
        splits=("train",),
    )

    with pytest.raises(ValueError, match="was not materialized"):
        loader(split="validation", epoch=0)


def mean_loader(*, split: str, epoch: int) -> list[Batch]:
    del split, epoch
    return [
        Batch(
            inputs=jnp.array([[-1.0], [0.0], [1.0]]),
            targets=jnp.array([[-2.0], [0.0], [2.0]]),
        )
    ]


def make_mean_stage(
    *,
    learning_rate: float = 0.1,
    checkpoint_store: MemoryCheckpointStore | None = None,
    early_stopper: EarlyStopper | None = None,
    validation: ValidationStrategy | None = None,
) -> tuple[MeanStage, TrainingState]:
    model = LinearModel(rngs=nnx.Rngs(0))
    optimizer = create_optimizer(model, optax.sgd(learning_rate))
    stage = MeanStage(
        model=model,
        optimizer=optimizer,
        train_loader=mean_loader,
        options=SupervisedStageOptions(
            epochs=10,
            checkpoint_store=checkpoint_store,
            checkpoint_key="mean-best",
            early_stopper=early_stopper,
            validation=validation,
        ),
    )
    return stage, TrainingState(rng_state=jax.random.key(1))


def test_mean_stage_prepares_trains_and_validates_lifecycle() -> None:
    stage, state = make_mean_stage()

    stage.prepare(state)
    initial_loss = float(
        stage.loss(
            stage.model,
            mean_loader(split="train", epoch=0)[0].inputs,
            mean_loader(split="train", epoch=0)[0].targets,
            None,
            jax.random.key(2),
            False,
        )
    )
    result = stage.train(state)

    assert state.lifecycle_state is StageState.MEAN_READY
    assert state.active_stage == "mean"
    assert state.model_components["mean_model"] is stage.model
    assert state.optimizer_states["mean_optimizer"] is stage.optimizer
    assert state.parameter_roles["mean_model"] is ParameterRole.MEAN
    assert result.loss is not None and result.loss < initial_loss
    assert len(state.metric_history["mean_training_loss"]) == 10
    assert "training_loss" not in state.metric_history
    assert stage.validate(state).passed


def test_mean_stage_rejects_invalid_lifecycle_without_advancing_state() -> None:
    stage, state = make_mean_stage()
    state.lifecycle_state = StageState.VARIANCE_READY

    with pytest.raises(ValueError, match="requires NEW or INITIALIZED"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.VARIANCE_READY


def test_mean_stage_rejects_conflicting_parameter_role() -> None:
    stage, state = make_mean_stage()
    state.parameter_roles["mean_model"] = ParameterRole.VARIANCE

    with pytest.raises(ValueError, match="already has role"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.NEW


def test_mean_stage_selects_existing_checkpoint() -> None:
    store = MemoryCheckpointStore()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=MetricSource.TRAINING,
    )
    stage, state = make_mean_stage(
        learning_rate=0.0,
        checkpoint_store=store,
        early_stopper=stopper,
    )

    stage.prepare(state)
    stage.train(state)
    reference = stage.select_checkpoint(state)

    assert reference.key == "mean-best"
    assert reference.metadata == {"stage": "mean"}
    assert "mean_training_loss" in store.load("mean-best").state.metric_history


def test_mean_stage_restores_and_finalizes_best_checkpoint() -> None:
    store = MemoryCheckpointStore()
    stopper = EarlyStopper(metric="validation_loss", mode="min", patience=0)

    def validation(
        current_state: TrainingState,
        *,
        epoch: int,
    ) -> ValidationResult:
        del current_state
        return ValidationResult(
            passed=True,
            metrics={"validation_loss": float(epoch)},
        )

    stage, state = make_mean_stage(
        checkpoint_store=store,
        early_stopper=stopper,
        validation=validation,
    )
    expected_stage, expected_state = make_mean_stage()
    expected_stage.options = SupervisedStageOptions(epochs=1)

    stage.prepare(state)
    result = stage.train(state)
    expected_stage.prepare(expected_state)
    expected_stage.train(expected_state)

    checkpoint = store.load("mean-best")
    assert checkpoint.epoch == 0
    assert checkpoint.state.lifecycle_state is StageState.MEAN_READY
    assert checkpoint.metadata == {"stage": "mean", "stage_complete": True}
    assert int(stage.optimizer.step.get_value()) == 1
    assert len(state.metric_history["mean_training_loss"]) == 1
    assert result.loss == state.metric_history["mean_training_loss"][-1]
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree.leaves(nnx.state(stage.model)),
            jax.tree.leaves(nnx.state(expected_stage.model)),
            strict=True,
        )
    )


def test_finalized_mean_checkpoint_can_resume_variance_preparation() -> None:
    store = MemoryCheckpointStore()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=MetricSource.TRAINING,
    )
    trained_stage, trained_state = make_mean_stage(
        learning_rate=0.0,
        checkpoint_store=store,
        early_stopper=stopper,
    )
    trained_stage.prepare(trained_state)
    trained_stage.train(trained_state)

    resumed_model = LinearModel(rngs=nnx.Rngs(8))
    resumed_optimizer = create_optimizer(resumed_model, optax.sgd(0.0))
    resumed_state = TrainingState(rng_state=jax.random.key(9))
    restore_checkpoint(
        store.load("mean-best"),
        state=resumed_state,
        model=resumed_model,
        optimizer=resumed_optimizer,
        model_name="mean_model",
        optimizer_name="mean_optimizer",
    )
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(10))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )

    variance_stage.prepare(resumed_state)

    assert resumed_state.lifecycle_state is StageState.MEAN_READY
    assert resumed_state.model_components["mean_model"] is resumed_model
    assert "mean_model" in resumed_state.frozen_components


def test_mean_stage_rejects_missing_checkpoint() -> None:
    stage, state = make_mean_stage()

    with pytest.raises(ValueError, match="not available"):
        stage.select_checkpoint(state)


def make_two_step_loader(
    inputs: jax.Array,
    targets: jax.Array,
) -> LoaderFactory:
    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=inputs, targets=targets)]

    return loader


def test_gamma_variance_stage_updates_variance_and_preserves_mean() -> None:
    data_key, mean_key, variance_key, rng_key = jax.random.split(
        jax.random.key(20),
        4,
    )
    inputs = jax.random.uniform(data_key, (256, 1), minval=-2.0, maxval=2.0)
    noise_scale = 0.1 + 0.3 * (inputs + 2.0)
    targets = 2.0 * inputs + noise_scale * jax.random.normal(
        jax.random.fold_in(data_key, 1),
        inputs.shape,
    )
    source_loader = make_two_step_loader(inputs, targets)
    events = EventCollector()

    mean_model = LinearModel(rngs=nnx.Rngs(mean_key))
    mean_optimizer = create_optimizer(mean_model, optax.adam(0.05))
    state = TrainingState(rng_state=rng_key)
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=mean_optimizer,
        train_loader=source_loader,
        options=SupervisedStageOptions(epochs=100, event_sinks=(events,)),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)
    mean_state_before = jax.tree.map(lambda value: value.copy(), nnx.state(mean_model))

    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(variance_key))
    variance_optimizer = create_optimizer(variance_model, optax.adam(0.03))
    variance_state_before = jax.tree.map(
        lambda value: value.copy(),
        nnx.state(variance_model),
    )
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=variance_optimizer,
        source_loader=source_loader,
        options=SupervisedStageOptions(epochs=150, event_sinks=(events,)),
        splits=("train",),
    )

    variance_stage.prepare(state)
    result = variance_stage.train(state)

    assert result.loss is not None and math.isfinite(result.loss)
    assert state.lifecycle_state is StageState.VARIANCE_READY
    assert state.parameter_roles["variance_model"] is ParameterRole.VARIANCE
    assert "mean_model" in state.frozen_components
    assert len(state.metric_history["mean_training_loss"]) == 100
    assert len(state.metric_history["variance_training_loss"]) == 150
    assert "training_loss" not in state.metric_history
    assert {event.stage for event in events.events} == {"mean", "variance"}
    assert variance_stage.validate(state).passed
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(mean_state_before),
            jax.tree.leaves(nnx.state(mean_model)),
            strict=True,
        )
    )
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(variance_state_before),
            jax.tree.leaves(nnx.state(variance_model)),
            strict=True,
        )
    )
    low_variance = variance_model(jnp.array([[-2.0]])).mean()
    high_variance = variance_model(jnp.array([[2.0]])).mean()
    assert bool(jnp.all(high_variance > low_variance))


def test_gamma_variance_stage_requires_ready_mean() -> None:
    model = GammaHead(1, 1, rngs=nnx.Rngs(0))
    stage = GammaVarianceStage(
        model=model,
        optimizer=create_optimizer(model, optax.sgd(0.1)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )
    state = TrainingState(rng_state=jax.random.key(0))

    with pytest.raises(ValueError, match="MEAN_READY"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.NEW


def test_gamma_variance_stage_requires_mean_role() -> None:
    mean_model = LinearModel(rngs=nnx.Rngs(0))
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
    state = TrainingState(
        model_components={"mean_model": mean_model},
        parameter_roles={"mean_model": ParameterRole.AUXILIARY},
        lifecycle_state=StageState.MEAN_READY,
        rng_state=jax.random.key(0),
    )
    stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.1)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )

    with pytest.raises(ValueError, match="MEAN parameter role"):
        stage.prepare(state)

    assert "variance_model" not in state.model_components


def test_gamma_variance_stage_builds_validation_from_residual_loader() -> None:
    mean_stage, state = make_mean_stage()
    mean_stage.prepare(state)
    mean_stage.train(state)
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
    observed_targets: list[jax.Array] = []

    def validation_factory(loader: LoaderFactory):
        def validation(
            current_state: TrainingState,
            *,
            epoch: int,
        ) -> ValidationResult:
            del current_state
            observed_targets.append(loader(split="validation", epoch=epoch)[0].targets)
            return ValidationResult(passed=True, metrics={"validation_loss": 0.0})

        return validation

    stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        validation_factory=validation_factory,
    )

    stage.prepare(state)
    stage.train(state)

    assert len(observed_targets) == 1
    assert bool(jnp.all(observed_targets[0] >= 0.0))
