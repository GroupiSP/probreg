from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probreg.core.checkpoints import Checkpoint
from probreg.core.early_stopping import EarlyStopper, MetricSource
from probreg.core.losses import GammaResidualNLLLoss
from probreg.core.types import Batch, ParameterRole, StageState, TrainingState
from probreg.jax.state import create_optimizer
from probreg.jax.two_step import (
    MeanStage,
    SupervisedStageOptions,
    make_gamma_residual_loss,
    make_mean_squared_error_loss,
    materialize_residual_loader,
)


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


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


def test_mean_squared_error_loss_supports_weights_and_reduction() -> None:
    model = LinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_mean_squared_error_loss(reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([[1.0], [5.0]]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_supports_weights_and_reduction() -> None:
    loss = make_gamma_residual_loss(
        GammaResidualNLLLoss(epsilon=1e-6),
        reduction=jnp.sum,
    )

    value = loss(
        LogPredictionModel(),
        jnp.ones((2, 1)),
        jnp.array([[0.0], [1.0]]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    expected = -2.0 * jnp.log(1e-6) - 3.0 * jnp.log(1.000001)
    assert value == pytest.approx(float(expected))


def test_gamma_residual_loss_has_finite_gradient_at_zero() -> None:
    loss = make_gamma_residual_loss(GammaResidualNLLLoss(epsilon=1e-6))

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


def test_mean_stage_rejects_missing_checkpoint() -> None:
    stage, state = make_mean_stage()

    with pytest.raises(ValueError, match="not available"):
        stage.select_checkpoint(state)
