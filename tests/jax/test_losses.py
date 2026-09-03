from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.losses import SquaredErrorLoss
from probreg.jax.losses import make_supervised_loss
from probreg.jax.state import create_optimizer
from probreg.jax.supervised import make_train_step
from probreg.jax.evaluation import make_evaluation_step


class ModeAwareModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        self.bias = nnx.Param(jnp.asarray(0.0))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        offset = 0.0 if self.dropout.deterministic else 1.0
        return inputs + self.bias[...] + offset


def test_supervised_loss_propagates_training_mode() -> None:
    model = ModeAwareModel(rngs=nnx.Rngs(0))
    model.eval()
    loss = make_supervised_loss(SquaredErrorLoss())
    train_step = make_train_step(loss)
    optimizer = create_optimizer(model, optax.sgd(0.0))
    inputs = jnp.ones((2, 1))

    output = train_step(
        model,
        optimizer,
        inputs,
        inputs,
        None,
        jax.random.key(0),
    )

    assert output["loss"] == 1.0
    assert model.dropout.deterministic is False


def test_supervised_loss_evaluates_inference_clone_without_mutating_model() -> None:
    model = ModeAwareModel(rngs=nnx.Rngs(0))
    model.train()
    loss = make_supervised_loss(SquaredErrorLoss())
    evaluation_step = make_evaluation_step(loss)
    inputs = jnp.ones((2, 1))

    first = evaluation_step(
        model,
        inputs,
        inputs,
        None,
        jax.random.key(0),
    )["loss"]
    second = evaluation_step(
        model,
        inputs,
        inputs,
        None,
        jax.random.key(1),
    )["loss"]

    assert first == 0.0
    assert second == first
    assert model.dropout.deterministic is False
