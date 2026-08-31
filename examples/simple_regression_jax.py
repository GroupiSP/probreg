"""Train a linear model on a toy regression dataset with the JAX backend.

This example wires together the pieces exposed by :mod:`probreg.jax` and
:mod:`probreg.core` to run :func:`probreg.jax.run_supervised` end to end:

* a synthetic ``y = 3x + 2 + noise`` regression dataset, split into a
  training and a held-out validation set;
* a single-layer NNX model trained with an Optax SGD optimizer;
* a :class:`~probreg.jax.HeldOutValidation` strategy evaluated after
  every epoch;
* an :class:`~probreg.core.early_stopping.EarlyStopper` monitoring the
  validation loss, backed by an in-memory checkpoint store that
  persists the best model seen so far;
* a minimal :class:`~probreg.core.tracking.EventSink` that prints epoch
  and early-stopping events to stdout.

Run it with:

    uv run --extra jax python examples/simple_regression_jax.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper, MetricSource, OptimizationMode
from probreg.core.tracking import TrainingEvent
from probreg.core.types import Batch
from probreg.jax import (
    HeldOutValidation,
    create_optimizer,
    initialize_training_state,
    run_supervised,
)


class LinearModel(nnx.Module):
    """A single trainable linear layer mapping one input to one output."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs)


def mean_squared_error(
    model: LinearModel,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    """Compute the (optionally weighted) mean squared error for a batch.

    Args:
        model: The linear model to evaluate.
        inputs: Batch inputs, shaped ``(batch, 1)``.
        targets: Batch targets, shaped ``(batch, 1)``.
        sample_weight: Optional per-sample weights, or ``None`` to weight
            every example equally.
        key: Unused PRNG key, kept for :class:`SupervisedLoss` compatibility.
        training: Unused training flag, kept for :class:`SupervisedLoss`
            compatibility.

    Returns:
        The scalar mean squared error over the batch.
    """
    del key, training
    errors = jnp.square(model(inputs) - targets)
    if sample_weight is not None:
        errors = errors * sample_weight
    return jnp.mean(errors)


def make_dataset(
    *, num_samples: int, noise_scale: float, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample inputs and noisy targets from ``y = 3x + 2 + noise``.

    Args:
        num_samples: Number of ``(input, target)`` pairs to generate.
        noise_scale: Standard deviation of the additive Gaussian noise.
        key: PRNG key used to sample inputs and noise.

    Returns:
        A tuple of ``(inputs, targets)`` arrays, each shaped
        ``(num_samples, 1)``.
    """
    inputs_key, noise_key = jax.random.split(key)
    inputs = jax.random.uniform(inputs_key, (num_samples, 1), minval=-3.0, maxval=3.0)
    noise = noise_scale * jax.random.normal(noise_key, (num_samples, 1))
    targets = 3.0 * inputs + 2.0 + noise
    return inputs, targets


def make_loader(inputs: jax.Array, targets: jax.Array, *, batch_size: int) -> "LoaderFactory":
    """Build a :class:`LoaderFactory` that yields shuffled mini-batches.

    Args:
        inputs: The full split's input array.
        targets: The full split's target array.
        batch_size: Number of examples per mini-batch.

    Returns:
        A callable ``loader(split, epoch)`` producing an iterable of
        :class:`~probreg.core.types.Batch` objects for that epoch, with a
        fresh shuffle derived from ``epoch``.
    """

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(jax.random.key(epoch), inputs.shape[0])
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets[permutation]
        return [
            Batch(
                inputs=shuffled_inputs[start : start + batch_size],
                targets=shuffled_targets[start : start + batch_size],
            )
            for start in range(0, inputs.shape[0], batch_size)
        ]

    return loader


class InMemoryCheckpointStore(CheckpointStore):
    """A minimal :class:`CheckpointStore` that keeps checkpoints in memory."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self._checkpoints[key] = checkpoint

    def load(self, key: str) -> Checkpoint:
        return self._checkpoints[key]

    def exists(self, key: str) -> bool:
        return key in self._checkpoints


class PrintingEventSink:
    """An :class:`EventSink` that prints a one-line summary per event."""

    def on_event(self, event: TrainingEvent) -> None:
        metrics = ", ".join(
            f"{name}={value:.4f}" for name, value in event.metrics.items()
        )
        print(f"[{event.name}] epoch={event.step} {metrics}")


def main() -> None:
    """Train a linear model with early stopping and report the result."""
    data_key, model_key, rng_key = jax.random.split(jax.random.key(0), 3)
    train_inputs, train_targets = make_dataset(
        num_samples=256, noise_scale=0.5, key=data_key
    )
    validation_key = jax.random.fold_in(data_key, 1)
    validation_inputs, validation_targets = make_dataset(
        num_samples=64, noise_scale=0.5, key=validation_key
    )

    model = LinearModel(rngs=nnx.Rngs(model_key))
    optimizer = create_optimizer(model, optax.sgd(learning_rate=0.05))
    state = initialize_training_state(model, optimizer, rng_key=rng_key)

    validation = HeldOutValidation(
        model=model,
        loader=make_loader(validation_inputs, validation_targets, batch_size=64),
        loss=mean_squared_error,
    )
    early_stopper = EarlyStopper(
        metric="validation_loss",
        mode=OptimizationMode.MIN,
        patience=5,
        source=MetricSource.VALIDATION,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_loader(train_inputs, train_targets, batch_size=32),
        loss=mean_squared_error,
        state=state,
        epochs=200,
        validation=validation,
        early_stopper=early_stopper,
        event_sinks=[PrintingEventSink()],
        checkpoint_store=InMemoryCheckpointStore(),
        checkpoint_key="best",
    )

    print(f"Final training loss: {result.loss:.4f}")
    print(f"Final metrics: {result.metrics}")


if __name__ == "__main__":
    main()
