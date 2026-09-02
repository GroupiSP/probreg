"""Train an MVE model on a heteroscedastic toy regression dataset.

This example wires together :mod:`probreg.jax` and :mod:`probreg.core` to
run :func:`probreg.jax.run_supervised` with a mean-variance estimation (MVE)
loss:

* a synthetic ``y = 2x + noise`` regression dataset where the noise scale
  grows with ``x``, split into a training and a held-out validation set;
* a :class:`~probreg.jax.distributions.GaussianHead` predicting both a mean
  and a positive aleatoric scale for each input, trained with an Optax Adam
  optimizer;
* a Gaussian negative log-likelihood loss
  (:class:`~probreg.core.losses.GaussianNLLLoss`), adapted into a
  :class:`~probreg.jax.evaluation.SupervisedLoss` by
  :func:`~probreg.jax.mve.make_mve_loss`;
* composable RMSE and sample/grid-based point-CRPS metrics for both training
  and held-out validation;
* a :class:`~probreg.jax.HeldOutValidation` strategy evaluated after every
  epoch;
* an :class:`~probreg.core.early_stopping.EarlyStopper` monitoring the
  validation loss, backed by an in-memory checkpoint store that persists
  the best model seen so far.

Run it with:

    uv run --extra jax --extra plot python examples/mve_regression_jax.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper, MetricSource, OptimizationMode
from probreg.core.losses import GaussianNLLLoss
from probreg.core.metric_registry import (
    EvaluationGrid,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
)
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import TrainingEvent
from probreg.core.types import Batch
from probreg.jax import (
    GaussianHead,
    GaussianPredictor,
    HeldOutValidation,
    MetricSuite,
    create_optimizer,
    initialize_training_state,
    run_supervised,
)
from probreg.jax.mve import make_mve_loss


def make_dataset(*, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sample inputs and noisy targets from ``y = 2x + noise``.

    The noise standard deviation grows linearly with ``x``, so the dataset
    is heteroscedastic: an MVE model should learn a wider predictive scale
    for larger inputs.

    Args:
        num_samples: Number of ``(input, target)`` pairs to generate.
        key: PRNG key used to sample inputs and noise.

    Returns:
        A tuple of ``(inputs, targets)`` arrays, each shaped
        ``(num_samples, 1)``.
    """
    inputs_key, noise_key = jax.random.split(key)
    inputs = jax.random.uniform(inputs_key, (num_samples, 1), minval=-3.0, maxval=3.0)
    noise_scale = 0.1 + 0.2 * (inputs + 3.0)
    noise = noise_scale * jax.random.normal(noise_key, (num_samples, 1))
    targets = 2.0 * inputs + noise
    return inputs, targets


def make_loader(
    inputs: jax.Array, targets: jax.Array, *, batch_size: int
) -> LoaderFactory:
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
    """Print train/validation loss, RMSE, and point-CRPS periodically.

    Only ``validation_end`` events are handled, and only every ``every``
    epochs, to keep console output readable across long training runs.

    Attributes:
        every: Print only on epochs that are a multiple of this value.
            Defaults to ``10``.
    """

    def __init__(self, *, every: int = 10) -> None:
        """Initialize the sink.

        Args:
            every: Print only on epochs that are a multiple of this
                value. Defaults to ``10``.
        """
        self.every = every

    def on_event(self, event: TrainingEvent) -> None:
        """Print current training/validation metrics, if due.

        Args:
            event: The training event to (conditionally) report on.
        """
        if event.name != "validation_end" or event.step % self.every != 0:
            return
        history = event.state.metric_history
        print(
            f"epoch={event.step} "
            f"training_loss={history['training_loss'][-1]:.4f} "
            f"training_rmse={history['training_rmse'][-1]:.4f} "
            f"training_point_crps={history['training_point_crps'][-1]:.4f} "
            f"validation_loss={event.metrics['validation_loss']:.4f} "
            f"validation_rmse={event.metrics['validation_rmse']:.4f} "
            f"validation_point_crps={event.metrics['validation_point_crps']:.4f}"
        )


def plot_predictions(
    model: nnx.Module,
    inputs: jax.Array,
    targets: jax.Array,
    *,
    save_path: str | None = None,
) -> None:
    """Plot validation targets against the model's mean and ±2σ interval.

    Args:
        model: An NNX module mapping inputs directly to a
            :class:`~probreg.core.distributions.PredictiveDistribution`
            (e.g. a :class:`~probreg.jax.distributions.GaussianHead`).
        inputs: Validation inputs, shaped ``(n, 1)``.
        targets: Validation targets, shaped ``(n, 1)``.
        save_path: If given, save the figure to this path instead of
            displaying it interactively.
    """
    order = jnp.argsort(inputs.squeeze(-1))
    sorted_inputs = inputs[order]
    x = np.asarray(sorted_inputs.squeeze(-1))
    y = np.asarray(targets[order].squeeze(-1))

    prediction = model(sorted_inputs)
    mean = np.asarray(prediction.mean().squeeze(-1))
    scale = np.asarray(jnp.sqrt(prediction.variance()).squeeze(-1))

    _, ax = plt.subplots()
    ax.scatter(x, y, s=10, alpha=0.6, label="validation data")
    ax.plot(x, mean, color="C1", label="predicted mean")
    ax.fill_between(
        x, mean - 2 * scale, mean + 2 * scale, color="C1", alpha=0.2, label="±2σ"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MVE predictions vs. validation data")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main() -> None:
    """Train an MVE model with early stopping and report the result."""
    data_key, model_key, rng_key = jax.random.split(jax.random.key(0), 3)
    train_inputs, train_targets = make_dataset(num_samples=256, key=data_key)
    validation_key = jax.random.fold_in(data_key, 1)
    validation_inputs, validation_targets = make_dataset(
        num_samples=64, key=validation_key
    )

    model = GaussianHead(1, 1, rngs=nnx.Rngs(model_key))
    optimizer = create_optimizer(model, optax.adam(learning_rate=0.05))
    state = initialize_training_state(model, optimizer, rng_key=rng_key)

    mve_loss = make_mve_loss(GaussianNLLLoss())
    metric_suite = MetricSuite(
        epoch=(
            RootMeanSquaredError(),
            PointContinuousRankedProbabilityScore(),
        ),
        predictor=GaussianPredictor(),
        predictive_sample_count=128,
        evaluation_grid=EvaluationGrid(np.linspace(-10.0, 10.0, 401)),
    )
    validation = HeldOutValidation(
        model=model,
        loader=make_loader(validation_inputs, validation_targets, batch_size=64),
        loss=mve_loss,
        metrics=metric_suite,
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
        loss=mve_loss,
        state=state,
        epochs=200,
        validation=validation,
        early_stopper=early_stopper,
        event_sinks=[PrintingEventSink()],
        checkpoint_store=InMemoryCheckpointStore(),
        checkpoint_key="best",
        metrics=metric_suite,
    )

    print(f"Final training loss: {result.loss:.4f}")
    print(f"Final training RMSE: {result.metrics['rmse']:.4f}")
    print(f"Final training point-CRPS: {result.metrics['point_crps']:.4f}")
    print(
        "Final validation RMSE: "
        f"{result.state.metric_history['validation_rmse'][-1]:.4f}"
    )
    print(
        "Final validation point-CRPS: "
        f"{result.state.metric_history['validation_point_crps'][-1]:.4f}"
    )
    print(f"Final training metrics: {result.metrics}")

    low_uncertainty_prediction = model(jnp.array([[-3.0]]))
    high_uncertainty_prediction = model(jnp.array([[3.0]]))
    print(
        "Predicted scale at x=-3: "
        f"{float(low_uncertainty_prediction.scale.squeeze()):.4f}"
    )
    print(
        "Predicted scale at x=3: "
        f"{float(high_uncertainty_prediction.scale.squeeze()):.4f}"
    )

    plot_predictions(model, validation_inputs, validation_targets)


if __name__ == "__main__":
    main()
