"""Train a tracked MVE model and watch the whole run in TensorBoard.

The regression problem is `examples/jax/mve_regression.py`'s, restated
here so that this example stands alone as the file a reader copies:
heteroscedastic noise on a linear mean, held-out validation, early
stopping on the validation loss, best-model checkpointing, all with the
same hyperparameters. There is one exception — the early-stopping
patience is raised from 5 to 20, so the tracked curves are long enough to
be worth opening TensorBoard for. What this example adds is the tracking:

* a `TensorBoardTracker` (in `tensorboard_tracker.py`, the file to copy)
  implementing `probreg.core.tracking.ExperimentTracker`;
* `probreg.core.tracking.TrackerEventSink`, the library's bridge from
  training events to that tracker, which namespaces every metric by its
  stage and its event;
* the example's own printing event sink, passed *alongside* the tracker
  sink: adding a tracker displaces nothing.

Each invocation writes to its own UTC-timestamped subdirectory of
`--logdir`, so successive runs appear side by side in TensorBoard rather
than interleaving into one broken curve.

Run it with:

    uv run --group example-tracking python examples/jax/tracking/run.py \
        --logdir runs/

then view the run with:

    uv run --group example-tracking tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx

# The figure exists to be logged, never to be shown: a plot window would
# undercut the example's point and break headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Allow the sibling-module import (`tensorboard_tracker`) both when run as
# a script and when loaded from an arbitrary working directory, e.g. via
# `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from tensorboard_tracker import TensorBoardTracker

from probreg.core.checkpoints import InMemoryCheckpointStore
from probreg.core.early_stopping import EarlyStopper, MetricSource, OptimizationMode
from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.core.metric_registry import (
    EvaluationGrid,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
)
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import ExperimentTracker, TrackerEventSink, TrainingEvent
from probreg.core.types import Batch
from probreg.jax import (
    GaussianHead,
    GaussianPredictor,
    HeldOutValidation,
    MetricSuite,
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

TRAIN_SAMPLES = 256
VALIDATION_SAMPLES = 64
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 64
LEARNING_RATE = 0.05
EPOCHS = 200
PATIENCE = 20
PREDICTIVE_SAMPLE_COUNT = 128
SEED = 0
INTERVAL_MULTIPLIER = 1.96
"""Half-width of the 95% predictive interval, in predictive scales."""


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


class PrintingEventSink:
    """Print train/validation loss, RMSE, and point-CRPS periodically.

    Only ``validation_end`` events are handled, and only every ``every``
    epochs, to keep console output readable across long training runs.
    This sink is passed alongside the tracker sink: the terminal output a
    reader already has stays exactly as it was.

    Validation metrics are read unprefixed, since this example clears the
    validation strategy's metric prefix so that the tracker owns the tag.

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
            f"validation_loss={event.metrics['loss']:.4f} "
            f"validation_rmse={event.metrics['rmse']:.4f} "
            f"validation_point_crps={event.metrics['point_crps']:.4f}"
        )


def plot_predictions(
    model: nnx.Module, inputs: jax.Array, targets: jax.Array
) -> Figure:
    """Plot validation targets against the mean and its predictive interval.

    Args:
        model: An NNX module mapping inputs directly to a
            :class:`~probreg.core.distributions.PredictiveDistribution`
            (e.g. a :class:`~probreg.jax.distributions.GaussianHead`).
        inputs: Validation inputs, shaped ``(n, 1)``.
        targets: Validation targets, shaped ``(n, 1)``.

    Returns:
        The figure, for the caller to log and close. It is never shown:
        the Matplotlib backend is non-interactive.
    """
    order = jnp.argsort(inputs.squeeze(-1))
    sorted_inputs = inputs[order]
    x = np.asarray(sorted_inputs.squeeze(-1))
    y = np.asarray(targets[order].squeeze(-1))

    prediction = model(sorted_inputs)
    mean = np.asarray(prediction.mean().squeeze(-1))
    scale = np.asarray(jnp.sqrt(prediction.variance()).squeeze(-1))
    half_width = INTERVAL_MULTIPLIER * scale

    figure, ax = plt.subplots()
    ax.scatter(x, y, s=10, alpha=0.6, label="validation data")
    ax.plot(x, mean, color="C1", label="predicted mean")
    ax.fill_between(
        x,
        mean - half_width,
        mean + half_width,
        color="C1",
        alpha=0.2,
        label="95% predictive interval",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MVE predictions vs. validation data")
    ax.legend()
    return figure


def run_tracked_training(tracker: ExperimentTracker, *, epochs: int = EPOCHS) -> None:
    """Train the MVE model, recording the whole run to a tracker.

    Typed against :class:`~probreg.core.tracking.ExperimentTracker`, so
    swapping TensorBoard for another tracker changes one constructor line
    in :func:`main` and nothing here.

    Args:
        tracker: The tracker receiving the run's parameters, per-epoch
            metrics and final figure.
        epochs: Maximum number of epochs, before early stopping.

    Returns:
        None.
    """
    data_key, model_key, rng_key = jax.random.split(jax.random.key(SEED), 3)
    train_inputs, train_targets = make_dataset(num_samples=TRAIN_SAMPLES, key=data_key)
    validation_key = jax.random.fold_in(data_key, 1)
    validation_inputs, validation_targets = make_dataset(
        num_samples=VALIDATION_SAMPLES, key=validation_key
    )

    model = GaussianHead(1, 1, rngs=nnx.Rngs(model_key))
    optimizer = create_optimizer(model, optax.adam(learning_rate=LEARNING_RATE))
    state = initialize_training_state(model, optimizer, rng_key=rng_key)

    loss_definition = NegativeLogLikelihoodLoss()
    mve_loss = make_supervised_loss(loss_definition)
    epoch_metrics = (
        RootMeanSquaredError(),
        PointContinuousRankedProbabilityScore(),
    )
    metric_suite = MetricSuite(
        epoch=epoch_metrics,
        predictor=GaussianPredictor(),
        predictive_sample_count=PREDICTIVE_SAMPLE_COUNT,
        evaluation_grid=EvaluationGrid(np.linspace(-10.0, 10.0, 401)),
    )
    # The validation prefix is cleared so that the tracker owns the whole
    # tag: `TrackerEventSink` then yields matched `<stage>/train/loss` and
    # `<stage>/validation/loss` tags, which TensorBoard renders as two
    # series on one chart.
    validation = HeldOutValidation(
        model=model,
        loader=make_loader(
            validation_inputs, validation_targets, batch_size=VALIDATION_BATCH_SIZE
        ),
        loss=mve_loss,
        metrics=metric_suite,
        metric_prefix="",
    )
    early_stopper = EarlyStopper(
        metric="loss",
        mode=OptimizationMode.MIN,
        patience=PATIENCE,
        source=MetricSource.VALIDATION,
    )

    tracker.log_params(
        {
            "optimizer": {"name": "adam", "learning_rate": LEARNING_RATE},
            "data": {
                "train_samples": TRAIN_SAMPLES,
                "validation_samples": VALIDATION_SAMPLES,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "validation_batch_size": VALIDATION_BATCH_SIZE,
                "seed": SEED,
            },
            "training": {"epochs": epochs, "patience": PATIENCE},
            # What was optimized, not only how fast: two runs differing in
            # objective are otherwise indistinguishable in the HParams table.
            "loss": type(loss_definition).__name__,
            "metrics": tuple(type(metric).__name__ for metric in epoch_metrics),
            "predictive_sample_count": PREDICTIVE_SAMPLE_COUNT,
        }
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_loader(
            train_inputs, train_targets, batch_size=TRAIN_BATCH_SIZE
        ),
        loss=mve_loss,
        state=state,
        epochs=epochs,
        validation=validation,
        early_stopper=early_stopper,
        event_sinks=[PrintingEventSink(), TrackerEventSink(tracker)],
        checkpoint_store=InMemoryCheckpointStore(),
        checkpoint_key="best",
        metrics=metric_suite,
    )

    print(f"Final training loss: {result.loss:.4f}")
    print(f"Final training RMSE: {result.metrics['rmse']:.4f}")
    print(f"Final training point-CRPS: {result.metrics['point_crps']:.4f}")

    figure = plot_predictions(model, validation_inputs, validation_targets)
    try:
        tracker.log_artifact("predictions", figure)
    finally:
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the tracked training and write its TensorBoard event files.

    Args:
        argv: Command-line arguments. Defaults to the process arguments;
            pass a sequence to drive the example programmatically, as the
            end-to-end test does.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Train a TensorBoard-tracked MVE model."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs"),
        help=(
            "Directory holding the runs. Each invocation writes to its own "
            "UTC-timestamped subdirectory of it. Defaults to 'runs/'."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            f"Maximum number of epochs, before early stopping. Defaults to {EPOCHS}."
        ),
    )
    args = parser.parse_args(argv)

    run_directory = args.logdir / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    tracker = TensorBoardTracker(run_directory)
    try:
        run_tracked_training(tracker, epochs=args.epochs)
    finally:
        tracker.close()
    print(f"Wrote the run to {run_directory}.")
    print(f"View it with: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
