"""End-to-end CMAPSS FD001 point-RUL training and evaluation.

Loads the FD001 training and test splits, builds standardized sliding
windows, trains a 1D-CNN mean model through `probreg`'s `MeanStage`
supervised-runner contract, and reports RMSE against the official FD001
test split. The test split follows a single-window-per-trajectory
protocol: one window per test unit, ending at its last observed (possibly
truncated) cycle, scored against that unit's ground-truth RUL from
`RUL_FD001.txt`.

Run it with:

    uv run --group example-cmapss python examples/jax/cmapss/run.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Allow sibling-module imports (`data`, `model`, `preprocessing`) both when
# run as a script and when loaded from an arbitrary working directory, e.g.
# via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from data import load_fd001_data, load_fd001_test_data, load_fd001_test_rul  # noqa: E402
from model import Cnn1DMeanModel  # noqa: E402
from preprocessing import (  # noqa: E402
    apply_standardization,
    build_last_windows,
    build_windows,
    fit_standardization,
    split_by_unit,
)

from probreg.core.losses import SquaredErrorLoss
from probreg.core.metrics import rmse
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import TrainingEvent
from probreg.core.types import Batch, TrainingState
from probreg.jax import (
    HeldOutValidation,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    make_supervised_loss,
)

_SENSOR_NAMES = [f"sensor_{i}" for i in (11, 12, 4, 7, 15, 20, 21, 2, 17)]


@dataclass(frozen=True)
class CmapssConfig:
    """Configuration for the CMAPSS FD001 point-RUL example.

    Attributes:
        window_length: Number of cycles per sliding window.
        validation_fraction: Fraction of training units held out for
            validation, grouped by whole unit ID.
        batch_size: Number of windows per mini-batch.
        hidden_channels: Convolutional channel width of the mean model.
        kernel_size: Convolution kernel width along the time axis.
        learning_rate: Adam learning rate for the mean model.
        epochs: Number of mean-stage training epochs.
        seed: Base JAX random seed for splitting, model init, and training.
    """

    window_length: int = 30
    validation_fraction: float = 0.2
    batch_size: int = 32
    hidden_channels: int = 16
    kernel_size: int = 5
    learning_rate: float = 1e-3
    epochs: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate size, fraction, and epoch configuration.

        Raises:
            ValueError: If sizes/epochs are not positive or the validation
                fraction is not in (0, 1).
        """
        if self.window_length <= 0 or self.batch_size <= 0:
            raise ValueError("window_length and batch_size must be positive.")
        if self.hidden_channels <= 0 or self.kernel_size <= 0:
            raise ValueError("hidden_channels and kernel_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1).")


def make_window_loader(
    windows: np.ndarray, targets: np.ndarray, *, batch_size: int
) -> LoaderFactory:
    """Build a `LoaderFactory` yielding shuffled window/target mini-batches.

    Args:
        windows: Sliding-window inputs, shape `(n_windows, window_length,
            n_sensors)`.
        targets: Aligned RUL targets, shape `(n_windows,)`.
        batch_size: Number of windows per mini-batch.

    Returns:
        A callable `loader(split, epoch)` producing an iterable of
        `Batch` objects for that epoch, freshly shuffled from `epoch`.
    """
    inputs = jnp.asarray(windows, dtype=jnp.float32)
    targets_array = jnp.asarray(targets, dtype=jnp.float32).reshape(-1, 1)

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(jax.random.key(epoch), inputs.shape[0])
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets_array[permutation]
        return [
            Batch(
                inputs=shuffled_inputs[start : start + batch_size],
                targets=shuffled_targets[start : start + batch_size],
            )
            for start in range(0, inputs.shape[0], batch_size)
        ]

    return loader


class PrintingEventSink:
    """An event sink that prints a one-line summary per epoch."""

    def on_event(self, event: TrainingEvent) -> None:
        """Print the stage-qualified epoch metrics for one training event.

        Args:
            event: The training event to summarize.
        """
        metrics = ", ".join(
            f"{name}={value:.4f}" for name, value in event.metrics.items()
        )
        print(f"[{event.name}] epoch={event.step} {metrics}")


def train_mean_model(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    config: CmapssConfig,
) -> Cnn1DMeanModel:
    """Train a 1D-CNN mean model through `MeanStage` on windowed RUL data.

    Args:
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation RUL targets.
        config: Example configuration controlling model size and training.

    Returns:
        The trained `Cnn1DMeanModel`.
    """
    n_sensors = train_windows.shape[-1]
    model_key, rng_key = jax.random.split(jax.random.key(config.seed))
    model = Cnn1DMeanModel(
        n_sensors,
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
    loss = make_supervised_loss(SquaredErrorLoss())
    train_loader = make_window_loader(
        train_windows, train_targets, batch_size=config.batch_size
    )
    validation_loader = make_window_loader(
        validation_windows, validation_targets, batch_size=config.batch_size
    )

    stage = MeanStage(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss=loss,
        options=SupervisedStageOptions(
            epochs=config.epochs,
            validation=HeldOutValidation(
                model=model,
                loader=validation_loader,
                loss=loss,
            ),
            event_sinks=[PrintingEventSink()],
        ),
    )
    state = TrainingState(rng_state=rng_key)
    stage.prepare(state)
    stage.train(state)
    return model


def evaluate_rmse(
    model: Cnn1DMeanModel, test_windows: np.ndarray, test_rul: np.ndarray
) -> float:
    """Compute RMSE of the trained model's point RUL predictions.

    Args:
        model: A trained mean model producing shape `(batch, 1)`
            predictions.
        test_windows: One trailing window per test unit, shape `(n_units,
            window_length, n_sensors)`.
        test_rul: Ground-truth RUL per test unit, shape `(n_units,)`.

    Returns:
        The root mean squared error between predicted and true RUL.
    """
    evaluation_model = nnx.clone(model)
    evaluation_model.eval()
    predictions = jax.device_get(
        evaluation_model(jnp.asarray(test_windows, dtype=jnp.float32))
    ).reshape(-1)
    return rmse(test_rul, predictions)


def prepare_cmapss_windows(
    config: CmapssConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, split, standardize, and window the FD001 train/test splits.

    Args:
        config: Example configuration controlling the window length and
            the train/validation split.

    Returns:
        A tuple `(train_windows, train_targets, validation_windows,
        validation_targets, test_windows, test_rul)`. Standardization
        statistics are fit on the training subset only and reused for the
        validation and test subsets.
    """
    train_data = load_fd001_data()
    train_df, validation_df = split_by_unit(
        train_data, test_size=config.validation_fraction, random_state=config.seed
    )
    stats = fit_standardization(train_df, _SENSOR_NAMES)
    train_df = apply_standardization(train_df, stats)
    validation_df = apply_standardization(validation_df, stats)

    train_windows, train_targets = build_windows(
        train_df, _SENSOR_NAMES, window_length=config.window_length
    )
    validation_windows, validation_targets = build_windows(
        validation_df, _SENSOR_NAMES, window_length=config.window_length
    )

    test_data = apply_standardization(load_fd001_test_data(), stats)
    test_windows = build_last_windows(
        test_data, _SENSOR_NAMES, window_length=config.window_length
    )
    test_rul = load_fd001_test_rul()

    return (
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        test_windows,
        test_rul,
    )


def main() -> None:
    """Train the CMAPSS mean model end to end and print the test RMSE."""
    config = CmapssConfig()
    (
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        test_windows,
        test_rul,
    ) = prepare_cmapss_windows(config)

    model = train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
    )
    test_rmse = evaluate_rmse(model, test_windows, test_rul)
    print(f"FD001 test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
