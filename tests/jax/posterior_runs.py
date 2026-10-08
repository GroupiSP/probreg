"""A small regression problem and real mean and variance runs on it.

The posterior-stage tests of every inference method start from the same
variance-ready state, so its data, loader and stage builders live here once.
The `make_stages` and `variance_ready_run` fixtures in `conftest.py` wrap the
builders; the plain names below are imported directly because module-level
helpers and Hypothesis `@given` bodies cannot take fixtures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.checkpoints import Checkpoint, InMemoryCheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.tracking import EventSink
from probreg.core.types import Batch, PyTree, TrainingState
from probreg.jax import (
    GammaHead,
    GammaVarianceStage,
    HeldOutValidation,
    MeanStage,
    SupervisedLoss,
    SupervisedStageOptions,
    create_optimizer,
)

#: The number of training examples ``N`` of `regression_loader`.
DATASET_SIZE = 16
_BATCH_SIZE = 8


def regression_data() -> tuple[jax.Array, jax.Array]:
    """Sixteen noisy points on the line ``y = 2x``."""
    data_key = jax.random.key(3)
    inputs = jnp.linspace(-1.0, 1.0, DATASET_SIZE)[:, None]
    targets = 2.0 * inputs + 0.3 * jax.random.normal(data_key, inputs.shape)
    return inputs, targets


def regression_loader(*, split: str, epoch: int) -> list[Batch]:
    """Two batches of eight; the validation split is shifted by 0.25."""
    del epoch
    inputs, targets = regression_data()
    offset = 0.25 if split == "validation" else 0.0
    return [
        Batch(
            inputs=inputs[start : start + _BATCH_SIZE],
            targets=targets[start : start + _BATCH_SIZE] + offset,
        )
        for start in range(0, DATASET_SIZE, _BATCH_SIZE)
    ]


class RecordingStore(InMemoryCheckpointStore):
    """A store that also records every checkpoint saved, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.saves: list[tuple[str, Checkpoint]] = []

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.saves.append((key, checkpoint))
        super().save(key, checkpoint)


@dataclass
class VarianceReadyRun:
    """A training state after real mean and variance stages."""

    state: TrainingState
    mean_model: nnx.Module
    variance_model: nnx.Module


MakeVarianceReadyRun = Callable[..., VarianceReadyRun]


@dataclass
class MeanAndVarianceStages:
    """A mean and a variance stage with fresh models, ready to train or restore."""

    mean: MeanStage
    variance: GammaVarianceStage


MakeStages = Callable[..., MeanAndVarianceStages]


def build_stages(
    linear_model: type[Any],
    squared_error: SupervisedLoss,
    *sinks: EventSink,
    seed: int = 0,
    checkpoint_store: InMemoryCheckpointStore | None = None,
) -> MeanAndVarianceStages:
    """Build both stages; a checkpoint store also adds patient early stoppers.

    Args:
        linear_model: The mean model's class.
        squared_error: The mean stage's validation loss.
        *sinks: The event sinks attached to both stages.
        seed: Seeds the fresh models' initial weights.
        checkpoint_store: Where both stages save their checkpoints.

    Returns:
        The two stages.
    """
    stopper = (
        None
        if checkpoint_store is None
        else EarlyStopper(metric="loss", mode="min", patience=100)
    )
    mean_model = linear_model(rngs=nnx.Rngs(seed))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.adam(0.1)),
        train_loader=regression_loader,
        options=SupervisedStageOptions(
            epochs=20,
            validation=HeldOutValidation(
                model=mean_model, loader=regression_loader, loss=squared_error
            ),
            event_sinks=sinks,
            early_stopper=stopper,
            checkpoint_store=checkpoint_store,
        ),
    )
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(seed + 2))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(0.05)),
        source_loader=regression_loader,
        options=SupervisedStageOptions(
            epochs=5,
            event_sinks=sinks,
            early_stopper=stopper,
            checkpoint_store=checkpoint_store,
        ),
        validation_factory=lambda residuals: HeldOutValidation(
            model=variance_model, loader=residuals, loss=variance_stage.loss
        ),
    )
    return MeanAndVarianceStages(mean_stage, variance_stage)


def run_to_variance_ready(
    stages: MeanAndVarianceStages, *, mean_only: bool = False
) -> VarianceReadyRun:
    """Train a validated mean stage and, unless ``mean_only``, a variance stage.

    Args:
        stages: Fresh stages from `build_stages`.
        mean_only: Stop after the mean stage.

    Returns:
        The shared state and the two trained models.
    """
    state = TrainingState(rng_state=jax.random.key(1))
    stages.mean.prepare(state)
    stages.mean.train(state)
    if not mean_only:
        stages.variance.prepare(state)
        stages.variance.train(state)
    return VarianceReadyRun(state, stages.mean.model, stages.variance.model)


def restore_mean_and_variance(
    stages: MeanAndVarianceStages, store: InMemoryCheckpointStore
) -> TrainingState:
    """Restore the finalized mean and variance checkpoints into a fresh state."""
    state = TrainingState()
    stages.mean.restore(state, store.load("mean/best"))
    stages.variance.restore(state, store.load("variance/best"))
    return state


def leaves_equal(left: PyTree, right: PyTree) -> bool:
    """Whether two trees hold exactly equal leaves."""
    return all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )
