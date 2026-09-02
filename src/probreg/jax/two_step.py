"""JAX objectives for explicit mean and Gamma variance training stages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from probreg.core.losses import GammaResidualNLLLoss
from probreg.core.protocols import LoaderFactory
from probreg.core.types import Batch
from probreg.jax.evaluation import SupervisedLoss

_DEFAULT_GAMMA_RESIDUAL_LOSS = GammaResidualNLLLoss()


def make_mean_squared_error_loss(
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a supervised mean-squared-error objective.

    Args:
        reduction: Callable reducing the optionally weighted squared errors
            to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A supervised loss for deterministic mean models.
    """

    def mean_squared_error(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        errors = jnp.square(model(inputs) - targets)
        if sample_weight is not None:
            errors = errors * sample_weight
        return reduction(errors)

    return mean_squared_error


def make_gamma_residual_loss(
    loss: GammaResidualNLLLoss = _DEFAULT_GAMMA_RESIDUAL_LOSS,
    *,
    reduction: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> SupervisedLoss:
    """Build a supervised Gamma NLL objective for squared residual targets.

    Args:
        loss: Backend-neutral Gamma residual objective.
        reduction: Callable reducing the optionally weighted per-example
            losses to a scalar. Defaults to ``jnp.mean``.

    Returns:
        A supervised loss for models returning Gamma predictions.
    """

    def gamma_residual_loss(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        prediction = model(inputs)
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        per_example = loss.per_example(prediction, batch)
        if sample_weight is not None:
            per_example = per_example * sample_weight
        return reduction(per_example)

    return gamma_residual_loss


def materialize_residual_loader(
    mean_model: nnx.Module,
    source_loader: LoaderFactory,
    *,
    splits: Sequence[str] = ("train", "validation"),
    source_epoch: int = 0,
) -> LoaderFactory:
    """Materialize detached squared residual targets for fixed data splits.

    The source loader is consumed exactly once per configured split. The
    returned loader replays the cached batches for every later epoch, making
    the Step 1 mean predictions and source data a fixed Step 2 training
    snapshot.

    Args:
        mean_model: Trained deterministic mean model.
        source_loader: Loader providing inputs and original regression targets.
        splits: Split names to materialize. Defaults to training and validation.
        source_epoch: Source-loader epoch used for the one-time snapshot.

    Returns:
        A loader factory serving cached batches whose targets are detached
        squared residuals.

    Raises:
        ValueError: If splits are empty, duplicated, unknown at replay time, or
            produce no batches; if a source batch lacks targets; or if mean
            predictions and targets have different shapes.
    """
    split_names = tuple(splits)
    if not split_names:
        raise ValueError("splits must contain at least one split name.")
    if any(not name for name in split_names):
        raise ValueError("split names must not be empty.")
    if len(set(split_names)) != len(split_names):
        raise ValueError("split names must be unique.")

    prediction_model = nnx.clone(mean_model)
    prediction_model.eval()
    cached: dict[str, tuple[Batch, ...]] = {}

    for split in split_names:
        residual_batches = tuple(
            _materialize_residual_batch(prediction_model, batch)
            for batch in source_loader(split=split, epoch=source_epoch)
        )
        if not residual_batches:
            raise ValueError(f"source split {split!r} must provide at least one batch.")
        cached[split] = residual_batches

    def residual_loader(*, split: str, epoch: int) -> tuple[Batch, ...]:
        del epoch
        if split not in cached:
            raise ValueError(f"residual split {split!r} was not materialized.")
        return cached[split]

    return residual_loader


def _materialize_residual_batch(mean_model: nnx.Module, batch: Batch) -> Batch:
    """Return one batch with detached squared residual targets."""
    if batch.targets is None:
        raise ValueError("source batches must provide targets.")
    predictions = mean_model(batch.inputs)
    if predictions.shape != batch.targets.shape:
        raise ValueError("mean predictions and targets must have matching shapes.")
    residuals = jax.lax.stop_gradient(jnp.square(batch.targets - predictions))
    return Batch(
        inputs=batch.inputs,
        targets=residuals,
        sample_weight=batch.sample_weight,
        metadata=batch.metadata,
    )
