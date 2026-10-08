"""Best and finalized checkpoints, as every stage saves and restores them.

Internal to the JAX backend: the mean, variance and posterior stages share
these helpers, and none of them is exported from `probreg.jax`.

A stage saves its best checkpoint under its checkpoint key during training,
then, once trained, saves its finalized checkpoint under the same key with
metadata ``{"stage": <stage name>, "stage_complete": True}``. Only a
finalized checkpoint can be restored through the stage's ``restore``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.naming import Split, parse_metric_tag
from probreg.core.types import (
    CheckpointRef,
    ParameterRole,
    StageResult,
    StageState,
    TrainingState,
)
from probreg.jax.state import freeze_training_state

STAGE_METADATA_KEY = "stage"
"""Checkpoint metadata key naming the stage that wrote the checkpoint."""

STAGE_COMPLETE_METADATA_KEY = "stage_complete"
"""Checkpoint metadata key marking a stage's finalized checkpoint."""


def load_best_checkpoint(
    *,
    early_stopper: EarlyStopper | None,
    checkpoint_store: CheckpointStore | None,
    key: str,
) -> Checkpoint | None:
    """Return the best checkpoint a trained stage must resume from, if any.

    Args:
        early_stopper: The stage's early stopper; without one no best
            checkpoint is restored.
        checkpoint_store: The stage's checkpoint store.
        key: The stage's checkpoint key.

    Returns:
        The best checkpoint, or ``None`` without an early stopper, a store,
        or a checkpoint saved under ``key``.
    """
    if early_stopper is None or checkpoint_store is None:
        return None
    if not checkpoint_store.exists(key):
        return None
    return checkpoint_store.load(key)


def finalized_checkpoint(
    state: TrainingState,
    *,
    stage: str,
    epoch: int,
    early_stopping_state: Any,
    parameters: Any = None,
    optimizer_state: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Checkpoint:
    """Return the finalized checkpoint of a trained, ready ``state``.

    Args:
        state: The live state, frozen into the checkpoint.
        stage: Name of the stage finalizing the checkpoint.
        epoch: The epoch the finalized state was reached in.
        early_stopping_state: The early stopper's state at that epoch, or
            ``None``.
        parameters: The stage's model snapshot, if it stores one.
        optimizer_state: The stage's optimizer snapshot, if it stores one.
        metadata: Further metadata, kept unless it names the stage keys.

    Returns:
        A checkpoint marked as ``stage``'s finalized checkpoint.
    """
    return Checkpoint(
        state=freeze_training_state(state),
        epoch=epoch,
        parameters=parameters,
        optimizer_state=optimizer_state,
        rng_state=state.rng_state,
        early_stopping_state=early_stopping_state,
        metadata={
            **(metadata or {}),
            STAGE_METADATA_KEY: stage,
            STAGE_COMPLETE_METADATA_KEY: True,
        },
    )


def select_checkpoint(
    checkpoint_store: CheckpointStore | None, *, key: str, stage: str
) -> CheckpointRef:
    """Return a reference to the checkpoint a stage saves under ``key``.

    Args:
        checkpoint_store: The stage's checkpoint store.
        key: The stage's checkpoint key.
        stage: Name of the stage, recorded in the reference's metadata.

    Returns:
        Reference to ``key``, with the stage-name metadata.

    Raises:
        ValueError: If no checkpoint exists under that key.
    """
    if checkpoint_store is None or not checkpoint_store.exists(key):
        raise ValueError(f"checkpoint {key!r} is not available.")
    return CheckpointRef(key=key, metadata={STAGE_METADATA_KEY: stage})


def require_finalized(
    checkpoint: Checkpoint,
    *,
    stage: str,
    ready: StageState,
) -> None:
    """Reject a checkpoint that is not the named stage's finalized checkpoint.

    Args:
        checkpoint: The checkpoint a stage is asked to restore.
        stage: Name of the stage that must have finalized ``checkpoint``.
        ready: The lifecycle state ``stage`` finalizes its checkpoint in.

    Raises:
        ValueError: If the checkpoint's lifecycle state is not ``ready``, or
            its metadata does not name ``stage`` and mark it complete.
    """
    if (
        checkpoint.state.lifecycle_state is not ready
        or checkpoint.metadata.get(STAGE_METADATA_KEY) != stage
        or checkpoint.metadata.get(STAGE_COMPLETE_METADATA_KEY) is not True
    ):
        raise ValueError(
            f"checkpoint is not a finalized {stage!r} checkpoint: expected "
            f"lifecycle state {ready.value!r} and metadata "
            f"{{{STAGE_METADATA_KEY!r}: {stage!r}, "
            f"{STAGE_COMPLETE_METADATA_KEY!r}: True}}."
        )


def require_component_names(
    checkpoint: Checkpoint,
    *,
    model_name: str,
    role: ParameterRole,
    frozen: str | None = None,
) -> None:
    """Reject a checkpoint saved under component names other than a stage's.

    The checkpoint's parameter roles and frozen components are restored as
    saved, while the live model is registered under ``model_name``, so the
    names must agree for the restored state to pass the stage's validation.

    Args:
        checkpoint: The checkpoint a stage is asked to restore.
        model_name: Name the stage registers its model under.
        role: Role the checkpoint must give ``model_name``.
        frozen: Component the checkpoint must mark frozen, if any.

    Raises:
        ValueError: If ``model_name`` does not have ``role`` in the checkpoint,
            or ``frozen`` is not among its frozen components.
    """
    saved = checkpoint.state
    if saved.parameter_roles.get(model_name) is not role:
        raise ValueError(
            f"checkpoint does not give model component {model_name!r} the "
            f"{role.value!r} role; it was saved under other component names."
        )
    if frozen is not None and frozen not in saved.frozen_components:
        raise ValueError(
            f"checkpoint does not mark model component {frozen!r} frozen; it was "
            "saved under other component names."
        )


def latest_training_result(state: TrainingState, stage_name: str) -> StageResult:
    """Return a result reporting the stage's latest recorded training metrics.

    History keys that are not metric tags, such as ones a caller recorded
    directly, are skipped.

    Args:
        state: State whose ``metric_history`` is keyed by metric tags.
        stage_name: Stage whose training metrics to report.

    Returns:
        A result whose metrics are the last recorded value of every training
        metric of the stage, keyed by bare metric name.

    Raises:
        ValueError: If the stage recorded no training loss.
    """
    metrics = {}
    for tag, values in state.metric_history.items():
        try:
            parsed = parse_metric_tag(tag)
        except ValueError:
            continue
        if parsed.stage == stage_name and parsed.split is Split.TRAIN and values:
            metrics[parsed.metric] = values[-1]
    if "loss" not in metrics:
        raise ValueError("selected checkpoint does not contain a training loss.")
    return StageResult(state=state, metrics=metrics, loss=metrics["loss"])
