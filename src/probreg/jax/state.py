"""NNX and Optax state adapters for the JAX backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.checkpoints import Checkpoint
from probreg.core.types import TrainingState


@dataclass(frozen=True)
class NnxSnapshot:
    """An opaque, complete snapshot of an NNX graph and its state.

    Attributes:
        graphdef: The static NNX graph definition.
        state: The NNX variable state captured at snapshot time.
    """

    graphdef: nnx.GraphDef
    state: nnx.State


def create_optimizer(
    model: nnx.Module, transformation: optax.GradientTransformation
) -> nnx.Optimizer:
    """Create an NNX optimizer that updates all model parameters.

    Args:
        model: The NNX module whose parameters will be optimized.
        transformation: The Optax gradient transformation defining the
            optimization algorithm.

    Returns:
        An ``nnx.Optimizer`` bound to ``model``'s parameters.
    """
    return nnx.Optimizer(model, transformation, wrt=nnx.Param)


def initialize_training_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    *,
    rng_key: jax.Array,
    model_name: str = "model",
    optimizer_name: str = "optimizer",
) -> TrainingState:
    """Create generic training state backed by an NNX model and optimizer.

    Args:
        model: The NNX module to register as a model component.
        optimizer: The NNX optimizer to register as an optimizer state.
        rng_key: The initial JAX PRNG key for the training state.
        model_name: Key under which ``model`` is stored in
            ``TrainingState.model_components``. Defaults to ``"model"``.
        optimizer_name: Key under which ``optimizer`` is stored in
            ``TrainingState.optimizer_states``. Defaults to
            ``"optimizer"``.

    Returns:
        A :class:`TrainingState` with ``model`` and ``optimizer``
        registered and ``rng_state`` initialized to ``rng_key``.
    """
    return TrainingState(
        model_components={model_name: model},
        optimizer_states={optimizer_name: optimizer},
        rng_state=rng_key,
    )


def freeze_training_state(state: TrainingState) -> TrainingState:
    """Create an independent, checkpoint-safe snapshot of a training state.

    ``TrainingState`` holds several mutable containers (``metric_history``,
    ``checkpoint_registry``, ``parameter_roles``) that training runners
    keep appending to and mutating in place across epochs. A plain
    ``dataclasses.replace`` only shallow-copies the state, so a checkpoint
    built that way would keep sharing those containers with live
    training and silently drift after being "saved". This function
    deep-copies every mutable container field instead.

    The live ``model_components`` and ``optimizer_states`` are dropped
    from the returned snapshot rather than copied, since they hold
    mutable NNX modules/optimizers that keep changing across epochs;
    callers that need an immutable capture of model or optimizer state
    should use :func:`snapshot` instead and store it alongside the
    frozen training state (see
    :class:`probreg.core.checkpoints.Checkpoint`).

    Args:
        state: The live training state to snapshot.

    Returns:
        A new :class:`TrainingState` whose mutable container fields are
        independent copies, unaffected by later in-place mutation of
        ``state``, and whose ``model_components``/``optimizer_states``
        are empty.
    """
    return replace(
        state,
        model_components={},
        optimizer_states={},
        parameter_roles=dict(state.parameter_roles),
        checkpoint_registry=dict(state.checkpoint_registry),
        metric_history={
            name: list(values) for name, values in state.metric_history.items()
        },
    )


def snapshot(module: nnx.Module) -> NnxSnapshot:
    """Capture an NNX module's graph definition and complete variable state.

    Args:
        module: The NNX module to snapshot.

    Returns:
        An :class:`NnxSnapshot` containing ``module``'s graph definition
        and variable state at the time of the call.
    """
    graphdef, state = nnx.split(module)
    copied_state = jax.tree.map(_copy_snapshot_leaf, state)
    return NnxSnapshot(graphdef=graphdef, state=copied_state)


def restore_checkpoint(
    checkpoint: Checkpoint,
    *,
    state: TrainingState,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    model_name: str = "model",
    optimizer_name: str = "optimizer",
) -> None:
    """Restore a JAX checkpoint into existing live training objects.

    The model and optimizer are updated in place so callers retaining their
    identities, including stage registries, continue to reference the restored
    objects. Checkpoint-safe training states omit mutable model and optimizer
    objects, so this helper re-registers the supplied live objects after
    restoring the backend-neutral state fields.

    Args:
        checkpoint: Checkpoint containing NNX model and optimizer snapshots.
        state: Existing shared training state to restore in place.
        model: Live model object updated from ``checkpoint.parameters``.
        optimizer: Live optimizer updated from ``checkpoint.optimizer_state``.
        model_name: Registry name for ``model``. Defaults to ``"model"``.
        optimizer_name: Registry name for ``optimizer``. Defaults to
            ``"optimizer"``.

    Raises:
        TypeError: If the checkpoint does not contain NNX snapshots or a JAX
            random key.
        ValueError: If the live model or optimizer is incompatible with the
            checkpoint snapshot.
    """
    if not isinstance(checkpoint.parameters, NnxSnapshot):
        raise TypeError("checkpoint.parameters must be an NnxSnapshot.")
    if not isinstance(checkpoint.optimizer_state, NnxSnapshot):
        raise TypeError("checkpoint.optimizer_state must be an NnxSnapshot.")
    if not isinstance(checkpoint.rng_state, jax.Array):
        raise TypeError("checkpoint.rng_state must be a JAX random key.")

    _validate_graph(model, checkpoint.parameters, kind="model")
    _validate_state_structure(
        optimizer,
        checkpoint.optimizer_state,
        kind="optimizer",
    )
    restored_optimizer = nnx.merge(
        checkpoint.optimizer_state.graphdef,
        checkpoint.optimizer_state.state,
    )
    nnx.update(model, checkpoint.parameters.state)
    _restore_optimizer_static_fields(optimizer, restored_optimizer)
    nnx.update(optimizer, checkpoint.optimizer_state.state)
    _restore_training_state(state, checkpoint.state)
    state.rng_state = checkpoint.rng_state
    state.register_component(model_name, model)
    state.register_optimizer(optimizer_name, optimizer)


def _restore_training_state(state: TrainingState, restored: TrainingState) -> None:
    """Copy checkpointed workflow fields into an existing state object."""
    state.model_components = {}
    state.parameter_roles = dict(restored.parameter_roles)
    state.frozen_components = frozenset(restored.frozen_components)
    state.optimizer_states = {}
    state.posterior_state = restored.posterior_state
    state.rng_state = restored.rng_state
    state.lifecycle_state = restored.lifecycle_state
    state.stage = restored.stage
    state.outer_iteration = restored.outer_iteration
    state.data_fingerprint = restored.data_fingerprint
    state.checkpoint_registry = dict(restored.checkpoint_registry)
    state.metric_history = {
        name: list(values) for name, values in restored.metric_history.items()
    }


def _copy_snapshot_leaf(value: object) -> object:
    """Return a storage-independent copy of one NNX state leaf."""
    if isinstance(value, jax.Array):
        return jnp.array(value, copy=True)
    return copy.deepcopy(value)


def _validate_graph(
    module: nnx.Module,
    restored: NnxSnapshot,
    *,
    kind: str,
) -> None:
    """Reject a module whose static NNX graph differs from the snapshot."""
    graphdef, _ = nnx.split(module)
    if graphdef != restored.graphdef:
        raise ValueError(f"{kind} graph is incompatible with checkpoint state.")


def _restore_optimizer_static_fields(
    optimizer: nnx.Optimizer,
    restored: nnx.Optimizer,
) -> None:
    """Copy checkpointed optimizer static fields into an existing optimizer."""
    optimizer.graph = restored.graph
    optimizer.tx = restored.tx
    optimizer.wrt = restored.wrt


def _validate_state_structure(
    module: nnx.Module,
    restored: NnxSnapshot,
    *,
    kind: str,
) -> None:
    """Reject incompatible variable paths, shapes, or dtypes before mutation."""
    _, current_state = nnx.split(module)
    if jax.tree.structure(current_state) != jax.tree.structure(restored.state):
        raise ValueError(f"{kind} state is incompatible with checkpoint state.")
    current_leaves = jax.tree.leaves(current_state)
    restored_leaves = jax.tree.leaves(restored.state)
    if any(
        current.shape != saved.shape or current.dtype != saved.dtype
        for current, saved in zip(current_leaves, restored_leaves, strict=True)
    ):
        raise ValueError(f"{kind} state is incompatible with checkpoint state.")
