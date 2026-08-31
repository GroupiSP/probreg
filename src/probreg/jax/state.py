"""NNX and Optax state adapters for the JAX backend."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import optax
from flax import nnx

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
    return NnxSnapshot(graphdef=graphdef, state=state)
