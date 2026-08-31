"""Shared NNX evaluation primitives used by training runners and validation strategies.

This module has no dependency on any specific training runner, so both
:mod:`probreg.jax.supervised` and :mod:`probreg.jax.validation` may depend
on it without either depending on the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import jax
from flax import nnx

from probreg.core.types import Batch
from probreg.jax.rng import split_key


class SupervisedLoss(Protocol):
    """A callable computing a supervised loss for an NNX model.

    Implementations compute a scalar loss from a model, a batch of inputs/targets/sample
    weights, a PRNG key (e.g. for stochastic layers such as dropout), and a flag
    indicating whether the call happens during training (as opposed to evaluation).
    """

    def __call__(
        self,
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
        training: bool,
        /,
    ) -> jax.Array:
        """Compute the scalar supervised loss for a batch.

        Args:
            model: The NNX module to evaluate.
            inputs: The batch inputs.
            targets: The batch targets.
            sample_weight: Per-sample weights for the batch.
            key: A JAX PRNG key, e.g. for stochastic layers.
            training: Whether the loss is being computed during training
                (as opposed to evaluation).

        Returns:
            The scalar loss value.
        """
        ...


def _metric_mean(values: Iterable[float]) -> float:
    """Compute the arithmetic mean of an iterable of metric values.

    Args:
        values: The metric values to average, e.g. per-batch losses.

    Returns:
        The arithmetic mean of ``values``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    values = tuple(values)
    if not values:
        raise ValueError("a loader must provide at least one batch.")
    return sum(values) / len(values)


def make_evaluation_step(loss: SupervisedLoss) -> Callable[..., jax.Array]:
    """Create a JIT-compiled NNX supervised evaluation step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.

    Returns:
        A JIT-compiled function ``evaluate_step(model, inputs, targets,
        sample_weight, key)`` that returns the scalar loss value without
        updating any parameters.
    """

    @nnx.jit
    def evaluate_step(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
    ) -> jax.Array:
        return loss(model, inputs, targets, sample_weight, key, False)

    return evaluate_step


def evaluate_loader(
    model: nnx.Module,
    loader: Iterable[Batch],
    *,
    key: jax.Array,
    evaluation_step: Callable[..., jax.Array],
) -> tuple[dict[str, float], jax.Array]:
    """Evaluate a loader and return mean loss plus the advanced random key.

    Args:
        model: The NNX module to evaluate.
        loader: An iterable of batches to evaluate.
        key: The JAX PRNG key to use, advanced once per batch.
        evaluation_step: The JIT-compiled evaluation step, e.g. one
            created by :func:`make_evaluation_step`.

    Returns:
        A tuple ``(metrics, key)`` where ``metrics`` maps ``"loss"`` to
        the mean loss over ``loader`` and ``key`` is the PRNG key
        advanced past all consumed batches.
    """
    losses: list[float] = []
    for batch in loader:
        key, batch_key = split_key(key)
        losses.append(
            float(
                evaluation_step(
                    model,
                    batch.inputs,
                    batch.targets,
                    batch.sample_weight,
                    batch_key,
                )
            )
        )
    return {"loss": _metric_mean(losses)}, key
