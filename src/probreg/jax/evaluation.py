"""Shared NNX evaluation primitives used by training runners and validation strategies.

This module has no dependency on any specific training runner, so both
:mod:`probreg.jax.supervised` and :mod:`probreg.jax.validation` may depend
on it without either depending on the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

import jax
from flax import nnx

from probreg.core.types import Batch
from probreg.jax.metrics import (
    BatchMetricSpec,
    MetricSuite,
    _finite_float,
    _metric_mean,
    merge_metric_inputs,
    resolve_metric_inputs,
)
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


def make_evaluation_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX supervised evaluation step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.
        metrics: Registered JAX-native batch metrics.

    Returns:
        A JIT-compiled function ``evaluate_step(model, inputs, targets,
        sample_weight, key)`` returning a mapping containing ``"loss"``
        plus one scalar value per registered batch metric.
    """

    @nnx.jit
    def evaluate_step(
        model: nnx.Module,
        inputs: Any,
        targets: Any,
        sample_weight: Any,
        key: jax.Array,
    ) -> Mapping[str, jax.Array]:
        loss_value = loss(model, inputs, targets, sample_weight, key, False)
        values: dict[str, jax.Array] = {"loss": loss_value}
        for spec in metrics:
            values[spec.name] = spec.metric(
                model,
                inputs,
                targets,
                sample_weight,
                key,
                False,
            )
        return values

    return evaluate_step


def evaluate_loader(
    model: nnx.Module,
    loader: Iterable[Batch],
    *,
    key: jax.Array,
    evaluation_step: Callable[..., Mapping[str, jax.Array]],
    metrics: MetricSuite = MetricSuite(),
) -> tuple[dict[str, float], jax.Array]:
    """Evaluate a loader and return reduced metrics and the advanced random key.

    Args:
        model: The NNX module to evaluate.
        loader: An iterable of batches to evaluate.
        key: The JAX PRNG key to use, advanced once per batch.
        evaluation_step: The JIT-compiled evaluation step, e.g. one
            created by :func:`make_evaluation_step`.
        metrics: Registered batch/epoch metrics for evaluation.

    Returns:
        A tuple ``(metrics, key)`` where ``metrics`` contains ``"loss"``
        plus any registered metrics, and ``key`` is advanced past all
        consumed batches.
    """
    losses: list[float] = []
    batch_metric_values: dict[str, list[float]] = {
        spec.name: [] for spec in metrics.batch
    }
    epoch_metric_parts = [] if metrics.epoch else None

    for batch in loader:
        key, batch_key = split_key(key)
        step_output = evaluation_step(
            model,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            batch_key,
        )
        losses.append(float(step_output["loss"]))
        for spec in metrics.batch:
            if spec.name not in step_output:
                raise ValueError(
                    f"evaluation step did not produce registered metric {spec.name!r}."
                )
            batch_metric_values[spec.name].append(float(step_output[spec.name]))
        if epoch_metric_parts is not None:
            epoch_metric_parts.append(
                resolve_metric_inputs(suite=metrics, model=model, batch=batch)
            )

    reduced: dict[str, float] = {"loss": _finite_float("loss", _metric_mean(losses))}
    for spec in metrics.batch:
        reduced[spec.name] = _finite_float(
            spec.name,
            spec.reduce(batch_metric_values[spec.name]),
        )
    if epoch_metric_parts is not None:
        merged = merge_metric_inputs(epoch_metric_parts)
        for epoch_metric in metrics.epoch:
            reduced[epoch_metric.name] = _finite_float(
                epoch_metric.name,
                float(epoch_metric(merged)),
            )
    return reduced, key
