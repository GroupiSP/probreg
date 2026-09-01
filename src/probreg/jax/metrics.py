"""Metric registration primitives for the JAX backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import numpy as np
from flax import nnx
from numpy.typing import NDArray

from probreg.core.metric_registry import EpochMetric, MetricInputs, SupportsMetricInputs
from probreg.core.types import Batch


def _metric_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of metric values.

    Args:
        values: Metric values to average.

    Returns:
        The arithmetic mean.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("a loader must provide at least one batch.")
    return sum(values) / len(values)


class BatchMetric(Protocol):
    """A JAX-native metric computed per batch."""

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
        """Return a scalar batch metric value."""
        ...


@dataclass(frozen=True, slots=True)
class BatchMetricSpec:
    """A named batch metric with a host-side reduction policy."""

    name: str
    metric: BatchMetric
    reduce: Callable[[Sequence[float]], float] = _metric_mean

    def __post_init__(self) -> None:
        """Validate a batch metric registration.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not self.name:
            raise ValueError("batch metric name must not be empty.")


class Predictor(Protocol):
    """Override for producing epoch metric inputs from a model and batch."""

    def __call__(self, model: nnx.Module, batch: Batch, /) -> MetricInputs:
        """Produce epoch metric inputs for ``batch``."""
        ...


@dataclass(frozen=True, slots=True)
class MetricSuite:
    """A complete metric registration suite for training or validation."""

    batch: tuple[BatchMetricSpec, ...] = ()
    epoch: tuple[EpochMetric, ...] = ()
    predictor: Predictor | None = None

    def __post_init__(self) -> None:
        """Validate registered metric names.

        Raises:
            ValueError: If names are duplicated or ``"loss"`` is reused.
        """
        names = ["loss"]
        names.extend(spec.name for spec in self.batch)
        names.extend(metric.name for metric in self.epoch)
        duplicate_names = {
            name for name in names if names.count(name) > 1 and name != "loss"
        }
        if "loss" in [spec.name for spec in self.batch] or "loss" in [
            metric.name for metric in self.epoch
        ]:
            raise ValueError("'loss' is reserved and cannot be registered as a metric.")
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"duplicate metric names are not allowed: {duplicates}.")


def resolve_metric_inputs(
    *,
    suite: MetricSuite,
    model: nnx.Module,
    batch: Batch,
) -> MetricInputs:
    """Resolve epoch metric inputs using suite override or model protocol.

    Args:
        suite: The metric suite with optional predictor override.
        model: The model being evaluated.
        batch: The current batch.

    Returns:
        Materialized metric inputs for ``batch``.

    Raises:
        ValueError: If no predictor is available for non-empty epoch metrics.
    """
    if suite.predictor is not None:
        return suite.predictor(model, batch)
    if isinstance(model, SupportsMetricInputs):
        return model.produce_metric_inputs(batch)
    raise ValueError(
        "epoch metrics require either suite.predictor or a model implementing "
        "SupportsMetricInputs.produce_metric_inputs."
    )


def merge_metric_inputs(parts: Sequence[MetricInputs]) -> MetricInputs:
    """Concatenate per-batch metric inputs into one epoch input object.

    Args:
        parts: Per-batch metric inputs.

    Returns:
        A merged :class:`MetricInputs` object.

    Raises:
        ValueError: If ``parts`` is empty or optional fields are inconsistently present.
    """
    if not parts:
        raise ValueError("at least one metric input batch is required.")
    return MetricInputs(
        targets=_concat_required([part.targets for part in parts], name="targets"),
        mean=_concat_required([part.mean for part in parts], name="mean"),
        variance=_concat_optional([part.variance for part in parts], name="variance"),
        lower=_concat_optional([part.lower for part in parts], name="lower"),
        upper=_concat_optional([part.upper for part in parts], name="upper"),
        metadata=_merge_metadata([part.metadata for part in parts]),
    )


def _to_vector(values: Any, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return vector


def _concat_required(values: Sequence[Any], *, name: str) -> NDArray[np.float64]:
    return np.concatenate([_to_vector(value, name=name) for value in values])


def _concat_optional(
    values: Sequence[Any | None], *, name: str
) -> NDArray[np.float64] | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} must be present for all batches or none.")
    return np.concatenate([_to_vector(value, name=name) for value in values])  # type: ignore[arg-type]


def _merge_metadata(mappings: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not mappings:
        return {}
    keys: set[str] = set().union(*(mapping.keys() for mapping in mappings))
    merged: dict[str, Any] = {}
    for key in keys:
        if not all(key in mapping for mapping in mappings):
            raise ValueError(f"metadata[{key!r}] must be present for all batches or none.")
        merged[key] = np.concatenate(
            [_to_vector(mapping[key], name=f"metadata[{key!r}]") for mapping in mappings]
        )
    return merged
