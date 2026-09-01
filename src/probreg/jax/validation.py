"""JAX validation strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
from flax import nnx

from probreg.core.protocols import LoaderFactory
from probreg.core.types import TrainingState, ValidationResult
from probreg.jax.evaluation import (
    SupervisedLoss,
    evaluate_loader,
    make_evaluation_step,
)
from probreg.jax.metrics import MetricSuite


@dataclass(frozen=True)
class HeldOutValidation:
    """Evaluate an NNX model against a held-out validation loader.

    Attributes:
        model: The NNX module to evaluate.
        loader: Factory producing the validation batch loader for a
            given split and epoch.
        loss: The supervised loss function used for evaluation.
        metrics: Registered batch/epoch metrics for validation.
        metric_prefix: Prefix applied to metric names before returning
            them. Defaults to ``"validation_"``.
        metadata: Arbitrary caller-supplied metadata attached to this
            strategy.
    """

    model: nnx.Module
    loader: LoaderFactory
    loss: SupervisedLoss
    metrics: MetricSuite = MetricSuite()
    metric_prefix: str = "validation_"
    metadata: dict[str, object] = field(default_factory=dict)

    def __call__(self, state: TrainingState, *, epoch: int) -> ValidationResult:
        """Evaluate the held-out split after a training epoch.

        Args:
            state: The current training state, whose ``rng_state`` is
                advanced in place as validation batches are consumed.
            epoch: The epoch just completed, passed to ``loader`` to
                select the validation batches for this epoch.

        Returns:
            A :class:`ValidationResult` with prefixed metric names and
            ``passed=True``.

        Raises:
            TypeError: If ``state.rng_state`` is not a JAX random key.
        """
        if not isinstance(state.rng_state, jax.Array):
            raise TypeError("state.rng_state must be a JAX random key.")
        metrics, state.rng_state = evaluate_loader(
            self.model,
            self.loader(split="validation", epoch=epoch),
            key=state.rng_state,
            evaluation_step=make_evaluation_step(self.loss, metrics=self.metrics.batch),
            metrics=self.metrics,
        )
        return ValidationResult(
            passed=True,
            metrics={
                f"{self.metric_prefix}{name}": value for name, value in metrics.items()
            },
            message=None,
        )
