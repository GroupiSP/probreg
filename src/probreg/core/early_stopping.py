"""Backend-neutral early-stopping policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class OptimizationMode(StrEnum):
    """The direction in which a monitored metric is considered improved."""

    MIN = "min"
    MAX = "max"


class MetricSource(StrEnum):
    """The origin of a metric monitored by early stopping."""

    TRAINING = "training"
    VALIDATION = "validation"


@dataclass(frozen=True)
class EarlyStoppingState:
    """Checkpointable state for a configured early-stopping policy.

    Attributes:
        metric: Name of the monitored metric.
        mode: Direction in which the metric is considered improved.
        min_delta: Minimum change required to qualify as an improvement.
        patience: Number of consecutive non-improving epochs tolerated
            before stopping.
        source: Whether the monitored metric comes from training or
            validation.
        best_score: Best value observed so far, or ``None`` if no value
            has been observed yet.
        best_epoch: Epoch at which ``best_score`` was observed, or
            ``None`` if no value has been observed yet.
        non_improving_epochs: Number of consecutive epochs without an
            improvement.
        stopped: Whether patience has been exhausted and training should
            stop.
    """

    metric: str
    mode: OptimizationMode
    min_delta: float
    patience: int
    source: MetricSource = MetricSource.VALIDATION
    best_score: float | None = None
    best_epoch: int | None = None
    non_improving_epochs: int = 0
    stopped: bool = False


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """The result of observing one monitored metric value.

    Attributes:
        state: Updated early-stopping state after the observation.
        improved: Whether the observed value improved on the best score.
    """

    state: EarlyStoppingState
    improved: bool

    @property
    def should_stop(self) -> bool:
        """Whether the configured patience has been exhausted."""
        return self.state.stopped


class EarlyStopper:
    """Tracks a scalar metric and decides when training should stop."""

    def __init__(
        self,
        *,
        metric: str,
        mode: OptimizationMode | str,
        patience: int,
        min_delta: float = 0.0,
        source: MetricSource | str = MetricSource.VALIDATION,
        state: EarlyStoppingState | None = None,
    ) -> None:
        """Initialize an early stopper.

        Args:
            metric: Name of the metric to monitor.
            mode: Direction in which the metric is considered improved,
                either ``"min"`` or ``"max"``.
            patience: Number of consecutive non-improving epochs
                tolerated before stopping.
            min_delta: Minimum change required to qualify as an
                improvement. Defaults to ``0.0``.
            source: Whether the monitored metric comes from training or
                validation. Defaults to ``MetricSource.VALIDATION``.
            state: Previously checkpointed state to resume from. Must
                match the configuration passed to this constructor.

        Raises:
            ValueError: If ``metric`` is empty, ``patience`` is
                negative, ``min_delta`` is not a finite non-negative
                number, or ``state`` does not match the given
                configuration.
        """
        if not metric:
            raise ValueError("metric must be a non-empty string.")
        if patience < 0:
            raise ValueError("patience must be non-negative.")
        if not isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative.")

        normalized_mode = OptimizationMode(mode)
        normalized_source = MetricSource(source)
        self._state = state or EarlyStoppingState(
            metric=metric,
            mode=normalized_mode,
            min_delta=min_delta,
            patience=patience,
            source=normalized_source,
        )
        if (
            self._state.metric != metric
            or self._state.mode != normalized_mode
            or self._state.min_delta != min_delta
            or self._state.patience != patience
            or self._state.source != normalized_source
        ):
            raise ValueError("state does not match the early-stopper configuration.")

    @property
    def state(self) -> EarlyStoppingState:
        """Return the current checkpointable stopping state."""
        return self._state

    def expects_validation(self) -> bool:
        """Whether the monitored metric must come from validation.

        Returns:
            ``True`` if the metric configured for this stopper is
            sourced from validation (i.e. requires a
            :class:`~probreg.core.protocols.ValidationStrategy` to be
            supplied to the training loop), ``False`` if it is sourced
            from training.
        """
        return self._state.source is MetricSource.VALIDATION

    def monitored_metric_name(self) -> str:
        """Return the name of the metric this stopper monitors.

        Returns:
            The metric name passed to the constructor, used by callers
            to look up the corresponding value in per-epoch metrics.
        """
        return self._state.metric

    def observe(self, value: float, *, epoch: int) -> EarlyStoppingDecision:
        """Record a metric observed after ``epoch`` and return its decision.

        Args:
            value: The monitored metric value observed for this epoch.
            epoch: The epoch at which ``value`` was observed.

        Returns:
            An :class:`EarlyStoppingDecision` with the updated state and
            whether ``value`` improved on the best score so far.

        Raises:
            ValueError: If ``epoch`` is negative or ``value`` is not
                finite.
            RuntimeError: If early stopping has already been triggered.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        if not isfinite(value):
            raise ValueError("observed metric value must be finite.")
        if self._state.stopped:
            raise RuntimeError("cannot observe a metric after early stopping.")

        improved = self._is_improvement(value)
        if improved:
            self._state = EarlyStoppingState(
                metric=self._state.metric,
                mode=self._state.mode,
                min_delta=self._state.min_delta,
                patience=self._state.patience,
                source=self._state.source,
                best_score=value,
                best_epoch=epoch,
            )
        else:
            non_improving_epochs = self._state.non_improving_epochs + 1
            self._state = EarlyStoppingState(
                metric=self._state.metric,
                mode=self._state.mode,
                min_delta=self._state.min_delta,
                patience=self._state.patience,
                source=self._state.source,
                best_score=self._state.best_score,
                best_epoch=self._state.best_epoch,
                non_improving_epochs=non_improving_epochs,
                stopped=non_improving_epochs > self._state.patience,
            )
        return EarlyStoppingDecision(state=self._state, improved=improved)

    def _is_improvement(self, value: float) -> bool:
        """Determine whether ``value`` improves on the current best score.

        Args:
            value: The candidate metric value.

        Returns:
            ``True`` if ``value`` is an improvement over the current best
            score given the configured mode and ``min_delta``.
        """
        if self._state.best_score is None:
            return True
        match self._state.mode:
            case OptimizationMode.MIN:
                return value < self._state.best_score - self._state.min_delta
            case OptimizationMode.MAX:
                return value > self._state.best_score + self._state.min_delta
