"""Backend-neutral contracts and utilities for probabilistic regression."""

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.distributions import (
    DistributionHead,
    Likelihood,
    Loss,
    PredictiveDistribution,
)
from probreg.core.early_stopping import (
    EarlyStopper,
    EarlyStoppingDecision,
    EarlyStoppingState,
    MetricSource,
    OptimizationMode,
)
from probreg.core.metrics import cdf, coverage, crps, point_crps, rmse, wsu
from probreg.core.metric_registry import (
    ContinuousRankedProbabilityScore,
    EpochMetric,
    EpochPredictionData,
    EvaluationGrid,
    IntervalCoverage,
    MetricRequirements,
    PointContinuousRankedProbabilityScore,
    PredictionInterval,
    RootMeanSquaredError,
    WeightedSpread,
)
from probreg.core.protocols import (
    Dataset,
    LoaderFactory,
    Optimizer,
    Step,
    ValidationStrategy,
)
from probreg.core.stages import StageState, TrainingStage, validate_transition
from probreg.core.tracking import EventSink, ExperimentTracker, TrainingEvent
from probreg.core.types import (
    Array,
    Batch,
    CheckpointRef,
    ParameterRole,
    PyTree,
    StageResult,
    TrainingState,
    ValidationResult,
)

__all__ = [
    "Array",
    "Batch",
    "Checkpoint",
    "CheckpointRef",
    "CheckpointStore",
    "Dataset",
    "DistributionHead",
    "EarlyStopper",
    "EarlyStoppingDecision",
    "EarlyStoppingState",
    "EventSink",
    "ExperimentTracker",
    "Likelihood",
    "LoaderFactory",
    "Loss",
    "MetricSource",
    "OptimizationMode",
    "Optimizer",
    "ParameterRole",
    "PredictiveDistribution",
    "PyTree",
    "StageResult",
    "StageState",
    "Step",
    "TrainingEvent",
    "TrainingStage",
    "TrainingState",
    "ValidationResult",
    "ValidationStrategy",
    "ContinuousRankedProbabilityScore",
    "EpochMetric",
    "EpochPredictionData",
    "EvaluationGrid",
    "IntervalCoverage",
    "MetricRequirements",
    "PointContinuousRankedProbabilityScore",
    "PredictionInterval",
    "RootMeanSquaredError",
    "WeightedSpread",
    "cdf",
    "coverage",
    "crps",
    "point_crps",
    "rmse",
    "validate_transition",
    "wsu",
]
