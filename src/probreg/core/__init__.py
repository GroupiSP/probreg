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
from probreg.core.losses import BetaNLLLoss, GammaResidualNLLLoss, GaussianNLLLoss
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
from probreg.core.metrics import cdf, coverage, crps, point_crps, rmse, wsu
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
    "BetaNLLLoss",
    "Checkpoint",
    "CheckpointRef",
    "CheckpointStore",
    "ContinuousRankedProbabilityScore",
    "Dataset",
    "DistributionHead",
    "EarlyStopper",
    "EarlyStoppingDecision",
    "EarlyStoppingState",
    "EpochMetric",
    "EpochPredictionData",
    "EvaluationGrid",
    "EventSink",
    "ExperimentTracker",
    "GammaResidualNLLLoss",
    "GaussianNLLLoss",
    "IntervalCoverage",
    "Likelihood",
    "LoaderFactory",
    "Loss",
    "MetricRequirements",
    "MetricSource",
    "OptimizationMode",
    "Optimizer",
    "ParameterRole",
    "PointContinuousRankedProbabilityScore",
    "PredictionInterval",
    "PredictiveDistribution",
    "PyTree",
    "RootMeanSquaredError",
    "StageResult",
    "StageState",
    "Step",
    "TrainingEvent",
    "TrainingStage",
    "TrainingState",
    "ValidationResult",
    "ValidationStrategy",
    "WeightedSpread",
    "cdf",
    "coverage",
    "crps",
    "point_crps",
    "rmse",
    "validate_transition",
    "wsu",
]
