"""Backend-neutral contracts and utilities for probabilistic regression."""

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.distributions import (
    DistributionHead,
    Likelihood,
    Loss,
    PredictiveDistribution,
)
from probreg.core.metrics import cdf, coverage, crps, point_crps, rmse, wsu
from probreg.core.protocols import Dataset, LoaderFactory, Optimizer, Step
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
    "EventSink",
    "ExperimentTracker",
    "Likelihood",
    "LoaderFactory",
    "Loss",
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
    "cdf",
    "coverage",
    "crps",
    "point_crps",
    "rmse",
    "validate_transition",
    "wsu",
]
