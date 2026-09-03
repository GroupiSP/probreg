"""Optional JAX/Flax NNX training backend."""

from probreg.jax.distributions import Gamma, GammaHead, Gaussian, GaussianHead
from probreg.jax.evaluation import (
    SupervisedLoss,
    evaluate_loader,
    make_evaluation_step,
)
from probreg.jax.losses import make_supervised_loss
from probreg.jax.metrics import (
    BatchMetric,
    BatchMetricSpec,
    CoordinateExtractor,
    GaussianPredictor,
    MetricSuite,
    PredictionRequirements,
    Predictor,
    ReferenceSamplesExtractor,
    merge_epoch_prediction_data,
)
from probreg.jax.mve import make_mve_loss
from probreg.jax.rng import split_key
from probreg.jax.state import (
    NnxSnapshot,
    create_optimizer,
    freeze_training_state,
    initialize_training_state,
    snapshot,
)
from probreg.jax.supervised import make_train_step, run_supervised
from probreg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
    materialize_residual_loader,
)
from probreg.jax.validation import HeldOutValidation

__all__ = [
    "BatchMetric",
    "BatchMetricSpec",
    "CoordinateExtractor",
    "Gamma",
    "GammaHead",
    "GammaVarianceStage",
    "Gaussian",
    "GaussianHead",
    "GaussianPredictor",
    "HeldOutValidation",
    "MeanStage",
    "MetricSuite",
    "NnxSnapshot",
    "PredictionRequirements",
    "Predictor",
    "ReferenceSamplesExtractor",
    "SupervisedLoss",
    "SupervisedStageOptions",
    "create_optimizer",
    "evaluate_loader",
    "freeze_training_state",
    "initialize_training_state",
    "make_evaluation_step",
    "make_mve_loss",
    "make_supervised_loss",
    "make_train_step",
    "materialize_residual_loader",
    "merge_epoch_prediction_data",
    "run_supervised",
    "snapshot",
    "split_key",
]
