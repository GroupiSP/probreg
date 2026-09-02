"""Optional JAX/Flax NNX training backend."""

from probreg.jax.distributions import Gamma, GammaHead, Gaussian, GaussianHead
from probreg.jax.evaluation import (
    SupervisedLoss,
    evaluate_loader,
    make_evaluation_step,
)
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
from probreg.jax.two_step import (
    make_gamma_residual_loss,
    make_mean_squared_error_loss,
)
from probreg.jax.validation import HeldOutValidation

__all__ = [
    "BatchMetric",
    "BatchMetricSpec",
    "CoordinateExtractor",
    "Gamma",
    "GammaHead",
    "Gaussian",
    "GaussianHead",
    "GaussianPredictor",
    "HeldOutValidation",
    "MetricSuite",
    "NnxSnapshot",
    "PredictionRequirements",
    "Predictor",
    "ReferenceSamplesExtractor",
    "SupervisedLoss",
    "create_optimizer",
    "evaluate_loader",
    "freeze_training_state",
    "initialize_training_state",
    "make_evaluation_step",
    "make_gamma_residual_loss",
    "make_mean_squared_error_loss",
    "make_mve_loss",
    "make_train_step",
    "merge_epoch_prediction_data",
    "run_supervised",
    "snapshot",
    "split_key",
]
