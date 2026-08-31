"""Optional JAX/Flax NNX training backend."""

from probreg.jax.evaluation import (
    SupervisedLoss,
    evaluate_loader,
    make_evaluation_step,
)
from probreg.jax.rng import split_key
from probreg.jax.state import (
    NnxSnapshot,
    create_optimizer,
    freeze_training_state,
    initialize_training_state,
    snapshot,
)
from probreg.jax.supervised import make_train_step, run_supervised
from probreg.jax.validation import HeldOutValidation

__all__ = [
    "HeldOutValidation",
    "NnxSnapshot",
    "SupervisedLoss",
    "create_optimizer",
    "evaluate_loader",
    "freeze_training_state",
    "initialize_training_state",
    "make_evaluation_step",
    "make_train_step",
    "run_supervised",
    "snapshot",
    "split_key",
]
