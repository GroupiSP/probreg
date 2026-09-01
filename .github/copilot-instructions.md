# Copilot instructions for Probreg

## Project overview

Probreg is a Python library for probabilistic regression. It is structured in stages, in order to allow training of the probabilistic models in steps. For example, a first stage might involve fitting a mean predicting model, and a second stage would instead learn the data variance, using the already trained mean predictor.

The library has been inspired by the work of [Yi and Bessa, 2025](https://arxiv.org/abs/2505.02743), which proposes the step-wise training logic for avoiding gradient pathologies of mean-variance estimators (MVEs) and iterative training of variance predictor and Bayesian Neural Networks (BNNs) to disentangle aleatoric and epistemic uncertainty.

## Current status

`src/probreg/core/` provides the backend-neutral foundation: `types.py`
(`Batch`, `TrainingState`, `StageResult`, etc.), `protocols.py`
(`Dataset`, `LoaderFactory`, `Optimizer`, `Step`), `distributions.py`,
`stages.py`, `checkpoints.py`, `tracking.py`, `metrics.py`, and
`early_stopping.py` (backend-neutral `EarlyStopper`/`EarlyStoppingState`).
None of these modules import JAX, Flax, or any other training backend.

`src/probreg/jax/` is an optional JAX/Flax NNX training backend (installed
via the `jax` extra) implementing a single-model supervised training runner,
`run_supervised`, on top of the core contracts: NNX/Optax state adapters
(`state.py`), RNG splitting (`rng.py`), shared evaluation primitives
(`evaluation.py`), a held-out `ValidationStrategy` (`validation.py`), and the
runner itself (`supervised.py`). It supports fixed-epoch training, optional
caller-injected validation strategies, early stopping, best-model
checkpointing, and event-sink notifications.

`examples/simple_regression_jax.py` demonstrates the full JAX-backend stack
end to end on a synthetic linear-regression dataset.

Not yet implemented: the distribution-valued Gaussian head, MVE and staged
(Step 1/Step 2/VeBNN Step 3) training stages, and any qmodem integration
adapter. The qmodem adapter is intentionally out of scope for this
repository — it belongs downstream in `qmodem`, which depends on `probreg`.
