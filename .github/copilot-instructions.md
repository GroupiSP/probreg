# Copilot instructions for Probreg

## Project overview

Probreg is a Python library for stage-oriented probabilistic regression. Its
architecture represents training as explicit, composable stages so that model
components can be fitted in a controlled sequence. A typical workflow first
trains a mean predictor, then estimates input-dependent variance while reusing
or freezing the trained mean, and can subsequently add a posterior model for
epistemic uncertainty.

The library is inspired by [Yi and Bessa, 2025](https://arxiv.org/abs/2505.02743),
which proposes step-wise training to avoid gradient pathologies in
mean-variance estimators (MVEs), followed by iterative variance-predictor and
Bayesian neural network (BNN) training to disentangle aleatoric and epistemic
uncertainty. It also builds on
[Sluijterman, Cator, and Heskes, 2023](https://arxiv.org/abs/2302.08875),
which studies optimal MVE training and recommends a mean-only warm-up before
variance optimization, with separate regularization of the mean and variance
subnetworks. The implemented MVE objectives additionally follow Gaussian
negative log-likelihood (Nix and Weigend, 1994) and beta-NLL (Seitzer et al.,
2022).

## Project structure

- `src/probreg/core/` contains the backend-neutral public contracts and NumPy
  utilities: shared value objects, stage lifecycle protocols, predictive
  distribution and loss interfaces, checkpointing, tracking, early stopping,
  numerical metrics, and typed epoch-metric registration.
- `src/probreg/jax/` contains the optional JAX backend, built with JAX, Flax NNX,
  and Optax. It provides model and optimizer state adapters, RNG management,
  Gaussian distributions and heads, supervised training and evaluation,
  validation strategies, supervised loss adapters, and metric collection.
- `tests/core/` and `tests/jax/` mirror the source layout with pytest tests for
  the backend-neutral foundation and optional JAX integration, respectively.
- `examples/` contains runnable synthetic-regression examples for deterministic
  supervised training and heteroscedastic Gaussian MVE training.
- `.github/` contains CI, issue templates, contributor instructions, custom
  agents, and project skills. Repository-wide development conventions are also
  documented in `AGENTS.md` and `CONTRIBUTING.md`.
- `pyproject.toml` defines the Python 3.12 package, the minimal NumPy dependency,
  the optional `jax` and `plot` extras, and the development toolchain. The
  resolved environment is recorded in `uv.lock`.

## Current status

`src/probreg/core/` provides the backend-neutral foundation: `types.py`
(`Batch`, `TrainingState`, `StageResult`, etc.), `protocols.py`
(`Dataset`, `LoaderFactory`, `Optimizer`, `Step`), `distributions.py`,
`losses.py`, `stages.py`, `checkpoints.py`, `tracking.py`, `metrics.py`,
`metric_registry.py`, and `early_stopping.py`. These modules define the stage
lifecycle, Gaussian NLL and beta-NLL objectives, persistence and event
interfaces, early-stopping state machines, NumPy metric functions, and typed
host-side epoch metrics without importing JAX, Flax, or another training
backend.

`src/probreg/jax/` is an optional JAX/Flax NNX training backend (installed
via the `jax` extra) implementing a single-model supervised training runner,
`run_supervised`, on top of the core contracts: NNX/Optax state adapters
(`state.py`), RNG splitting (`rng.py`), shared evaluation primitives
(`evaluation.py`), a held-out `ValidationStrategy` (`validation.py`), and the
runner itself (`supervised.py`). The runner supports fixed-epoch training,
caller-injected validation strategies, early stopping on training or validation
metrics, best-model checkpointing, event-sink notifications, and composable
batch and epoch metric registration.

The metric stack spans `core/metric_registry.py` and `jax/metrics.py`.
Backend-neutral `EpochPredictionData` materializes scalar targets, moments,
sample matrices, labelled intervals, coordinates, and evaluation grids on the
host. `MetricSuite` combines JAX-native batch metrics with host-side epoch
metrics such as RMSE, interval coverage, weighted spread, point CRPS, and CRPS.
`GaussianPredictor` transfers only the fields required by registered metrics,
uses a dedicated metric RNG namespace, and evaluates an inference-mode model
clone so metric collection does not mutate training state or alter the loss RNG
trajectory.

For MVE, `jax/distributions.py` implements the concrete JAX-backed `Gaussian`
and `GaussianHead`, with an explicit positive scale, while
`jax/losses.py` provides `make_supervised_loss` to adapt core per-example
objectives to `SupervisedLoss`. This lets distribution-valued MVE models reuse
`run_supervised` without coupling the core loss definitions to JAX.

`examples/simple_regression_jax.py` demonstrates the supervised JAX stack on a
synthetic linear-regression dataset.
`examples/mve_regression_jax.py` demonstrates MVE end to end on synthetic
heteroscedastic data, including a `plot_predictions` helper (behind the
optional `plot` extra) and registered probabilistic epoch metrics.

The backend-neutral stage states, transition validation, shared training state,
and `TrainingStage` protocol are implemented, but concrete orchestration for
the staged mean-training, variance-training, and VeBNN posterior-training
workflow is not yet implemented. There is also no ensemble or BNN backend yet.
Any qmodem integration adapter remains intentionally out of scope for this
repository; it belongs downstream in `qmodem`, which depends on `probreg`.
