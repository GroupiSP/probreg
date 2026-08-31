# probreg

`probreg` is a stage-oriented library for probabilistic regression.

The initial release establishes backend-neutral contracts for batches, datasets,
optimizers, staged training, predictive distributions, likelihoods, loss
functions, checkpoints, tracking, numerical metrics, and early stopping.

The optional JAX backend supplies Flax NNX/Optax state and RNG adapters plus a
single-model supervised runner. It supports fixed-epoch training without
validation, early stopping on explicit training metrics, and caller-injected
validation strategies. `HeldOutValidation` is provided for the conventional
validation-loader case; callers can inject fold, rolling, grouped, bootstrap,
or external validation without embedding its control flow in the runner.

## Development

Install the development dependencies and run the core tests:

```bash
uv sync --group dev
uv run pytest tests/core -v
uv run ruff check src tests
```

## JAX backend

Install the optional JAX backend, then include its focused tests:

```bash
uv sync --extra jax --group dev
uv run pytest tests/core tests/jax -v
```

When no validation strategy or early stopper is passed to `run_supervised`, it
runs for the requested number of epochs. An early stopper monitoring validation
metrics requires a validation strategy; use a training-metric source explicitly
for convergence-driven training without validation.
