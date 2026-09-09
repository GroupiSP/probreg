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

It also supplies explicit two-step training for disentangling a deterministic
mean estimate from input-dependent aleatoric variance:

```text
NEW -> INITIALIZED -> MEAN_READY -> VARIANCE_READY
```

`MeanStage` trains a mean-only model with MSE and restores its selected best
checkpoint after early stopping. The completed checkpoint is marked
`MEAN_READY` so it can resume directly into `GammaVarianceStage`, which clones
the restored mean model in inference mode, materializes detached squared
residual targets once, and trains a separate Gamma shape/rate model. The mean
model is recorded as frozen and is never passed to the variance optimizer. The
aleatoric variance estimate is the Gamma mean,
`concentration / rate`; the Gamma distribution's own variance is
`concentration / rate**2`.

Residual materialization is intentionally an in-memory snapshot: configured
loader splits are consumed once at a caller-selected source epoch and replayed
unchanged during variance training. Dynamic residuals for cooperative outer
iterations are outside this initial two-step implementation.

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

### Epoch metrics

Host-side epoch metrics consume `EpochPredictionData`, a typed representation of
materialized scalar targets, moments, sample matrices, labelled intervals, and an
optional coordinate. Distribution objects remain the generic model/loss boundary;
a JAX predictor such as `GaussianPredictor` explicitly transfers only the fields
required by the registered metrics to the host. This keeps epoch merging independent
of a distribution backend and avoids retaining device-backed distribution objects.

Configure sample/grid-based metrics explicitly:

```python
import numpy as np

from probreg.core import (
    EvaluationGrid,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
)
from probreg.jax import GaussianPredictor, MetricSuite

metrics = MetricSuite(
    epoch=(RootMeanSquaredError(), PointContinuousRankedProbabilityScore()),
    predictor=GaussianPredictor(),
    predictive_sample_count=128,
    evaluation_grid=EvaluationGrid(np.linspace(-10.0, 10.0, 401)),
)
```

`PointContinuousRankedProbabilityScore` compares each scalar target with that
unit's predictive draws. `ContinuousRankedProbabilityScore` instead averages the
predictive CRPS over each unit's empirical reference distribution and therefore
requires an explicit `reference_samples_extractor` on the predictor. Both score each
independent scalar unit and then average; non-scalar event distributions are rejected.
The shared grid must be finite and strictly increasing. Predictive sample storage costs
`O(n_scoring_units * predictive_sample_count)` host memory.

Intervals are selected by their declared confidence level. `WeightedSpread` also
requires an explicit numeric coordinate extractor at the JAX/domain boundary; generic
batch metadata never enters the core epoch metric contract. Metric sample keys are
derived in a separate namespace, so registering sampled epoch metrics does not alter
the loss or batch-metric RNG trajectory. Prediction adapters run an inference-mode
clone of the model, so stateful layers and internal RNG streams are not mutated by
metric collection.

## Examples

The runnable examples live under `examples/jax/`. Each one carries its own
run command in its module docstring; they are collected here as well.

| Example | What it shows |
| --- | --- |
| [`examples/jax/simple_regression.py`](examples/jax/simple_regression.py) | `run_supervised` end to end on a toy linear dataset, with held-out validation, early stopping, an in-memory checkpoint store, and a printing event sink |
| [`examples/jax/mve_regression.py`](examples/jax/mve_regression.py) | Joint Gaussian mean-variance estimation on a heteroscedastic dataset, with RMSE and point-CRPS epoch metrics |
| [`examples/jax/xsin/mve.py`](examples/jax/xsin/mve.py) | Joint Gaussian MVE on the XSin-inspired benchmark |
| [`examples/jax/xsin/two_steps.py`](examples/jax/xsin/two_steps.py) | Explicit mean-then-Gamma training on the same benchmark, for comparison |
| [`examples/jax/cmapss/run.py`](examples/jax/cmapss/run.py) | Two-stage probabilistic RUL estimation on NASA CMAPSS FD001, evaluated against the official test split |
| [`examples/jax/cmapss/data.py`](examples/jax/cmapss/data.py) | Loading, caching, and plotting the CMAPSS FD001 sensor trajectories |
| [`examples/jax/tracking/run.py`](examples/jax/tracking/run.py) | The MVE example again, with the whole run tracked to TensorBoard through `TrackerEventSink` ([README](examples/jax/tracking/README.md)) |

`examples/jax/xsin/benchmark.py`, `examples/jax/cmapss/model.py`,
`examples/jax/cmapss/preprocessing.py`, `examples/jax/cmapss/plots.py`, and
`examples/jax/tracking/tensorboard_tracker.py` are shared modules imported by
the examples above rather than scripts to run.

Examples that visualize predictions need the optional `plot` extra
(Matplotlib); the CMAPSS and tracking examples have their own dependency
groups:

```bash
uv sync --extra jax --extra plot --group dev
uv run --extra jax python examples/jax/simple_regression.py
uv run --extra jax --extra plot python examples/jax/mve_regression.py
uv run --group example-cmapss python examples/jax/cmapss/run.py
uv run --group example-tracking python examples/jax/tracking/run.py
```

### Tracked training with TensorBoard

`examples/jax/tracking/` trains the MVE example's model and records the run to
TensorBoard: per-epoch training and validation scalars under stage-namespaced
tags, the hyperparameters and the loss and metric identities in the HParams
table, and a final figure of the mean with its 95% predictive interval. The
example's `TensorBoardTracker` is the only TensorBoard-specific code; the
library depends on no tracker, and the bridge from training events to a tracker
is `probreg.core.tracking.TrackerEventSink`:

```bash
uv run --group example-tracking python examples/jax/tracking/run.py --logdir runs/
uv run --group example-tracking tensorboard --logdir runs/
```

Each invocation writes to its own UTC-timestamped subdirectory, so successive
runs appear side by side. See
[`examples/jax/tracking/README.md`](examples/jax/tracking/README.md) for
details.

### XSin-inspired comparison

Two paired examples compare joint Gaussian MVE with explicit mean-then-Gamma
training on the same nonlinear XSin-inspired heteroscedastic dataset. Following
the paper's qualitative setup, models train only on `x in (0, 10)` and are
evaluated on the wider `x in (-5, 15)` range to expose both interpolation and
extrapolation behavior:

```bash
uv run --extra jax --extra plot python examples/jax/xsin/mve.py
uv run --extra jax --extra plot python examples/jax/xsin/two_steps.py
```

Both scripts print overall, interpolation, and extrapolation errors against the
known data-generating functions and use the same interactive plot layout.
Training-domain boundaries and extrapolation regions are marked explicitly.
Under the seeded example configuration, the two-step method improves both
interpolation estimates by preventing the variance objective from distorting
the already-trained mean; extrapolation metrics are reported rather than
assumed to improve.

The examples reproduce the qualitative XSin comparison motivated by Yi and
Bessa (2025), with compact settings suitable for a library demonstration. They
do not claim exact reproduction of the paper's architectures, runtime, or
reported numerical values.
