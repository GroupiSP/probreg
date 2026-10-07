# Examples

The runnable examples live under
[`examples/jax/`](https://github.com/GroupiSP/probreg/tree/main/examples/jax) in
the repository, so run the commands below from a clone of it. They are ordered
from the gentlest upward: start with the first and move down as you need more
of the library. Each one brings its own dependencies, either through a project
extra (`jax`, `plot`) or through an example-only dependency group.

## Simple regression

`run_supervised` end to end on a toy linear dataset: one model, held-out
validation, early stopping on the validation loss, an in-memory checkpoint
store keeping the best model, and an event sink that prints each epoch.

```bash
uv run --extra jax python examples/jax/simple_regression.py
```

Source:
[`examples/jax/simple_regression.py`](https://github.com/GroupiSP/probreg/blob/main/examples/jax/simple_regression.py)

## Mean-variance regression

The same loop with a predictive distribution: a Gaussian head trained with a
negative log-likelihood loss on data whose noise grows with the input, scored
with RMSE and point-CRPS epoch metrics, and a plot of the predictive interval.

```bash
uv run --extra jax --extra plot python examples/jax/mve_regression.py
```

Source:
[`examples/jax/mve_regression.py`](https://github.com/GroupiSP/probreg/blob/main/examples/jax/mve_regression.py)

## XSin benchmark

Two paired scripts on an XSin-inspired benchmark that compare joint Gaussian
mean-variance estimation with two-step training: a mean stage followed by a
Gamma variance stage fitted on its residuals.

```bash
uv run --extra jax --extra plot python examples/jax/xsin/mve.py
uv run --extra jax --extra plot python examples/jax/xsin/two_steps.py
```

Source:
[`examples/jax/xsin/`](https://github.com/GroupiSP/probreg/tree/main/examples/jax/xsin)

## CMAPSS remaining useful life

Two-step probabilistic remaining-useful-life estimation on NASA CMAPSS FD001: a
CNN mean stage, a Gamma variance stage on its squared residuals, and a
composite Gaussian scored against the official test split.

```bash
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

The first run downloads the CMAPSS archive from NASA and caches it on disk, so
it needs network access once. The README explains how to point the example at
a copy you already have.

README:
[`examples/jax/cmapss/README.md`](https://github.com/GroupiSP/probreg/blob/main/examples/jax/cmapss/README.md)

## Tracking a run

The mean-variance regression again, with the whole run recorded to TensorBoard
through `TrackerEventSink`: per-epoch metrics under their metric tags, the
hyperparameters under their parameter paths, and a final figure.

```bash
uv run --group example-tracking python examples/jax/tracking/run.py --logdir runs/
uv run --group example-tracking tensorboard --logdir runs/
```

README:
[`examples/jax/tracking/README.md`](https://github.com/GroupiSP/probreg/blob/main/examples/jax/tracking/README.md)
