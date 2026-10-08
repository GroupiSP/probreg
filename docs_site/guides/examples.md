# Examples

The runnable examples live under
[`examples/jax/`](https://github.com/GroupiSP/probreg/tree/main/examples/jax) in
the repository, so run the commands below from a clone of it. They are ordered
from the gentlest upward: start with the first and move down as you need more
of the library. Each one brings its own dependencies, either through a project
extra (`jax`, `plot`) or through an example-only dependency group.

Only the commands shown here are entry points. The other modules beside them,
`xsin/benchmark.py`, `cmapss/model.py`, `cmapss/preprocessing.py`,
`cmapss/plots.py` and `tracking/tensorboard_tracker.py`, hold shared code the
scripts import.

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
Gamma variance stage fitted on its residuals. Both train only on `x` in
`(0, 10)` and are evaluated on the wider `(-5, 15)`, so each run shows
interpolation and extrapolation side by side.

```bash
uv run --extra jax --extra plot python examples/jax/xsin/mve.py
uv run --extra jax --extra plot python examples/jax/xsin/two_steps.py
```

Each script prints its mean and variance errors against the known
data-generating functions, overall, inside the training domain and outside
it, and plots the result with the training-domain boundaries marked. Under the
seeded configuration, two-step training improves both interpolation errors,
because the variance objective can no longer distort the trained mean; the
extrapolation errors are reported, not assumed to improve. The benchmark
reproduces the qualitative comparison motivated by Yi and Bessa (2025) with
compact settings; it does not claim to reproduce the paper's architectures,
runtime or reported values.

A third script adds the optional [posterior stage](posterior-stage.md) after
the two stages, once with Bayes by Backprop and once with pSGLD:

```bash
uv run --extra jax --extra plot python examples/jax/xsin/posterior.py
```

For each inference method it prints the NLL and the CRPS of the variance
stage's Gaussian and of the posterior predictive, scored on one noisy
observation per evaluation point, overall, inside the training domain and
outside it. It plots the moment-matched predictive's 95% band over the variance
stage's band, with the aleatoric and epistemic variance below. Under the
seeded configuration both methods lower the overall NLL and CRPS, mostly
in the extrapolation region, where the variance stage's band is far too narrow.
pSGLD's epistemic variance grows away from the training data and its scores
improve inside the training domain too; Bayes by Backprop's epistemic variance
is spread through the training domain instead and slightly worsens the NLL
there. Both bands remain far too narrow in extrapolation: the draws stay close
to the trained mean, which extrapolates the oscillation poorly.

Source:
[`examples/jax/xsin/`](https://github.com/GroupiSP/probreg/tree/main/examples/jax/xsin)

## CMAPSS remaining useful life

Two-step probabilistic remaining-useful-life estimation on NASA CMAPSS FD001: a
CNN mean stage, a Gamma variance stage on its squared residuals, and a
composite Gaussian scored against the official test split.

```bash
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

To fetch the archive ahead of time, or only to plot the raw sensor
trajectories, run the data module on its own:

```bash
uv run --group example-cmapss python examples/jax/cmapss/data.py --fetch
uv run --group example-cmapss python examples/jax/cmapss/data.py
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
