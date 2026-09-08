# TensorBoard-tracked MVE regression

This example trains a deliberately trivial one-dimensional mean-variance
estimation (MVE) model and records the whole run to TensorBoard: per-epoch
training and validation scalars, the run's hyperparameters, and a final figure
of the fitted mean with its predictive interval.

The regression problem is
[`examples/jax/mve_regression.py`](../mve_regression.py) reused verbatim —
heteroscedastic noise on a linear mean, held-out validation, early stopping on
the validation loss, best-model checkpointing — so diffing the two examples
shows the tracking and nothing else. There is **one exception**: the
early-stopping patience is raised from 5 to 20. Patience 5 stops after a few
dozen epochs and produces too thin a curve to justify opening TensorBoard.

## Installing and running

TensorBoard and its writer are example-only tooling, declared as the
`example-tracking` dependency group rather than a project extra: the library
itself gains no dependency, hard or optional, from this example.

```shell
uv run --group example-tracking python examples/jax/tracking/run.py --logdir runs/
```

Then view the run:

```shell
uv run --group example-tracking tensorboard --logdir runs/
```

`--logdir` defaults to `runs/`, and each invocation writes to its own
UTC-timestamped subdirectory of it, so successive runs appear side by side in
TensorBoard instead of interleaving into one broken curve. `--epochs` caps the
number of epochs (200 by default) before early stopping intervenes.

## Description of the files

### `tensorboard_tracker.py`

`TensorBoardTracker`, an implementation of
`probreg.core.tracking.ExperimentTracker` — the file to copy into your own
project. It is the only TensorBoard-specific code here, so swapping TensorBoard
for MLflow, Weights & Biases or Aim means rewriting this module and changing
one constructor line in `run.py`.

It is backend-neutral: its surface accepts Python floats, strings and
Matplotlib figures, never anything JAX-shaped, so the same tracker serves any
`probreg` backend. Converting backend arrays to floats belongs to the run
script, which is already the backend-specific layer.

- The writer is a **constructor argument**, defaulting to a `tensorboardX`
  summary writer on the given log directory. `tensorboardX` is imported only
  when that default is built, so the module's mapping logic stays importable —
  and testable — with the dependency absent.
- `log_metrics` writes one scalar summary per entry, at the given step.
- `log_params` writes through TensorBoard's HParams plugin, so runs are
  comparable in a sortable table. Nested values are flattened onto one entry
  per leaf, keys joined with `/`; a key containing `/` is rejected rather than
  allowed to collide with a nested key path. Recorded parameters cover the
  hyperparameters **and** the loss and metric identities, so runs differing in
  objective are distinguishable in the table.
- `log_artifact` dispatches on the value's type: a Matplotlib figure becomes an
  image summary, a string a text summary, and anything else raises `TypeError`.
- `flush` and `close` are exposed so the run script owns the writer's lifetime.

The module is deliberately **not** named `tensorboard.py`: that would shadow the
installed `tensorboard` package for anything importing from this directory.

### `run.py`

The runner. `main(argv)` parses the arguments above, builds the tracker for this
run's directory, and calls `run_tracked_training`, which is typed against
`ExperimentTracker` rather than against the concrete tracker.

Tracking reaches the run through `probreg.core.tracking.TrackerEventSink`, the
library's bridge from training events to a tracker, passed **alongside** the
example's own printing event sink: `event_sinks` is a sequence, and adding a
tracker displaces nothing a reader already sees in the terminal.

The sink tags every metric `<stage>/<event prefix><metric>`, with the stage
segment unconditional so that two stages of a staged run cannot overwrite each
other's curves. The validation strategy's metric prefix is cleared here so that
the tracker owns the whole tag, which yields matched `supervised/train/loss` and
`supervised/validation/loss` tags — TensorBoard renders those as two series on
one chart, where overfitting is visible without switching charts. The cost of
clearing that prefix is that `state.metric_history` records validation metrics
unprefixed, which is why the printing sink reads them without a prefix.

After training, the script logs one figure showing the validation data, the
predicted mean, and the 95% predictive interval `loc ± 1.96 * scale`. The
interval widening with `x` is the evidence that the model learned the
input-dependent noise. The Matplotlib backend is set to a non-interactive one
and the figure is never shown: it exists to be logged, and a plot window would
break headless and CI runs.
