# Tracking a run

A runner reports what happens during training as
[training events][probreg.core.TrainingEvent], and anything that wants to watch
a run receives them as an [event sink][probreg.core.EventSink]. To record a run
in TensorBoard, MLflow or any other experiment tracker, you implement the three
methods of [`ExperimentTracker`][probreg.core.ExperimentTracker] and pass
[`TrackerEventSink`][probreg.core.TrackerEventSink], which wraps it, to the runner.
This page covers those three types, why the library stops there, and how the tag
for each metric and hyperparameter is built. Terms in bold are defined on the
[Glossary](../glossary.md) page.

## Events, sinks and trackers

A **training event** is a structured observation from a named point in a
stage's lifecycle. [`run_supervised`][probreg.jax.run_supervised] emits four of
them:

| Event name | Split | Carries |
| --- | --- | --- |
| `epoch_end` | `train` | the epoch's training metrics |
| `validation_end` | `validation` | the epoch's validation metrics, when a [`ValidationStrategy`][probreg.core.ValidationStrategy] is given |
| `best_model` | the early stopper's split | a [`Decision`][probreg.core.Decision], no metrics |
| `early_stop` | the early stopper's split | a [`Decision`][probreg.core.Decision], no metrics |

Every event carries its `stage`, its `split`, its `step` (the epoch) and the live
training state. An **event sink** is anything with an `on_event` method. Sinks
are passive: they observe the run and cannot change it. Pass any number of them
in `event_sinks`, so adding a tracker displaces nothing else that watches the run.

An **experiment tracker** is a destination that records parameters, metrics and
artifacts for one **run**. It knows nothing about events. It is told what to
record, never when a run reaches a point of interest. `TrackerEventSink` is the
adapter between the two: for each event it calls
[`log_metrics`][probreg.core.ExperimentTracker.log_metrics] with the event's
metrics under their metric tags, at the event's step. It never calls
[`log_params`][probreg.core.ExperimentTracker.log_params] or
[`log_artifact`][probreg.core.ExperimentTracker.log_artifact], since no training
event carries hyperparameters or figures. Your script calls those itself, before
and after the run.

The tracker below keeps everything in memory, which is enough to see what a real
one would receive:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import EarlyStopper, TrackerEventSink, flatten_parameters
from probreg.core.types import Batch
from probreg.jax import (
    HeldOutValidation,
    create_optimizer,
    initialize_training_state,
    run_supervised,
)


class InMemoryTracker:
    """An ExperimentTracker that keeps what it is told in memory."""

    def __init__(self):
        self.params = {}
        self.scalars = {}
        self.artifacts = {}

    def log_params(self, values):
        self.params.update(flatten_parameters(values))

    def log_metrics(self, values, *, step):
        for tag, value in values.items():
            self.scalars.setdefault(tag, []).append((step, value))

    def log_artifact(self, name, value):
        self.artifacts[name] = value


def squared_error(model, inputs, targets, sample_weight, key, training):
    return jnp.mean(jnp.square(model(inputs) - targets))


inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
targets = 3.0 * inputs + 2.0


def loader(*, split, epoch):
    return [Batch(inputs=inputs, targets=targets)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))

tracker = InMemoryTracker()
tracker.log_params({"optimizer": {"name": "sgd", "learning_rate": 0.1}})

run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=squared_error,
    state=state,
    epochs=3,
    validation=HeldOutValidation(model=model, loader=loader, loss=squared_error),
    early_stopper=EarlyStopper(metric="loss", mode="min", patience=1),
    event_sinks=[TrackerEventSink(tracker)],
)
tracker.log_artifact("notes", "three epochs of SGD")

print(sorted(tracker.params))
# ['optimizer/learning_rate', 'optimizer/name']
print(sorted(tracker.scalars))
# ['supervised/train/loss', 'supervised/validation/loss']
print([step for step, _ in tracker.scalars["supervised/validation/loss"]])
# [0, 1, 2]
```

The tags `supervised/train/loss` and `supervised/validation/loss` share their
bare metric name, so a tracker such as TensorBoard can draw them as two series on
one chart. The [tracking example](examples.md#tracking-a-run) does the same with a
real TensorBoard tracker in `tensorboard_tracker.py`, the file to copy into your
own project.

## Why core ships the bridge but never a tracker

`probreg.core` declares the event and tracker protocols and ships
`TrackerEventSink`, and that is all of its tracking support. Every concrete
tracker lives outside the library, in an example you copy.

The split follows what can be reused. Turning an event into a tracker record is
the same for every destination, and a mistake there costs you without showing
itself: a tag without the stage lets one stage's curve silently overwrite
another's. So the library owns that translation. Writing a scalar to a given
tracker, on the other hand, takes a handful of vendor-specific lines, and those
are most useful where you can see and edit them. A `probreg.trackers` package or
a `probreg[tensorboard]` extra would turn each of those few lines into a
dependency to pin and support. `TrackerEventSink` imports only core modules, so
probreg depends on no tracker by construction.

The cost is that a tracker the repository has no example for starts from the
TensorBoard one. That file is kept small and self-contained so it is easy to
rewrite. The decision is recorded in ADR 0005, and ADR 0006 amends it.

## The naming scheme

Every value a tracker receives is recorded under a name that the rules below
determine, so you can predict each tag before you open the tracker. The scheme
lives in one module, [`probreg.core.naming`](../reference/core.md#probregcorenaming),
and no other code joins or splits these strings.

### Metric names are bare

A **metric name** is the one name a metric has, chosen by whatever computes it:
bare `snake_case` such as `loss`, `rmse` or `point_crps`. No layer that passes
the metric along adds to it. The training loss and the validation loss are both
`loss`, in every stage. An early stopper names the metric it monitors the same
way, and picks the **split** with its `source` argument:

```python
from probreg.core import EarlyStopper, Split

stopper = EarlyStopper(metric="loss", mode="min", patience=5, source=Split.TRAIN)
print(stopper.monitored_metric_name())
# loss
```

### Metric tags are built where the run is observed

A **metric tag** is the namespaced identity of a recorded metric,
`stage/split/metric`. It is built only where a run is observed, by
[`metric_tag`][probreg.core.metric_tag], from the event's stage, the event's
split and the bare name. `/` is the only separator, and no segment may be empty
or contain it, so [`parse_metric_tag`][probreg.core.parse_metric_tag] always
recovers exactly the parts the tag was built from:

```python
from probreg.core import Split, metric_tag, parse_metric_tag

tag = metric_tag("mean", Split.VALIDATION, "loss")
print(tag)
# mean/validation/loss
print(parse_metric_tag(tag))
# MetricTag(stage='mean', split=<Split.VALIDATION: 'validation'>, metric='loss')

try:
    metric_tag("mean", Split.TRAIN, "nll/sum")
except ValueError as error:
    print(error)
# metric 'nll/sum' may not contain '/'.
```

The split is data, not guesswork. Each event carries the
[`Split`][probreg.core.Split] its metrics were measured on, `train` or
`validation`, set by the runner, which is the only party that knows it. Neither
the event's name nor the metric's name is used to infer it.

A run's own history uses the same strings. `state.metric_history` is keyed by
metric tag, so a history key and a tracker tag for the same series are equal:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.types import Batch
from probreg.jax import create_optimizer, initialize_training_state, run_supervised


def squared_error(model, inputs, targets, sample_weight, key, training):
    return jnp.mean(jnp.square(model(inputs) - targets))


def loader(*, split, epoch):
    inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    return [Batch(inputs=inputs, targets=2.0 * inputs)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))
run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=squared_error,
    state=state,
    epochs=2,
)
print(list(state.metric_history))
# ['supervised/train/loss']
```

### The stage segment is always present

The **stage segment**, the first part of a tag, is never dropped, even when the
run has only one stage. `run_supervised` names its stage `supervised` unless you
pass `stage=`, so the run above records `supervised/train/loss` and not
`train/loss`.

The reason is that each stage counts its own steps from zero. In a two-step run,
[`MeanStage`][probreg.jax.MeanStage] and
[`GammaVarianceStage`][probreg.jax.GammaVarianceStage] both report a training
loss from epoch 0 onwards. Under the tags `mean/train/loss` and
`variance/train/loss` they are two curves. Without the stage segment both would
write to `train/loss` at the same steps, and the variance stage's curve would
overwrite the mean stage's. Keeping the segment in single-stage runs too means a
script's tags do not change when it gains a second stage, and a tag always has
exactly three parts.

### Parameter paths name hyperparameters

A **parameter path** is the identity of a recorded hyperparameter: its key path
through a nested parameter mapping, joined with the same `/`. A tracker receives
the nested mapping in
[`log_params`][probreg.core.ExperimentTracker.log_params] and, if it stores flat
names, flattens it with
[`flatten_parameters`][probreg.core.flatten_parameters]. Each leaf key is a bare
`snake_case` name, and a split is a level of nesting of its own, never part of a
leaf key: `data/train/samples`, not `data/train_samples`.

```python
from probreg.core import flatten_parameters

parameters = {
    "optimizer": {"name": "adam", "learning_rate": 1e-3},
    "data": {
        "train": {"samples": 256, "batch_size": 32},
        "validation": {"samples": 64, "batch_size": 64},
    },
}
for path, value in flatten_parameters(parameters).items():
    print(path, value)
# optimizer/name adam
# optimizer/learning_rate 0.001
# data/train/samples 256
# data/train/batch_size 32
# data/validation/samples 64
# data/validation/batch_size 64
```

A key that is empty or contains `/` raises `ValueError`, so two different key
paths can never collapse onto one parameter path. Converting values the tracker
cannot store, such as a tuple for TensorBoard's HParams, stays in the tracker.

### Decision events carry no metrics

A **decision event** reports a judgement about a measurement that was already
emitted: `best_model` when the early stopper sees an improvement, `early_stop`
when it gives up. Its `metrics` mapping is empty. The measurement it judged
travels in its `decision` field as a **decision**, the monitored metric's bare
name and its value at that step, and its `split` is the split that metric was
measured on.

So `TrackerEventSink` logs nothing for a decision event. On an improving epoch,
the validation loss reaches the tracker once, from `validation_end`, and not a
second time from `best_model` at the same step. To record decisions as well,
write a sink of your own that reads `event.decision`:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import EarlyStopper, Split
from probreg.core.types import Batch
from probreg.jax import create_optimizer, initialize_training_state, run_supervised


class DecisionPrinter:
    """An event sink that prints decision events and ignores the rest."""

    def on_event(self, event):
        if event.decision is not None:
            print(event.name, event.step, event.split, event.metrics, event.decision.metric)


def squared_error(model, inputs, targets, sample_weight, key, training):
    return jnp.mean(jnp.square(model(inputs) - targets))


def loader(*, split, epoch):
    inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    return [Batch(inputs=inputs, targets=2.0 * inputs)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))
run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=squared_error,
    state=state,
    epochs=2,
    early_stopper=EarlyStopper(metric="loss", mode="min", patience=0, source=Split.TRAIN),
    event_sinks=[DecisionPrinter()],
)
# best_model 0 train {} loss
# best_model 1 train {} loss
```

The design decisions behind this section are recorded in ADR 0006.
