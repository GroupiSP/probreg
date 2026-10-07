# Validation and early stopping

[`run_supervised`][probreg.jax.run_supervised] trains for a fixed number of
epochs unless you give it two optional collaborators: a
[`ValidationStrategy`][probreg.core.ValidationStrategy] that measures the model
after each epoch, and an [`EarlyStopper`][probreg.core.EarlyStopper] that
decides from a monitored metric when to stop. This guide explains how the two
fit together and why an early stopper that monitors validation metrics cannot
run without a strategy. It assumes the setup from
[Training a supervised model](training.md).

## Validation is a strategy you inject

The runner does not know what "validation" means for your problem. It might be
a held-out loader, a fold of a cross-validation, a rolling window over a time
series, a grouped split, or an external evaluator. Rather than build any of
these into the training loop, `run_supervised` takes a strategy: a callable
`validation(state, *, epoch)` that returns a
[`ValidationResult`][probreg.core.ValidationResult] whose `metrics` are keyed
by bare metric names.

The runner calls it once after every training epoch, records each metric in
`state.metric_history` under the `validation` split (for example
`supervised/validation/loss`), and emits a `validation_end`
[training event](../glossary.md). Where the data comes from and how it is
scored stay inside the strategy.

For the conventional case,
[`HeldOutValidation`][probreg.jax.HeldOutValidation] evaluates the model on a
loader called with `split="validation"`, using the same kind of
[`SupervisedLoss`][probreg.jax.SupervisedLoss] as training and, optionally, its
own [`MetricSuite`][probreg.jax.MetricSuite]. It holds a reference to the live
model, so it always scores the parameters the last epoch produced.

## Stopping on a validation metric

An `EarlyStopper` monitors one metric by its bare name, in a chosen direction
(`mode`), measured on a chosen split (`source`). Its `source` defaults to
`validation`, so the stopper below watches the validation loss:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import (
    Batch,
    EarlyStopper,
    InMemoryCheckpointStore,
    SquaredErrorLoss,
)
from probreg.jax import (
    HeldOutValidation,
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
targets = 3.0 * inputs + 2.0


def loader(*, split, epoch):
    half = slice(0, None, 2) if split == "train" else slice(1, None, 2)
    return [Batch(inputs=inputs[half], targets=targets[half])]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))
loss = make_supervised_loss(SquaredErrorLoss())
store = InMemoryCheckpointStore()

result = run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=loss,
    state=state,
    epochs=3,
    validation=HeldOutValidation(model=model, loader=loader, loss=loss),
    early_stopper=EarlyStopper(metric="loss", mode="min", patience=1),
    checkpoint_store=store,
)

assert len(state.metric_history["supervised/validation/loss"]) == 3
assert store.exists("best")
```

The metric is `"loss"`, not `"validation_loss"`: a [metric name](../glossary.md)
is the same on every split, and the stopper's `source` is what says which split
to read it from. If the monitored name is not among the metrics the chosen split
produced, the runner raises a `ValueError` at the end of the first epoch.

After each epoch the stopper observes the monitored value. The first value is
always an improvement; after that, a value improves only if it beats the best
so far by more than `min_delta`. Training stops once the number of consecutive
non-improving epochs exceeds `patience`, so `patience=0` stops at the first
epoch that fails to improve. `epochs` remains the upper bound either way.

## Why the stopper needs a strategy

A stopper whose `source` is `validation` reads its metric from the
`ValidationResult` of the same epoch. Without a strategy no such result exists,
so there is nothing to observe. Rather than silently fall back to the training
metric of the same name, which would stop on a quantity you did not ask for,
`run_supervised` refuses the combination before it touches the model or the
training state:

```python
import jax
import optax
from flax import nnx

from probreg.core import EarlyStopper, SquaredErrorLoss
from probreg.jax import (
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))

try:
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=lambda *, split, epoch: [],
        loss=make_supervised_loss(SquaredErrorLoss()),
        state=state,
        epochs=3,
        early_stopper=EarlyStopper(metric="loss", mode="min", patience=1),
    )
except ValueError as error:
    print(error)  # validation metric monitoring requires a validation strategy.
else:
    raise AssertionError("expected a ValueError")
```

[`EarlyStopper.expects_validation`][probreg.core.EarlyStopper.expects_validation]
is the check the runner makes.

## The training-metric alternative

When you have no validation data, or want to stop on convergence of the fit
itself, monitor a training metric by setting `source` to `train` explicitly.
No strategy is needed, and the stopper observes the epoch's training metrics:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import Batch, EarlyStopper, Split, SquaredErrorLoss
from probreg.jax import (
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
targets = 3.0 * inputs + 2.0


def train_loader(*, split, epoch):
    return [Batch(inputs=inputs, targets=targets)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))

result = run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss=make_supervised_loss(SquaredErrorLoss()),
    state=state,
    epochs=3,
    early_stopper=EarlyStopper(
        metric="loss", mode="min", patience=1, source=Split.TRAIN, min_delta=1e-4
    ),
)

assert "supervised/validation/loss" not in state.metric_history
```

The default is `validation` because that is what early stopping is usually
for: stopping when the model stops generalising. A training metric measures
only the fit to the data being trained on, so choosing it is a decision the
code states rather than one the runner makes for you.

## Best models and checkpoints

Whenever the stopper reports an improvement, the runner emits a `best_model`
[decision event](../glossary.md), and when it stops, an `early_stop` event.
Both carry the [`Decision`][probreg.core.Decision] they judged, the monitored
metric and its value, and no metrics of their own.

If you pass a [`CheckpointStore`][probreg.core.CheckpointStore], each
improvement also saves a [`Checkpoint`][probreg.core.Checkpoint] under
`checkpoint_key` (`"best"` by default), replacing the previous one. The runner
does not restore it: when training ends, the live model holds the last epoch's
parameters, not the best ones. To continue from the best model, load the
checkpoint and pass it to
[`restore_checkpoint`][probreg.jax.restore_checkpoint]; the
[Checkpoints](checkpoints.md) guide shows how.
