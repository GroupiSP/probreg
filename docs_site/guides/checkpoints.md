# Checkpoints

A checkpoint is everything needed to pick a staged workflow up again at the
point it was taken: the training state, the model and optimizer state, the
random key and the early stopper's progress. The runners save one whenever an
early stopper reports a new best model, so that the best model can be put back
once training stops. The mean stage goes on to finalize its best checkpoint as
the one it hands on, so that a variance stage can start from it later, in a
fresh training state.

## Checkpoints and stores

A [`Checkpoint`][probreg.core.Checkpoint] is a frozen value with these fields:

| Field | Holds |
| --- | --- |
| `state` | The [`TrainingState`][probreg.core.TrainingState] at that epoch: lifecycle, stage, parameter roles, frozen components and metric history |
| `epoch` | The epoch the checkpoint was taken at |
| `parameters` | The model's state; in the JAX backend an [`NnxSnapshot`][probreg.jax.NnxSnapshot] |
| `optimizer_state` | The optimizer's state, also an `NnxSnapshot` |
| `rng_state` | The random key |
| `early_stopping_state` | The [`EarlyStoppingState`][probreg.core.EarlyStoppingState] when it was taken |
| `metadata` | Free-form annotations, such as the stage that finalized it |

The JAX runners build the `state` field with
[`freeze_training_state`][probreg.jax.freeze_training_state], which copies the
mutable containers and leaves out the live model and optimizer objects. Those
are captured separately, as snapshots, so that a saved checkpoint does not keep
changing as training goes on.

Checkpoints are kept in a [`CheckpointStore`][probreg.core.CheckpointStore],
which saves, loads and checks for them under keys the caller chooses. The store
gives a key no meaning of its own. `probreg` ships one implementation,
[`InMemoryCheckpointStore`][probreg.core.InMemoryCheckpointStore], which keeps
checkpoints in process memory for as long as the store exists:

```python
import pytest

from probreg.core import Checkpoint, InMemoryCheckpointStore, TrainingState

store = InMemoryCheckpointStore()
assert not store.exists("best")

store.save("best", Checkpoint(state=TrainingState(), epoch=3))
store.save("best", Checkpoint(state=TrainingState(), epoch=7))
assert store.load("best").epoch == 7  # saving under a used key replaces it

with pytest.raises(KeyError):
    store.load("missing")
```

## Keeping the best model

[`run_supervised`][probreg.jax.run_supervised] saves a checkpoint only when it
has both an `early_stopper` and a `checkpoint_store`, and only on an epoch where
the early stopper reports an improvement. It saves under `checkpoint_key`, which
defaults to `"best"`, so the store always holds the best epoch so far. Without
an early stopper there is no notion of best, and nothing is saved.

The runner leaves the live model at its last epoch, not its best one.
[`restore_checkpoint`][probreg.jax.restore_checkpoint] puts the best one back,
updating the model and optimizer in place and restoring the training state that
was saved with them:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import Batch, EarlyStopper, InMemoryCheckpointStore, SquaredErrorLoss
from probreg.jax import (
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    restore_checkpoint,
    run_supervised,
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))
store = InMemoryCheckpointStore()

run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=make_supervised_loss(SquaredErrorLoss()),
    state=state,
    epochs=3,
    early_stopper=EarlyStopper(metric="loss", mode="min", patience=1, source="train"),
    checkpoint_store=store,
)

best = store.load("best")
history = state.metric_history["supervised/train/loss"]
assert history[best.epoch] == min(history)

restore_checkpoint(best, state=state, model=model, optimizer=optimizer)
assert state.metric_history["supervised/train/loss"] == history[: best.epoch + 1]
```

The early stopper here monitors the training loss to stay self-contained; one
that monitors a validation metric needs a validation strategy as well.

## The mean stage's checkpoint

[`MeanStage`][probreg.jax.MeanStage] goes one step further when its options
carry an early stopper and a checkpoint store. At the end of
[`train`][probreg.jax.MeanStage.train] it restores its best checkpoint into the
live mean model, so the variance stage computes residuals from the best mean
model rather than the last one. It then saves that checkpoint again under the
same key, finalized: its training state is now `MEAN_READY`, and its metadata
records `{"stage": "mean", "stage_complete": True}`. The finalized checkpoint is
the hand-off point between the two stages.

[`GammaVarianceStage`][probreg.jax.GammaVarianceStage] saves its best
checkpoint the same way but does not restore it: its live model holds the last
epoch, and `restore_checkpoint` brings back the best one.

Both stages read the key from their own
[`SupervisedStageOptions`][probreg.jax.SupervisedStageOptions], and both default
to `"best"`. When the two stages share a store, give each its own key, or the
variance stage's best checkpoint replaces the mean stage's.
[`select_checkpoint`][probreg.jax.MeanStage.select_checkpoint] returns a
[`CheckpointRef`][probreg.core.CheckpointRef] to the key a stage was configured
with.

## Resuming across stages

The finalized mean checkpoint is enough to run the variance stage in a new
training state, with freshly built objects: a mean model and optimizer of the
same architecture to restore into, and nothing else from the first run. The
snippet below trains the mean stage, then starts over from the store alone, as a
later process would:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import (
    Batch,
    EarlyStopper,
    InMemoryCheckpointStore,
    StageState,
    TrainingState,
)
from probreg.jax import (
    GammaHead,
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    restore_checkpoint,
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jnp.sin(7.0 * inputs)


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


store = InMemoryCheckpointStore()

# The first run trains the mean stage and keeps its best checkpoint.
mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
mean_stage = MeanStage(
    model=mean_model,
    optimizer=create_optimizer(mean_model, optax.adam(0.1)),
    train_loader=loader,
    options=SupervisedStageOptions(
        epochs=3,
        early_stopper=EarlyStopper(
            metric="loss", mode="min", patience=1, source="train"
        ),
        checkpoint_store=store,
        checkpoint_key="mean/best",
    ),
)
first_state = TrainingState(rng_state=jax.random.key(0))
mean_stage.prepare(first_state)
mean_stage.train(first_state)

# A later run rebuilds the mean model and restores it from the store.
checkpoint = store.load("mean/best")
assert checkpoint.state.lifecycle_state is StageState.MEAN_READY
assert checkpoint.metadata["stage_complete"]

restored_model = nnx.Linear(1, 1, rngs=nnx.Rngs(1))
state = TrainingState()
restore_checkpoint(
    checkpoint,
    state=state,
    model=restored_model,
    optimizer=create_optimizer(restored_model, optax.adam(0.1)),
    model_name="mean_model",
    optimizer_name="mean_optimizer",
)
assert state.lifecycle_state is StageState.MEAN_READY
assert jnp.allclose(restored_model(inputs), mean_model(inputs))

# The variance stage starts from the restored state as if it had never stopped.
variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
variance_stage = GammaVarianceStage(
    model=variance_model,
    optimizer=create_optimizer(variance_model, optax.adam(0.1)),
    source_loader=loader,
    options=SupervisedStageOptions(epochs=2),
    splits=("train",),
)
variance_stage.prepare(state)
variance_stage.train(state)
assert state.lifecycle_state is StageState.VARIANCE_READY
assert "mean/train/loss" in state.metric_history  # carried over by the checkpoint
```

The restored model and optimizer must have the same structure as the ones that
were saved; `restore_checkpoint` raises `ValueError` before changing anything if
they do not. Pass the names the mean stage registered them under, `mean_model`
and `mean_optimizer`, so the variance stage finds the mean model where it
expects it. The restored state carries the mean stage's metric history, its
parameter roles and the random key, so the variance stage continues the run
rather than starting a new one.

[Two-step mean/variance training](two-step-training.md) explains the stages
themselves.

## Keeping checkpoints beyond one process

An `InMemoryCheckpointStore` is gone when the process ends, so the resume above
works only within one process. Keeping checkpoints across processes takes a
store of your own. `CheckpointStore` is a protocol, so any object with matching
`save`, `load` and `exists` methods will do, with no base class to inherit.
Such a store has to serialize the whole checkpoint, including the backend's
snapshots and the random key, and `probreg` does not ship one.
