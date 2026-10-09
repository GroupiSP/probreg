# Checkpoints

A checkpoint is everything needed to pick a staged workflow up again at the
point it was taken: the training state, the model and optimizer state, the
random key and the early stopper's progress. The runners save one whenever an
early stopper reports a new best model, so that the best model can be put back
once training stops. Each stage goes on to finalize its best checkpoint as the
one it hands on, so that, for example, a variance stage can start from the mean
stage's checkpoint later, in a fresh training state.

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
defaults to the [checkpoint key](../glossary.md) `stage/best` (here
`supervised/best`, the runner's default stage), so the store always holds the
best epoch so far. Without
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

best = store.load("supervised/best")
history = state.metric_history["supervised/train/loss"]
assert history[best.epoch] == min(history)

restore_checkpoint(best, state=state, model=model, optimizer=optimizer)
assert state.metric_history["supervised/train/loss"] == history[: best.epoch + 1]
```

The early stopper here monitors the training loss to stay self-contained; one
that monitors a validation metric needs a validation strategy as well.

## A stage's finalized checkpoint

[`MeanStage`][probreg.jax.MeanStage] and
[`GammaVarianceStage`][probreg.jax.GammaVarianceStage] go one step further when
their options carry an early stopper and a checkpoint store, and both follow the
same rule. At the end of `train` the stage restores its best checkpoint into its
live model, so the model it leaves behind is the best one rather than the last.
It then saves that checkpoint again under the same key, finalized: its training
state is at the stage's ready lifecycle state, `MEAN_READY` or `VARIANCE_READY`,
and its metadata records the stage and `"stage_complete": True`, for example
`{"stage": "mean", "stage_complete": True}`. The returned result reports the
restored epoch's training metrics. So the variance stage computes residuals from
the best mean model, and a finalized checkpoint is the hand-off point after each
stage.

The variance checkpoint snapshots only the variance model and optimizer, since
the mean model is frozen and already lives in the mean stage's checkpoint. When
the variance stage restores it, it keeps the mean model and its optimizer
registered under `mean_model_name` and `mean_optimizer_name`, so the state still
passes [`validate`][probreg.jax.GammaVarianceStage.validate].
`restore_checkpoint` on its own restores clean-slate: it registers only the
model and optimizer you pass it.

Both stages read the key from their own
[`SupervisedStageOptions`][probreg.jax.SupervisedStageOptions]. Left unset, it
defaults to the stage's own key, `mean/best` or `variance/best`, so the two
stages can share one store without overwriting each other's checkpoints. An
explicit `checkpoint_key` still wins.
[`select_checkpoint`][probreg.jax.MeanStage.select_checkpoint] returns a
[`CheckpointRef`][probreg.core.CheckpointRef] to the key a stage saves under.

## Resuming across stages

The finalized checkpoints are enough to pick a staged run up again in a new
training state, with freshly built objects: models and optimizers of the same
architecture to restore into, and nothing else from the first run. Each stage
restores its own checkpoint with `restore`
([`MeanStage.restore`][probreg.jax.MeanStage.restore] and
[`GammaVarianceStage.restore`][probreg.jax.GammaVarianceStage.restore]), mean
first, then variance. The snippet below trains both stages, then starts over
from the store alone, as a later process would:

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
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jnp.sin(7.0 * inputs)


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


def build_stages(
    store: InMemoryCheckpointStore, seed: int
) -> tuple[MeanStage, GammaVarianceStage]:
    mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(seed))
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(seed + 1))
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
        ),
    )
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(0.1)),
        source_loader=loader,
        options=SupervisedStageOptions(
            epochs=3,
            early_stopper=EarlyStopper(
                metric="loss", mode="min", patience=1, source="train"
            ),
            checkpoint_store=store,
        ),
        splits=("train",),
    )
    return mean_stage, variance_stage


store = InMemoryCheckpointStore()

# The first run trains both stages and keeps their finalized checkpoints.
first_mean, first_variance = build_stages(store, seed=0)
first_state = TrainingState(rng_state=jax.random.key(0))
for stage in (first_mean, first_variance):
    stage.prepare(first_state)
    stage.train(first_state)

# A later run rebuilds the stages and restores them from the store, mean first.
mean_stage, variance_stage = build_stages(store, seed=10)
state = TrainingState()
mean_stage.restore(state, store.load("mean/best"))
variance_stage.restore(state, store.load("variance/best"))

assert state.lifecycle_state is StageState.VARIANCE_READY
assert variance_stage.validate(state).passed
assert state.model_components["mean_model"] is mean_stage.model
assert jnp.allclose(mean_stage.model(inputs), first_mean.model(inputs))
assert jnp.allclose(
    variance_stage.model(inputs).rate, first_variance.model(inputs).rate
)
```

`restore` takes only the stage's own finalized checkpoint, and checks it before
changing anything: it raises `ValueError` if the checkpoint's lifecycle state is
not the stage's ready state, or its metadata does not name the stage and mark it
complete. So the other stage's checkpoint, or a best checkpoint left behind by a
run that stopped mid-stage, is refused. The variance stage also refuses while no
mean model is registered, which is why the mean stage restores first. The
restored models and optimizers must have the same structure as the ones that
were saved, or `restore` raises `ValueError`, again before changing anything.

The restored state carries the stages' metric history, their parameter roles
and the random key. After the mean stage's `restore` alone, the state is
`MEAN_READY`, so the variance stage can also be prepared and trained from there
rather than restored, continuing the run rather than starting a new one.

Calling [`restore_checkpoint`][probreg.jax.restore_checkpoint] directly on a
staged checkpoint still works, but restores clean-slate and leaves the
registry names to you; the stage methods are the way to resume a staged run.

### The posterior stage

The optional [`PosteriorStage`][probreg.jax.PosteriorStage] saves under
`posterior/best` by default, so it can share the store with the other two
stages. With an early stopper, every improvement saves a best checkpoint holding
the inference method's full [`state`][probreg.jax.InferenceMethod.state], e.g.
variational parameters plus optimizer state. At the end of training the method
resumes from the best one, and the stage overwrites it with its finalized
checkpoint, which holds the posterior alone: the method's
[`posterior_state`][probreg.jax.InferenceMethod.posterior_state], kept as the
saved state's `posterior_state`, with lifecycle state `POSTERIOR_READY`.
Neither the mean and variance weights nor the optimizer state are saved again.
A method that does not support early stopping, such as SG-MCMC, writes only the
finalized checkpoint, at the end.

A later process resumes all three stages in order:

```{.python notest}
mean_stage.restore(state, store.load("mean/best"))
variance_stage.restore(state, store.load("variance/best"))
posterior_stage.restore(state, store.load("posterior/best"))
```

[`PosteriorStage.restore`][probreg.jax.PosteriorStage.restore] initializes its
inference method on the restored mean and variance models and hands it the
saved posterior state through
[`load_posterior`][probreg.jax.InferenceMethod.load_posterior]. It keeps the
mean and variance registrations, and refuses, before changing anything, while
either is missing, when the checkpoint is not its finalized checkpoint, or when
it was saved under other component names.

[Two-step mean/variance training](two-step-training.md) explains the stages
themselves.

## Keeping checkpoints beyond one process

An `InMemoryCheckpointStore` is gone when the process ends, so the resume above
works only within one process. Keeping checkpoints across processes takes a
store of your own. `CheckpointStore` is a protocol, so any object with matching
`save`, `load` and `exists` methods will do, with no base class to inherit.
Such a store has to serialize the whole checkpoint, including the backend's
snapshots and the random key, and `probreg` does not ship one.
