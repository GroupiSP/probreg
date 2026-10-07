# probreg

`probreg` is a library for stage-oriented probabilistic regression: training
models that predict a distribution over the target, not just a point estimate.
Its core is a set of backend-neutral contracts for batches, predictive
distributions, losses, metrics, checkpoints, early stopping and tracking. An
optional JAX backend built on Flax NNX and Optax implements them.

## What "stage-oriented" means

In `probreg`, training is a sequence of explicit, named stages rather than one
opaque `fit` call. Each stage declares what it requires and what it produces,
takes the shared training state, trains, validates, and selects the checkpoint
it hands on. The training state records where it is in the lifecycle
`NEW -> INITIALIZED -> MEAN_READY -> VARIANCE_READY`, and a transition that
skips a step is refused. That is what lets two-step training fit a mean model
first, checkpoint it, and resume from that checkpoint into a separate variance
stage, and it is why every metric a run reports is tagged with the stage that
produced it.

## Install

The base package has only NumPy as a dependency and holds the backend-neutral
contracts:

```bash
uv add "probreg @ git+https://github.com/GroupiSP/probreg"
```

The `jax` extra adds the JAX backend (JAX, Flax and Optax):

```bash
uv add "probreg[jax] @ git+https://github.com/GroupiSP/probreg"
```

With pip, use `pip install "probreg[jax] @ git+https://github.com/GroupiSP/probreg"`.

## Minimal example

--8<-- "README.md:minimal-example"
