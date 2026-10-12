# stagereg

[![PyPI version](https://img.shields.io/pypi/v/stagereg)](https://pypi.org/project/stagereg/)
[![Python versions](https://img.shields.io/pypi/pyversions/stagereg)](https://pypi.org/project/stagereg/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

`stagereg` is a library for stage-oriented probabilistic regression: training
models that predict a distribution over the target, not just a point estimate.
Training runs as a sequence of explicit, checkpointed stages, such as fitting a
mean model and then a separate variance model on its residuals. A core of
backend-neutral contracts is implemented by an optional JAX backend built on
Flax NNX and Optax.

**Documentation:** <https://groupisp.github.io/stagereg/>. The docs site tracks
the `main` branch, with no per-release docs before 1.0, so it may describe
features newer than your installed version.

## Install

<!-- --8<-- [start:install] -->
Install with the JAX backend (JAX, Flax and Optax):

```bash
pip install "stagereg[jax]"
```

The optional `plot` extra also installs matplotlib, for plotting results:

```bash
pip install "stagereg[jax,plot]"
```

With uv, use the equivalent `uv add` form:

```bash
uv add "stagereg[jax]"
uv add "stagereg[jax,plot]"
```

Drop the `[jax]` extra for the backend-neutral contracts alone, which depend
only on NumPy.
<!-- --8<-- [end:install] -->

## Minimal example

<!-- --8<-- [start:minimal-example] -->
Fit a linear model to `y = 3x + 2` for a fixed number of epochs with the JAX
backend:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core.types import Batch
from stagereg.jax import create_optimizer, initialize_training_state, run_supervised


def squared_error(model, inputs, targets, sample_weight, key, training):
    return jnp.mean(jnp.square(model(inputs) - targets))


inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
targets = 3.0 * inputs + 2.0


def loader(*, split, epoch):
    return [Batch(inputs=inputs, targets=targets)]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))

result = run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=squared_error,
    state=state,
    epochs=3,
)
print(f"final training loss: {result.loss:.4f}")
```
<!-- --8<-- [end:minimal-example] -->
