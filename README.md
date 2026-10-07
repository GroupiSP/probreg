# probreg

`probreg` is a library for stage-oriented probabilistic regression: training
models that predict a distribution over the target, not just a point estimate.
Training runs as a sequence of explicit, checkpointed stages, such as fitting a
mean model and then a separate variance model on its residuals. A core of
backend-neutral contracts is implemented by an optional JAX backend built on
Flax NNX and Optax.

**Documentation:** <https://groupisp.github.io/probreg/>

## Install

```bash
uv add "probreg[jax] @ git+https://github.com/GroupiSP/probreg"
```

With pip, use `pip install "probreg[jax] @ git+https://github.com/GroupiSP/probreg"`.
Drop the `[jax]` extra for the backend-neutral contracts alone.

## Minimal example

<!-- --8<-- [start:minimal-example] -->
Fit a linear model to `y = 3x + 2` for a fixed number of epochs with the JAX
backend:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.types import Batch
from probreg.jax import create_optimizer, initialize_training_state, run_supervised


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
