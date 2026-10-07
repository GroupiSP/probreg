"""Test doubles, a training-run driver and the documented packages, for all tests.

Core stays backend-neutral: nothing here imports the JAX backend at module
level, so `pytest tests/core` still collects without it. The fixtures that
need the backend import it lazily and skip when it is absent.

Every fixture is session-scoped and returns a class, a function or a
factory rather than a fresh instance, so a `@given` test can build a new
double per example without tripping Hypothesis's
`function_scoped_fixture` health check.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from probreg.core.early_stopping import EarlyStopper
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import EventSink, TrainingEvent
from probreg.core.types import StageResult

if TYPE_CHECKING:
    from probreg.jax import SupervisedLoss


class InMemoryTracker:
    """An event sink and experiment tracker that records what it is given."""

    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: dict[str, Any] = {}

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def log_params(self, values: dict[str, Any]) -> None:
        self.params.update(values)

    def log_metrics(self, values: dict[str, float], *, step: int) -> None:
        self.metrics.append((dict(values), step))

    def log_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    @property
    def tags(self) -> set[str]:
        """Every metric tag logged so far."""
        return {tag for values, _ in self.metrics for tag in values}


def _require_jax_backend() -> None:
    for dependency in ("jax", "jax.numpy", "flax.nnx", "optax"):
        pytest.importorskip(dependency)


# The packages the API reference documents, each with the optional dependencies it
# needs to import; a test of a package skips without them.
DOCUMENTED_PACKAGES = {
    "probreg.core": (),
    "probreg.jax": ("jax", "flax.nnx", "optax"),
}


@pytest.fixture(scope="session")
def documented_packages() -> tuple[str, ...]:
    """The names of every package the API reference documents."""
    return tuple(DOCUMENTED_PACKAGES)


@pytest.fixture(scope="session", params=sorted(DOCUMENTED_PACKAGES))
def documented_package(request: pytest.FixtureRequest) -> str:
    """The name of each documented package in turn."""
    return request.param


@pytest.fixture(scope="session")
def documented_module(documented_package: str) -> ModuleType:
    """The documented package, imported; skips without its optional dependencies."""
    for dependency in DOCUMENTED_PACKAGES[documented_package]:
        pytest.importorskip(dependency)
    return importlib.import_module(documented_package)


@pytest.fixture(scope="session")
def in_memory_tracker() -> type[InMemoryTracker]:
    """The recording event sink and experiment tracker class."""
    return InMemoryTracker


@pytest.fixture(scope="session")
def linear_model() -> type[Any]:
    """A one-input, one-output linear NNX model class.

    Typed loosely because tests reach into its `linear` layer, which an
    `nnx.Module` annotation would not admit.
    """
    _require_jax_backend()
    import jax
    from flax import nnx

    class LinearModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.linear = nnx.Linear(1, 1, rngs=rngs)

        def __call__(self, inputs: jax.Array) -> jax.Array:
            return self.linear(inputs)

    return LinearModel


@pytest.fixture(scope="session")
def squared_error() -> SupervisedLoss:
    """The optionally weighted mean squared error, as a supervised loss."""
    _require_jax_backend()
    import jax
    import jax.numpy as jnp
    from flax import nnx

    def squared_error(
        model: nnx.Module,
        inputs: jax.Array,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del key, training
        errors = jnp.square(model(inputs) - targets)
        if sample_weight is not None:
            errors = errors * sample_weight
        return jnp.mean(errors)

    return squared_error


@pytest.fixture(scope="session")
def constant_loader() -> LoaderFactory:
    """A loader of one constant batch, with a different target per split.

    The train target is 2 and every other split's is 1, so a model fitted on the train
    split never reaches zero validation loss.
    """
    _require_jax_backend()
    import jax.numpy as jnp

    from probreg.core.types import Batch

    def constant_loader(*, split: str, epoch: int) -> list[Batch]:
        del epoch
        target = 2.0 if split == "train" else 1.0
        return [Batch(inputs=jnp.array([[1.0]]), targets=jnp.array([[target]]))]

    return constant_loader


@pytest.fixture(scope="session")
def supervised_run(
    linear_model: type[Any],
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> Callable[..., StageResult]:
    """Return a driver of a real `run_supervised` call on a linear model."""
    _require_jax_backend()
    import jax
    import optax
    from flax import nnx

    from probreg.jax import (
        HeldOutValidation,
        create_optimizer,
        initialize_training_state,
        run_supervised,
    )

    def run(
        *sinks: EventSink,
        epochs: int = 3,
        learning_rate: float = 0.1,
        validate: bool = True,
        early_stopper: EarlyStopper | None = None,
        stage: str = "supervised",
    ) -> StageResult:
        """Fit the linear model on the constant loader through the given sinks.

        Args:
            *sinks: The event sinks to attach to the run.
            epochs: The maximum number of epochs to train for.
            learning_rate: The SGD learning rate.
            validate: Whether to validate on the held-out split every epoch.
            early_stopper: The early stopper to attach, if any.
            stage: The stage name the run records on its events.

        Returns:
            The run's stage result.
        """
        model = linear_model(rngs=nnx.Rngs(0))
        optimizer = create_optimizer(model, optax.sgd(learning_rate))
        state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
        return run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=constant_loader,
            loss=squared_error,
            state=state,
            epochs=epochs,
            validation=(
                HeldOutValidation(
                    model=model, loader=constant_loader, loss=squared_error
                )
                if validate
                else None
            ),
            early_stopper=early_stopper,
            event_sinks=sinks,
            stage=stage,
        )

    return run
