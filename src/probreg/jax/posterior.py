"""The seam between the posterior stage and its inference methods.

A [`PosteriorStage`][probreg.jax.PosteriorStage] hands every
[`InferenceMethod`][probreg.jax.InferenceMethod] the same
[`PosteriorProblem`][probreg.jax.PosteriorProblem], steps it once per batch,
and reads back a [`Posterior`][probreg.jax.Posterior] it consumes only through
draws of the mean function at given inputs (ADR-0009).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from probreg.core.protocols import LoaderFactory
from probreg.core.types import Batch, PyTree


class Prior(Protocol):
    """A log-density over the parameter tree of the posterior network."""

    def log_prob(self, parameters: PyTree) -> jax.Array:
        """Return the scalar log-density of ``parameters``.

        Args:
            parameters: A tree with the posterior network's parameter tree.

        Returns:
            The scalar prior log-density.
        """
        ...


class Posterior(Protocol):
    """An approximate posterior over mean functions, consumed through draws.

    A draw is a whole mean function: under the same key, draw ``s`` is the
    same function at every input and in every batch. A finite posterior
    (retained samples, ensemble members) has a fixed set of draws and always
    uses all of them; an unlimited one (a variational distribution,
    MC-Dropout) takes the number of draws from its caller.
    """

    @property
    def num_draws(self) -> int | None:
        """The number of draws of a finite posterior, or ``None`` if unlimited."""
        ...

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Draw mean predictions at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key selecting the draws. A finite posterior may
                ignore it.
            num_samples: The number of draws ``S``. Must be ``None`` for a
                finite posterior and given for an unlimited one.

        Returns:
            The draws' mean predictions, shaped ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is given to a finite posterior, or
                omitted for an unlimited one.
        """
        ...


@dataclass(frozen=True)
class PosteriorProblem:
    """Everything a posterior stage hands its inference method.

    Attributes:
        initial_parameters: The warm start: a copy of the trained mean
            network's parameters, in the posterior network's parameter tree.
        mean_function: Maps ``(parameters, inputs)`` to the posterior
            network's mean predictions at ``inputs``.
        log_likelihood: Maps ``(parameters, batch)`` to the scalar Gaussian
            log-likelihood of the batch targets around the mean predictions,
            under the fixed aleatoric variance, scaled by ``N / B`` so that a
            minibatch estimates the full-data log-likelihood. Differentiable in
            ``parameters``; the aleatoric variance carries no gradient.
        prior: The prior over the parameter tree.
        train_loader: Factory producing the training batches of each epoch.
        dataset_size: The number of training examples ``N``.
    """

    initial_parameters: PyTree
    mean_function: Callable[[PyTree, PyTree], jax.Array]
    log_likelihood: Callable[[PyTree, Batch], jax.Array]
    prior: Prior
    train_loader: LoaderFactory
    dataset_size: int


class InferenceMethod(Protocol):
    """How a posterior stage turns a posterior problem into a posterior.

    The stage owns the epoch loop: it calls
    [`init`][probreg.jax.InferenceMethod.init] once, then
    [`update`][probreg.jax.InferenceMethod.update] once per training batch,
    and reads the current [`posterior`][probreg.jax.InferenceMethod.posterior]
    to validate each epoch. Every method honors the problem's prior.
    """

    @property
    def supports_early_stopping(self) -> bool:
        """Whether stopping on a validation improvement is meaningful.

        ``False`` for SG-MCMC, whose chain an early stopper would truncate.
        """
        ...

    def init(self, problem: PosteriorProblem) -> None:
        """Start inference from ``problem``, discarding any previous state.

        Args:
            problem: The posterior problem.
        """
        ...

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Advance inference by one training batch.

        Args:
            batch: The training batch.
            key: A fresh PRNG key for this batch.

        Returns:
            A mapping holding the scalar ``"loss"`` the method minimizes.
        """
        ...

    def posterior(self) -> Posterior:
        """Return the current posterior, independent of later updates.

        Returns:
            The posterior inferred so far.
        """
        ...

    def state(self) -> PyTree:
        """Return the method's full state, independent of later updates.

        This is what a best checkpoint stores, so that inference can resume
        from it, e.g. variational parameters plus optimizer state.

        Returns:
            A tree of arrays.
        """
        ...

    def load_state(self, state: PyTree) -> None:
        """Resume inference from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][probreg.jax.InferenceMethod.state].
        """
        ...


@dataclass(frozen=True)
class IsotropicGaussianPrior:
    """An independent zero-mean Gaussian on every parameter.

    The default precision of 1 is VeBNN's κ = 1.

    Attributes:
        precision: The inverse variance of every parameter, positive and
            finite. Defaults to ``1.0``.
    """

    precision: float = 1.0

    def __post_init__(self) -> None:
        """Validate the precision.

        Raises:
            ValueError: If ``precision`` is not positive and finite.
        """
        if not math.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be positive and finite.")

    def log_prob(self, parameters: PyTree) -> jax.Array:
        """Return the normalized log-density of a parameter tree.

        Args:
            parameters: A tree of parameter arrays.

        Returns:
            The scalar sum, over every parameter, of its Gaussian log-density.
        """
        leaves = jax.tree.leaves(parameters)
        count = sum(jnp.size(leaf) for leaf in leaves)
        squared_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        normalizer = 0.5 * count * math.log(self.precision / (2.0 * math.pi))
        return normalizer - 0.5 * self.precision * jnp.asarray(squared_norm)
