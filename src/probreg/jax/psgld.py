"""Preconditioned stochastic-gradient Langevin dynamics (pSGLD).

An SG-MCMC [`InferenceMethod`][probreg.jax.InferenceMethod]: a Langevin chain
over the posterior network's parameters, preconditioned by an RMSprop estimate
of the gradient's scale (Li et al., 2016). The chain's positions retained
after burn-in, at the thinning interval, are the draws of a finite
[`RetainedSamplesPosterior`][probreg.jax.RetainedSamplesPosterior].
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from probreg.core.types import Batch, PyTree
from probreg.jax.posterior import PosteriorProblem


@dataclass(frozen=True)
class RetainedSamplesPosterior:
    """A finite posterior whose draws are retained parameter samples.

    Draw ``s`` is the posterior network with the ``s``-th retained
    parameters, so it is the same mean function at every input, and every
    call uses all the draws.

    Attributes:
        mean_function: Maps ``(parameters, inputs)`` to the posterior
            network's mean predictions at ``inputs``.
        samples: The retained parameter trees, stacked along a leading axis of
            length ``S``.
    """

    mean_function: Callable[[PyTree, PyTree], jax.Array]
    samples: PyTree

    @property
    def num_draws(self) -> int:
        """The number of retained samples ``S``."""
        return int(jax.tree.leaves(self.samples)[0].shape[0])

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Evaluate every retained sample's mean function at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: Ignored: the draws are fixed.
            num_samples: Must be ``None``; all retained samples are used.

        Returns:
            The mean predictions, shaped ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is given.
        """
        del key
        if num_samples is not None:
            raise ValueError(
                "a finite posterior uses all its retained samples; "
                "num_samples must be None."
            )
        return jax.vmap(self.mean_function, in_axes=(0, None))(self.samples, inputs)


@dataclass
class PreconditionedSGLD:
    """RMSprop-preconditioned stochastic-gradient Langevin dynamics.

    Each [`update`][probreg.jax.PreconditionedSGLD.update] takes one step on
    the batch's estimate ``U`` of the negative log posterior, the problem's
    log-likelihood (already scaled to the full data set) plus its prior.
    With gradient ``g = ∇U``, the preconditioner tracks
    ``V ← decay · V + (1 − decay) · g²`` and ``G = 1 / (stability + √V)``,
    and the parameters move by ``−(step_size / 2) · G · g`` plus Gaussian
    noise of variance ``step_size · G``. The preconditioner's own drift
    correction is neglected, as is usual.

    The step is compiled once per [`init`][probreg.jax.PreconditionedSGLD.init];
    the log-likelihood sees each batch's inputs, targets and sample weights but
    not its metadata.

    Burn-in and thinning count update steps, i.e. training batches, not
    epochs. Steps are numbered from 1; the position after step ``t`` is
    retained when ``t = burn_in + k · thinning`` for some ``k ≥ 1``.

    SG-MCMC declares no early-stopping support, so a
    [`PosteriorStage`][probreg.jax.PosteriorStage] refuses an early stopper
    for it and writes only a finalized checkpoint, holding the stacked
    retained samples.

    Attributes:
        step_size: The Langevin step size ``ε``, positive and finite.
        burn_in: The number of initial update steps whose positions are
            discarded. Defaults to ``0``.
        thinning: Retain one position every ``thinning`` steps after burn-in.
            Defaults to ``1``, which retains every position.
        decay: The RMSprop decay of the squared-gradient average, in
            ``[0, 1)``. Defaults to ``0.99``.
        stability: Added to ``√V`` before inverting it, positive. Defaults to
            ``1e-5``.
    """

    step_size: float
    burn_in: int = 0
    thinning: int = 1
    decay: float = 0.99
    stability: float = 1e-5
    _problem: PosteriorProblem | None = field(default=None, init=False, repr=False)
    _step: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _parameters: PyTree = field(default=None, init=False, repr=False)
    _second_moment: PyTree = field(default=None, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)
    _samples: list[PyTree] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If ``step_size`` or ``stability`` is not positive and
                finite, ``burn_in`` is negative, ``thinning`` is not positive,
                or ``decay`` is outside ``[0, 1)``.
        """
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite.")
        if not math.isfinite(self.stability) or self.stability <= 0.0:
            raise ValueError("stability must be positive and finite.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive.")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1).")

    @property
    def supports_early_stopping(self) -> bool:
        """``False``: an early stopper would truncate the chain."""
        return False

    def init(self, problem: PosteriorProblem) -> None:
        """Start a chain at the problem's warm start, discarding any samples.

        Args:
            problem: The posterior problem.
        """
        self._problem = problem
        self._step = _make_step(
            problem,
            step_size=self.step_size,
            decay=self.decay,
            stability=self.stability,
        )
        self._parameters = problem.initial_parameters
        self._second_moment = jax.tree.map(jnp.zeros_like, problem.initial_parameters)
        self._steps = 0
        self._samples = []

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Take one Langevin step, retaining the new position if it is due.

        Args:
            batch: The training batch.
            key: The PRNG key of this step's injected noise.

        Returns:
            ``{"loss": U}``, the negative log posterior estimate at the
            position before the step.

        Raises:
            ValueError: If the method was not initialized.
        """
        if self._step is None:
            raise ValueError("call init before update.")
        self._parameters, self._second_moment, loss = self._step(
            self._parameters,
            self._second_moment,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            key,
        )
        self._steps += 1
        retained_after = self._steps - self.burn_in
        if retained_after > 0 and retained_after % self.thinning == 0:
            self._samples.append(self._parameters)
        return {"loss": loss}

    def posterior(self) -> RetainedSamplesPosterior:
        """Return the posterior of the samples retained so far.

        Returns:
            A finite posterior with one draw per retained sample.

        Raises:
            ValueError: If no sample has been retained yet, e.g. during
                burn-in.
        """
        return self._posterior_of(self.posterior_state())

    def state(self) -> PyTree:
        """Return the chain's full state, from which it can resume.

        Returns:
            The position, the preconditioner's squared-gradient average, the
            step count and the retained samples (stacked, possibly none).
        """
        return {
            "parameters": self._parameters,
            "second_moment": self._second_moment,
            "steps": jnp.asarray(self._steps),
            "samples": _stack(self._samples, like=self._parameters),
        }

    def load_state(self, state: PyTree) -> None:
        """Resume the chain from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][probreg.jax.PreconditionedSGLD.state].
        """
        self._parameters = state["parameters"]
        self._second_moment = state["second_moment"]
        self._steps = int(state["steps"])
        self._samples = _unstack(state["samples"])

    def posterior_state(self) -> PyTree:
        """Return the retained samples, stacked along a leading axis.

        Returns:
            The parameter tree with a leading sample axis.

        Raises:
            ValueError: If no sample has been retained yet.
        """
        if not self._samples:
            raise ValueError(
                "no sample has been retained yet; train beyond burn_in "
                f"({self.burn_in} steps) plus thinning ({self.thinning} steps)."
            )
        return _stack(self._samples, like=self._parameters)

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild the posterior of the given retained samples after ``init``.

        Args:
            state: Stacked samples previously returned by
                [`posterior_state`][probreg.jax.PreconditionedSGLD.posterior_state].
        """
        self._samples = _unstack(state)

    def _posterior_of(self, samples: PyTree) -> RetainedSamplesPosterior:
        if self._problem is None:
            raise ValueError("call init before reading the posterior.")
        return RetainedSamplesPosterior(
            mean_function=self._problem.mean_function, samples=samples
        )


def _make_step(
    problem: PosteriorProblem, *, step_size: float, decay: float, stability: float
) -> Callable[..., Any]:
    """Compile one pSGLD step for ``problem``."""

    def negative_log_posterior(parameters: PyTree, batch: Batch) -> jax.Array:
        return -(
            problem.log_likelihood(parameters, batch)
            + problem.prior.log_prob(parameters)
        )

    @jax.jit
    def step(
        parameters: PyTree,
        second_moment: PyTree,
        inputs: PyTree,
        targets: PyTree | None,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> tuple[PyTree, PyTree, jax.Array]:
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        loss, gradient = jax.value_and_grad(negative_log_posterior)(parameters, batch)
        second_moment = jax.tree.map(
            lambda v, g: decay * v + (1.0 - decay) * jnp.square(g),
            second_moment,
            gradient,
        )
        leaves, treedef = jax.tree.flatten(parameters)
        keys = jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))

        def move(
            theta: jax.Array, g: jax.Array, v: jax.Array, leaf_key: jax.Array
        ) -> jax.Array:
            preconditioner = 1.0 / (stability + jnp.sqrt(v))
            noise = jax.random.normal(
                leaf_key, jnp.shape(theta), jnp.result_type(theta)
            )
            return (
                theta
                - 0.5 * step_size * preconditioner * g
                + jnp.sqrt(step_size * preconditioner) * noise
            )

        parameters = jax.tree.map(move, parameters, gradient, second_moment, keys)
        return parameters, second_moment, loss

    return step


def _stack(samples: list[PyTree], *, like: PyTree) -> PyTree:
    """Stack parameter trees along a new leading axis, which may be empty."""
    if not samples:
        return jax.tree.map(lambda leaf: jnp.zeros((0, *jnp.shape(leaf))), like)
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)


def _unstack(stacked: PyTree) -> list[PyTree]:
    """Split a tree stacked by ``_stack`` back into its parameter trees."""
    count = jax.tree.leaves(stacked)[0].shape[0]
    return [jax.tree.map(lambda leaf, i=i: leaf[i], stacked) for i in range(count)]
