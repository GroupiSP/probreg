"""A concrete JAX-backed Gaussian predictive distribution and head.

This module binds :class:`probreg.core.distributions.PredictiveDistribution`
and :class:`probreg.core.distributions.DistributionHead` to JAX arrays and an
NNX linear head, following the same Tier 2 (JAX backend) placement as
:mod:`probreg.jax.state` and :mod:`probreg.jax.evaluation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.stats as jstats
from flax import nnx


@dataclass(frozen=True)
class Gaussian:
    """A JAX-backed Gaussian predictive distribution parametrized by scale.

    Attributes:
        loc: The distribution mean, shaped like the model's output.
        scale: The distribution's positive standard deviation, broadcastable
            against ``loc``. Callers (e.g. :class:`GaussianHead`) are
            responsible for ensuring positivity, e.g. via a ``softplus``
            transform of an unconstrained raw output.
    """

    loc: jax.Array
    scale: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The broadcast shape of independent Gaussian components."""
        return jnp.broadcast_shapes(self.loc.shape, self.scale.shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of a single Gaussian event, always scalar."""
        return ()

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise Gaussian log-density of ``targets``.

        Args:
            targets: Target values broadcastable against ``loc``/``scale``.

        Returns:
            The elementwise log-density, shaped like the broadcast of
            ``targets`` against ``loc`` and ``scale``.
        """
        return jstats.norm.logpdf(targets, loc=self.loc, scale=self.scale)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw reparametrized samples from this distribution.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Samples shaped ``sample_shape + batch_shape``.
        """
        shape = sample_shape + self.batch_shape
        noise = jax.random.normal(key, shape)
        return self.loc + self.scale * noise

    def mean(self) -> jax.Array:
        """Return the distribution mean, i.e. ``loc``."""
        return self.loc

    def variance(self) -> jax.Array:
        """Return the elementwise variance, i.e. ``scale ** 2``."""
        return jnp.square(self.scale)


@dataclass(frozen=True)
class Gamma:
    """A JAX-backed Gamma distribution parametrized by shape and rate.

    Attributes:
        concentration: Positive Gamma shape parameter.
        rate: Positive Gamma rate parameter, i.e. the inverse scale.
    """

    concentration: jax.Array
    rate: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return the broadcast shape of independent Gamma components."""
        return jnp.broadcast_shapes(self.concentration.shape, self.rate.shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Return the shape of a single Gamma event, always scalar."""
        return ()

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise Gamma log-density of ``targets``.

        Args:
            targets: Positive target values broadcastable against the
                concentration and rate.

        Returns:
            Elementwise log-density values.
        """
        return (
            self.concentration * jnp.log(self.rate)
            - jsp.gammaln(self.concentration)
            + (self.concentration - 1.0) * jnp.log(targets)
            - self.rate * targets
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw keyed samples from the Gamma distribution.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Samples shaped ``sample_shape + batch_shape``.
        """
        shape = sample_shape + self.batch_shape
        concentration = jnp.broadcast_to(self.concentration, self.batch_shape)
        rate = jnp.broadcast_to(self.rate, self.batch_shape)
        return jax.random.gamma(key, concentration, shape=shape) / rate

    def mean(self) -> jax.Array:
        """Return the Gamma mean, i.e. ``concentration / rate``."""
        return self.concentration / self.rate

    def variance(self) -> jax.Array:
        """Return the Gamma variance, i.e. ``concentration / rate ** 2``."""
        return self.concentration / jnp.square(self.rate)


class GaussianHead(nnx.Module):
    """An NNX head producing a :class:`Gaussian` from model features.

    A single linear layer maps ``in_features`` to ``2 * out_features``
    outputs, split into an unconstrained location and an unconstrained scale
    that is passed through ``softplus`` (plus ``eps``) to guarantee a
    strictly positive standard deviation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the head's linear layer.

        Args:
            in_features: Number of input feature dimensions.
            out_features: Number of predicted target dimensions.
            rngs: NNX RNG collection used to initialize parameters.
            eps: A small positive constant added to the ``softplus``-mapped
                scale to keep it strictly positive.

        Raises:
            ValueError: If ``out_features`` is not positive.
        """
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        self.out_features = out_features
        self.eps = eps
        self.linear = nnx.Linear(in_features, 2 * out_features, rngs=rngs)

    def __call__(self, features: jax.Array) -> Gaussian:
        """Produce a :class:`Gaussian` prediction from ``features``.

        Args:
            features: Model features, shaped ``(..., in_features)``.

        Returns:
            A :class:`Gaussian` with ``loc``/``scale`` shaped
            ``(..., out_features)``.
        """
        raw_loc, raw_scale = jnp.split(self.linear(features), 2, axis=-1)
        scale = jax.nn.softplus(raw_scale) + self.eps
        return Gaussian(loc=raw_loc, scale=scale)


class GammaHead(nnx.Module):
    """An NNX head producing a shape/rate :class:`Gamma` distribution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the Gamma parameter projection.

        Args:
            in_features: Number of input feature dimensions.
            out_features: Number of predicted target dimensions.
            rngs: NNX RNG collection used to initialize parameters.
            eps: Positive finite offset added after ``softplus``.

        Raises:
            ValueError: If ``out_features`` or ``eps`` is invalid.
        """
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")
        self.out_features = out_features
        self.eps = eps
        self.linear = nnx.Linear(in_features, 2 * out_features, rngs=rngs)

    def __call__(self, features: jax.Array) -> Gamma:
        """Produce a Gamma prediction from model features.

        Args:
            features: Model features shaped ``(..., in_features)``.

        Returns:
            A Gamma distribution with positive concentration and rate shaped
            ``(..., out_features)``.
        """
        raw_concentration, raw_rate = jnp.split(
            self.linear(features),
            2,
            axis=-1,
        )
        concentration = jax.nn.softplus(raw_concentration) + self.eps
        rate = jax.nn.softplus(raw_rate) + self.eps
        return Gamma(concentration=concentration, rate=rate)
