"""A concrete JAX-backed Gaussian predictive distribution and head.

This module binds :class:`probreg.core.distributions.PredictiveDistribution`
and :class:`probreg.core.distributions.DistributionHead` to JAX arrays and an
NNX linear head, following the same Tier 2 (JAX backend) placement as
:mod:`probreg.jax.state` and :mod:`probreg.jax.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from flax import nnx

from probreg.core.metric_registry import MetricInputs
from probreg.core.types import Batch


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

    def produce_metric_inputs(self, batch: Batch, /) -> MetricInputs:
        """Decode a batch into host-side metric inputs.

        Args:
            batch: A batch whose ``inputs`` are forwarded through this head.

        Returns:
            Materialized metric inputs with flattened ``targets``, ``mean``,
            and ``variance`` arrays, plus forwarded batch metadata.

        Raises:
            ValueError: If ``batch.targets`` is ``None``.
        """
        if batch.targets is None:
            raise ValueError("batch.targets must be provided for epoch metrics.")
        prediction = self(batch.inputs)
        return MetricInputs(
            targets=np.asarray(jax.device_get(batch.targets), dtype=float).reshape(-1),
            mean=np.asarray(jax.device_get(prediction.mean()), dtype=float).reshape(-1),
            variance=np.asarray(
                jax.device_get(prediction.variance()), dtype=float
            ).reshape(-1),
            metadata=batch.metadata,
        )
