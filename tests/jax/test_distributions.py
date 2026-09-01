from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probreg.jax.distributions import Gaussian, GaussianHead


def test_gaussian_log_prob_matches_analytic_normal_density() -> None:
    loc = jnp.array([0.0, 1.0])
    scale = jnp.array([1.0, 2.0])
    targets = jnp.array([0.0, 3.0])
    distribution = Gaussian(loc=loc, scale=scale)

    expected = (
        -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((targets - loc) ** 2) / scale**2
    )

    assert jnp.allclose(distribution.log_prob(targets), expected)


def test_gaussian_mean_and_variance() -> None:
    loc = jnp.array([1.0, -2.0])
    scale = jnp.array([0.5, 3.0])
    distribution = Gaussian(loc=loc, scale=scale)

    assert jnp.allclose(distribution.mean(), loc)
    assert jnp.allclose(distribution.variance(), scale**2)


def test_gaussian_batch_and_event_shape() -> None:
    distribution = Gaussian(loc=jnp.zeros((4, 2)), scale=jnp.ones((4, 2)))

    assert distribution.batch_shape == (4, 2)
    assert distribution.event_shape == ()


def test_gaussian_sample_shape_includes_sample_and_batch_shape() -> None:
    distribution = Gaussian(loc=jnp.zeros((3,)), scale=jnp.ones((3,)))

    samples = distribution.sample(jax.random.key(0), sample_shape=(5,))

    assert samples.shape == (5, 3)


def test_gaussian_sample_matches_reparametrized_normal() -> None:
    loc = jnp.array([2.0])
    scale = jnp.array([0.5])
    distribution = Gaussian(loc=loc, scale=scale)
    key = jax.random.key(0)

    sample = distribution.sample(key)
    expected = loc + scale * jax.random.normal(key, (1,))

    assert jnp.allclose(sample, expected)


def test_gaussian_head_produces_positive_scale_under_extreme_inputs() -> None:
    head = GaussianHead(1, 2, rngs=nnx.Rngs(0))
    extreme_features = jnp.array([[1e6], [-1e6]])

    prediction = head(extreme_features)

    assert prediction.loc.shape == (2, 2)
    assert prediction.scale.shape == (2, 2)
    assert bool(jnp.all(prediction.scale > 0.0))
    assert bool(jnp.all(jnp.isfinite(prediction.scale)))
    assert bool(jnp.all(jnp.isfinite(prediction.loc)))


def test_gaussian_head_rejects_non_positive_out_features() -> None:
    with pytest.raises(ValueError, match="out_features"):
        GaussianHead(1, 0, rngs=nnx.Rngs(0))
