from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from probreg.jax.posterior import IsotropicGaussianPrior

_PRECISION = st.floats(min_value=0.0625, max_value=16.0)
_PARAMETERS = arrays(
    np.float64,
    st.integers(1, 6),
    elements=st.floats(min_value=-5.0, max_value=5.0),
)


def test_default_prior_is_the_standard_normal_density_at_the_origin() -> None:
    prior = IsotropicGaussianPrior()

    assert prior.precision == 1.0
    assert float(prior.log_prob({"w": jnp.zeros(())})) == pytest.approx(
        -0.5 * math.log(2.0 * math.pi)
    )


@settings(deadline=None, max_examples=25)
@given(precision=_PRECISION, values=_PARAMETERS, split=st.integers(0, 6))
def test_prior_density_does_not_depend_on_how_parameters_are_grouped(
    precision: float, values: np.ndarray, split: int
) -> None:
    prior = IsotropicGaussianPrior(precision=precision)
    flat = jnp.asarray(values)
    grouped = {"a": flat[:split], "b": {"c": flat[split:]}}

    assert float(prior.log_prob(grouped)) == pytest.approx(
        float(prior.log_prob(flat)), rel=1e-5, abs=1e-5
    )


@settings(deadline=None, max_examples=25)
@given(precision=_PRECISION, values=_PARAMETERS)
def test_prior_density_falls_with_the_precision_weighted_squared_norm(
    precision: float, values: np.ndarray
) -> None:
    prior = IsotropicGaussianPrior(precision=precision)
    parameters = jnp.asarray(values)

    drop = prior.log_prob(jnp.zeros_like(parameters)) - prior.log_prob(parameters)

    assert float(drop) == pytest.approx(
        0.5 * precision * float(np.sum(values**2)), rel=1e-4, abs=1e-4
    )


@pytest.mark.parametrize("precision", [0.0, -1.0, math.inf, math.nan])
def test_prior_refuses_a_precision_that_is_not_positive_and_finite(
    precision: float,
) -> None:
    with pytest.raises(ValueError, match="precision"):
        IsotropicGaussianPrior(precision=precision)
