"""Explicit JAX random-key utilities."""

from __future__ import annotations

import jax


def split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the next persistent key and one key for a single operation.

    Args:
        key: The current JAX PRNG key.

    Returns:
        A tuple ``(next_key, operation_key)`` where ``next_key`` should
        replace ``key`` for subsequent splits and ``operation_key`` is
        consumed by a single random operation.
    """
    next_key, operation_key = jax.random.split(key)
    return next_key, operation_key


# "net" in ASCII: the stream index of a posterior network's key.
_NETWORK_STREAM = 0x6E6574


def network_key(key: jax.Array) -> jax.Array:
    """Derive the key a posterior network's mean function is evaluated with.

    The derived key is independent of every key ``jax.random.split(key, n)``
    yields, so a method can keep using ``key`` exactly as before for its own
    randomness, e.g. a parameter sample, and its results for a deterministic
    posterior network stay bit-identical.

    Args:
        key: A JAX PRNG key the caller also uses for its own randomness.

    Returns:
        The key to pass to a posterior problem's mean function or
        log-likelihood.
    """
    return jax.random.fold_in(key, _NETWORK_STREAM)
