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
