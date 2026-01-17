from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def generate_eta_batch(n_level: int, batch_size: int, key: jax.Array) -> jax.Array:
    """Generate level-n coefficients for a batch.

    This matches how the notebooks generate `eta_x_batch = generate_eta_batch(max_n=n, ...)`.

    Args:
        n_level: refinement level (1,2,3,...). Level n uses 2^{n-1} coefficients.
        batch_size: number of particles
        key: JAX PRNG key

    Returns:
        eta: shape (batch_size, 2^{n_level-1})
    """
    if n_level < 1:
        raise ValueError(f"n_level must be >=1, got {n_level}")
    num_coeffs = 2 ** (n_level - 1)
    return random.normal(key, shape=(batch_size, num_coeffs))


def white_level(t: jax.Array, n_level: int, eta: jax.Array, *, eps: float = 1e-6) -> jax.Array:
    """Piecewise-constant 'white noise' for a given refinement level.

    This is a simplified (but equivalent) implementation of the notebook's
    `psi` + `white_level` logic: at level n, we have 2^{n-1} time bins on [0,1],
    pick the coefficient corresponding to the current bin index, and apply the
    same scaling pattern as in the notebooks.

    Args:
        t: scalar time in (0, 1] (can be a JAX tracer)
        n_level: refinement level
        eta: coefficients, shape (batch_size, 2^{n_level-1})
        eps: lower bound to avoid division by zero near t=0

    Returns:
        noise: shape (batch_size,)
    """
    t = jnp.clip(t, eps, 1.0)
    sqrt_factor = jnp.sqrt(2.0 / t)

    if n_level == 1:
        # Matches the notebook special case.
        return eta[:, 0] * sqrt_factor

    num_bins = 2 ** (n_level - 1)
    idx = jnp.clip((t * num_bins).astype(jnp.int32), 0, num_bins - 1)
    coeff = jnp.take(eta, idx, axis=1)  # (batch_size,)
    sign = jnp.where((idx % 2) == 0, 1.0, -1.0)

    # In the notebook implementation, the net per-level scaling for n>=2 collapses to 0.5.
    return 0.5 * sqrt_factor * sign * coeff


def generate_2d_white_level_noise(
    t: jax.Array,
    n_level: int,
    eta_x: jax.Array,
    eta_y: jax.Array,
) -> jax.Array:
    """Convenience wrapper to build 2D piecewise-constant noise."""
    noise_x = white_level(t, n_level, eta_x)
    noise_y = white_level(t, n_level, eta_y)
    return jnp.stack([noise_x, noise_y], axis=-1)
