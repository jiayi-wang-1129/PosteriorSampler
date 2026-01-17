from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random


@dataclass(frozen=True)
class DiffusionSchedule:
    """Diffusion-time schedule helpers used in the original notebooks."""

    beta0: float = 0.1
    beta1: float = 20.0

    def log_alpha(self, t: jax.Array) -> jax.Array:
        # Matches the notebook definition.
        return -0.5 * t * self.beta0 - 0.25 * (t**2) * (self.beta1 - self.beta0)

    def log_sigma(self, t: jax.Array) -> jax.Array:
        # Matches the notebook definition.
        return 0.5 * jnp.log(2.0 * t - t**2)

    def alpha(self, t: jax.Array) -> jax.Array:
        return jnp.exp(self.log_alpha(t))

    def sigma(self, t: jax.Array) -> jax.Array:
        return jnp.exp(self.log_sigma(t))

    def q_t(self, key: jax.Array, x0: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward noising step used for score matching.

        Returns (eps, x_t) where eps ~ N(0, I) and
            x_t = alpha(t) * x0 + sigma(t) * eps
        """
        eps = random.normal(key, shape=x0.shape)
        x_t = self.alpha(t) * x0 + self.sigma(t) * eps
        return eps, x_t
