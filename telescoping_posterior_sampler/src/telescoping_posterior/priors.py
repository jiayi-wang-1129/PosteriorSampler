from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import random


class Prior(Protocol):
    """Protocol for a prior distribution.

    The sampler only assumes:
    - `dim`: dimension of samples
    - `sample(key, n)`: draw `n` i.i.d. samples, shape (n, dim)
    """

    dim: int

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        """Return samples with shape (n, dim)."""
        ...


@dataclass(frozen=True)
class GMMPrior:
    """A simple 2D 4-component GMM-like prior.

    This matches the original notebook generator:
        binary in {0,1}^{2}
        means = 3*(binary - 0.5)  -> {-1.5, 1.5} per coordinate
        samples = means + noise_std * N(0, I)
    """

    noise_std: float = 0.4

    @property
    def dim(self) -> int:
        return 2

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        k1, k2 = random.split(key)
        binary = random.randint(k1, shape=(n, 2), minval=0, maxval=2)
        means = 3.0 * (binary.astype(jnp.float32) - 0.5)
        noise = self.noise_std * random.normal(k2, shape=(n, 2))
        return means + noise


@dataclass(frozen=True)
class SwissRollPrior:
    """A 2D 'Swiss roll' spiral prior.

    Notebook generator:
        theta ~ Unif(pi, 7pi)
        x = scale * theta * cos(theta)
        y = scale * theta * sin(theta)

    Note: This is a *2D spiral* (not the common 3D swiss-roll manifold).
    """

    theta_min: float = float(jnp.pi)
    theta_max: float = float(7.0 * jnp.pi)
    scale: float = 0.2

    @property
    def dim(self) -> int:
        return 2

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        theta = random.uniform(key, shape=(n,), minval=self.theta_min, maxval=self.theta_max)
        x = self.scale * theta * jnp.cos(theta)
        y = self.scale * theta * jnp.sin(theta)
        return jnp.stack([x, y], axis=-1)
