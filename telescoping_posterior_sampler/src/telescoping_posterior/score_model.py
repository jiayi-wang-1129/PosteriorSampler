from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLPScoreNet(nn.Module):
    """A small MLP used as a score network s_theta(t, x)."""

    hidden_dim: int = 256
    out_dim: int = 2

    @nn.compact
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Evaluate score.

        Args:
            t: shape (batch, 1) or broadcastable to that
            x: shape (batch, d)

        Returns:
            score: shape (batch, d)
        """
        h = jnp.concatenate([t, x], axis=-1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.swish(h)
        h = nn.Dense(self.out_dim)(h)
        return h
