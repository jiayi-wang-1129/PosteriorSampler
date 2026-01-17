from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _quad_form(x: jax.Array, cov: jax.Array) -> jax.Array:
    """Compute (x^T cov^{-1} x) for a batch of vectors.

    Args:
        x: shape (..., d)
        cov: shape (d, d)

    Returns:
        quadratic form: shape (...,)
    """
    # Solve cov * v = x for v, then x^T v.
    v = jnp.linalg.solve(cov, jnp.expand_dims(x, axis=-1)).squeeze(-1)
    return jnp.sum(x * v, axis=-1)


@dataclass(frozen=True)
class GaussianParams:
    """Gaussian parameters for a single refinement level."""

    mean: jax.Array  # (d,)
    cov: jax.Array  # (d, d)

    def log_unnormalized(self, x: jax.Array) -> jax.Array:
        """Return log of the *unnormalized* Gaussian density.

        log \tilde{N}(x; mean, cov) = -1/2 (x-mean)^T cov^{-1} (x-mean)

        Args:
            x: shape (..., d)
        """
        diff = x - self.mean
        return -0.5 * _quad_form(diff, self.cov)

    def logpdf(self, x: jax.Array) -> jax.Array:
        """Return log of the *normalized* Gaussian density."""
        d = int(self.mean.shape[-1])
        log_det = jnp.linalg.slogdet(self.cov)[1]
        log_norm = 0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det)
        return self.log_unnormalized(x) - log_norm


@dataclass(frozen=True)
class GaussianLikelihoodSchedule:
    """Level-wise Gaussian likelihood approximation.

    This matches how the original notebooks used different (mean, cov) pairs at
    different refinement levels.

    Typical usage in the sampler:
      - level 0: weight by L_0(x_0)
      - level n>0: incremental weight by exp(log L_n(x_n) - log L_{n-1}(x_{n-1}))
    """

    levels: Sequence[GaussianParams]

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError("GaussianLikelihoodSchedule.levels must be non-empty")

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def log_unnormalized(self, level: int, x: jax.Array) -> jax.Array:
        return self.levels[level].log_unnormalized(x)

    def logpdf(self, level: int, x: jax.Array) -> jax.Array:
        return self.levels[level].logpdf(x)


def normalize_log_weights(logw: jax.Array) -> jax.Array:
    """Turn log-weights into normalized weights (sum = 1) in a stable way."""
    logw = logw - logsumexp(logw)
    return jnp.exp(logw)
