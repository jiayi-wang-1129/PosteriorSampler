from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from diffrax import Dopri8, ODETerm, SaveAt, Tsit5, diffeqsolve

from .diffusion import DiffusionSchedule
from .likelihood import GaussianLikelihoodSchedule, normalize_log_weights
from .noise_basis import generate_2d_white_level_noise, generate_eta_batch
from .priors import Prior


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the telescoping sampler."""

    t0: float = 0.01
    t1: float = 1.0
    n_steps: int = 99
    dt0: float = 0.01

    solver_level0: Literal["Tsit5", "Dopri8"] = "Tsit5"
    solver_increment: Literal["Tsit5", "Dopri8"] = "Dopri8"

    resampling: Literal["branching", "multinomial"] = "branching"


@dataclass
class SamplingResult:
    """Outputs from :meth:`TelescopingPosteriorSampler.sample`."""

    final_samples: jax.Array  # (N, d)

    trajectories_generated: List[jax.Array]  # each (T, N_level, d) before resampling
    trajectories_selected: List[jax.Array]  # each (T, N_level, d) after resampling
    unique_selected: List[jax.Array]  # each (T, N_unique, d)
    weights: List[jax.Array]  # each (N_level,)

    meta: Dict[str, object]


def _make_solver(name: str):
    if name == "Tsit5":
        return Tsit5()
    if name == "Dopri8":
        return Dopri8()
    raise ValueError(f"Unknown solver: {name}")


class TelescopingPosteriorSampler:
    """Telescoping multilevel posterior sampler (toy 2D version).

    The implementation follows the structure in your notebooks:
    - Level 1 (coarsest): reverse ODE drift + a *constant* noise coefficient.
    - Level n>=2: solve a refinement ODE for the increment X_n with
      a conditional score drift and a piecewise-constant level-n noise basis.
    - Each level does: propagate -> compute incremental importance weights
      -> resample (branching or multinomial).

    Notes:
        - This code is intended for 2D toy experiments.
        - The likelihood is a *schedule* over levels (Gaussian at each level),
          matching the notebooks' level-wise conditioning.
    """

    def __init__(
        self,
        *,
        prior: Prior,
        likelihood_schedule: GaussianLikelihoodSchedule,
        score_model: object,
        score_params: dict,
        diffusion_schedule: Optional[DiffusionSchedule] = None,
        config: Optional[SamplerConfig] = None,
    ) -> None:
        self.prior = prior
        self.likelihood_schedule = likelihood_schedule
        self.score_model = score_model
        self.score_params = score_params
        self.diffusion_schedule = diffusion_schedule or DiffusionSchedule()
        self.config = config or SamplerConfig()

        self.ts = jnp.linspace(self.config.t0, self.config.t1, self.config.n_steps + 1)
        self._solver0 = _make_solver(self.config.solver_level0)
        self._solver_inc = _make_solver(self.config.solver_increment)

    # ---------- score drift helpers ----------
    def score(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Convenience wrapper around `model.apply(params, t, x)`."""
        return self.score_model.apply(self.score_params, t, x)

    def _score_drift(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Notebook drift: (1/t)x + 2(1/t)s(1-t, x)."""
        t_safe = jnp.clip(t, 1e-6, 1.0)
        return (1.0 / t_safe) * x + 2.0 * (1.0 / t_safe) * self.score(1.0 - t_safe, x)

    def _conditional_drift(self, t: jax.Array, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Notebook conditional drift for increments.

        (1/t)x + 2(1/t) [ s(1-t, x + x0) - s(1-t, x0) ]
        """
        t_safe = jnp.clip(t, 1e-6, 1.0)
        s_full = self.score(1.0 - t_safe, x + x0)
        s_base = self.score(1.0 - t_safe, x0)
        return (1.0 / t_safe) * x + 2.0 * (1.0 / t_safe) * (s_full - s_base)

    # ---------- time-indexing helper ----------
    def _x0_at_time(self, t: jax.Array, x0_traj_bt: jax.Array) -> jax.Array:
        """Select x0(t) from a saved trajectory.

        Args:
            t: scalar in [t0, t1]
            x0_traj_bt: shape (batch, T, d)

        Returns:
            x0: shape (batch, d)
        """
        # Find the rightmost index such that ts[idx] <= t.
        idx = jnp.searchsorted(self.ts, t, side="right") - 1
        idx = jnp.clip(idx, 0, self.ts.shape[0] - 1)
        return jnp.take(x0_traj_bt, idx, axis=1)

    # ---------- resampling ----------
    def _resample(
        self,
        key: jax.Array,
        weights: jax.Array,
        traj: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Resample a trajectory along the particle axis.

        Args:
            key: PRNG key
            weights: shape (N,)
            traj: shape (T, N, d)

        Returns:
            traj_resampled: shape (T, N_new, d)
            traj_unique: shape (T, N_unique, d)
            new_key: PRNG key
        """
        if self.config.resampling == "multinomial":
            key, sk = random.split(key)
            n = int(traj.shape[1])
            idx = random.choice(sk, a=n, shape=(n,), replace=True, p=weights)
            traj_resampled = traj[:, idx, :]
            unique_idx = jnp.unique(idx)
            traj_unique = traj[:, unique_idx, :]
            return traj_resampled, traj_unique, key

        if self.config.resampling != "branching":
            raise ValueError(f"Unknown resampling: {self.config.resampling}")

        n = int(traj.shape[1])
        key, sk = random.split(key)
        u = random.uniform(sk, shape=(n,))
        counts = jnp.floor(weights * n + u).astype(jnp.int32)
        total = int(jax.device_get(jnp.sum(counts)))

        if total <= 0:
            # Extremely degenerate case: fall back to multinomial.
            key, sk2 = random.split(key)
            idx = random.choice(sk2, a=n, shape=(n,), replace=True, p=weights)
            traj_resampled = traj[:, idx, :]
            unique_idx = jnp.unique(idx)
            traj_unique = traj[:, unique_idx, :]
            return traj_resampled, traj_unique, key

        nonzero = jnp.where(counts > 0)[0]
        repeated = jnp.repeat(jnp.arange(n), counts)

        traj_resampled = traj[:, repeated, :]
        traj_unique = traj[:, nonzero, :]
        return traj_resampled, traj_unique, key

    # ---------- level solvers ----------
    def _solve_level0(self, key: jax.Array, n_particles: int) -> Tuple[jax.Array, jax.Array]:
        """Solve the coarse (level 1) reverse ODE."""
        key, k_noise, k_init = random.split(key, 3)
        d = self.prior.dim

        noise_const = random.normal(k_noise, shape=(n_particles, d))
        y0 = random.normal(k_init, shape=(n_particles, d))
        bs = n_particles

        def vf(t, y, args):
            (noise_const,) = args
            t_batched = t * jnp.ones((bs, 1))
            drift = self._score_drift(t_batched, y)
            return drift + noise_const

        sol = diffeqsolve(
            terms=ODETerm(vf),
            solver=self._solver0,
            t0=self.config.t0,
            t1=self.config.t1,
            dt0=self.config.dt0,
            y0=y0,
            args=(noise_const,),
            saveat=SaveAt(ts=self.ts),
        )
        return sol.ys, key

    def _solve_increment(self, key: jax.Array, *, n_level: int, x0_traj: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Solve an increment trajectory X_n, n>=2."""
        bs = int(x0_traj.shape[1])
        d = self.prior.dim

        y0 = jnp.zeros((bs, d), dtype=jnp.float32)

        key, kx, ky = random.split(key, 3)
        eta_x = generate_eta_batch(n_level, bs, kx)
        eta_y = generate_eta_batch(n_level, bs, ky)

        x0_traj_bt = jnp.transpose(x0_traj, (1, 0, 2))  # (bs, T, d)

        def vf(t, y, args):
            x0_traj_bt, eta_x, eta_y = args
            noise = generate_2d_white_level_noise(t, n_level, eta_x, eta_y)
            x0_t = self._x0_at_time(t, x0_traj_bt)
            t_batched = t * jnp.ones((bs, 1))
            drift = self._conditional_drift(t_batched, y, x0_t)
            return drift + noise

        sol = diffeqsolve(
            terms=ODETerm(vf),
            solver=self._solver_inc,
            t0=self.config.t0,
            t1=self.config.t1,
            dt0=self.config.dt0,
            y0=y0,
            args=(x0_traj_bt, eta_x, eta_y),
            saveat=SaveAt(ts=self.ts),
        )
        return sol.ys, key

    # ---------- public API ----------
    def sample(
        self,
        *,
        key: jax.Array,
        n_particles: int,
        n_levels: Optional[int] = None,
    ) -> SamplingResult:
        """Generate posterior samples.

        Args:
            key: JAX PRNGKey
            n_particles: initial number of particles at level 1
            n_levels: how many likelihood levels to use (defaults to schedule length)

        Returns:
            SamplingResult containing trajectories and final selected samples.
        """
        max_levels = self.likelihood_schedule.num_levels
        n_levels = max_levels if n_levels is None else int(n_levels)
        n_levels = max(1, min(n_levels, max_levels))

        traj_gen_list: List[jax.Array] = []
        traj_sel_list: List[jax.Array] = []
        unique_list: List[jax.Array] = []
        weights_list: List[jax.Array] = []

        # ---- level 0 (noise level 1) ----
        traj0, key = self._solve_level0(key, n_particles)
        x0_final = traj0[-1, :, :]
        logw0 = self.likelihood_schedule.log_unnormalized(0, x0_final)
        w0 = normalize_log_weights(logw0)

        traj0_sel, uniq0, key = self._resample(key, w0, traj0)

        traj_gen_list.append(traj0)
        traj_sel_list.append(traj0_sel)
        unique_list.append(uniq0)
        weights_list.append(w0)

        prev_traj = traj0_sel

        # ---- refinement levels ----
        for level_idx in range(1, n_levels):
            n_level = level_idx + 1  # noise resolution level (2,3,...)

            inc_traj, key = self._solve_increment(key, n_level=n_level, x0_traj=prev_traj)
            traj_gen = prev_traj + inc_traj

            x_prev_final = prev_traj[-1, :, :]
            x_new_final = traj_gen[-1, :, :]

            logw = self.likelihood_schedule.log_unnormalized(level_idx, x_new_final) - self.likelihood_schedule.log_unnormalized(
                level_idx - 1, x_prev_final
            )
            w = normalize_log_weights(logw)

            traj_sel, uniq, key = self._resample(key, w, traj_gen)

            traj_gen_list.append(traj_gen)
            traj_sel_list.append(traj_sel)
            unique_list.append(uniq)
            weights_list.append(w)

            prev_traj = traj_sel

        final_samples = prev_traj[-1, :, :]
        meta = {
            "n_levels_run": n_levels,
            "initial_n_particles": n_particles,
            "final_n_particles": int(prev_traj.shape[1]),
            "t0": self.config.t0,
            "t1": self.config.t1,
            "n_steps": self.config.n_steps,
            "resampling": self.config.resampling,
        }

        return SamplingResult(
            final_samples=final_samples,
            trajectories_generated=traj_gen_list,
            trajectories_selected=traj_sel_list,
            unique_selected=unique_list,
            weights=weights_list,
            meta=meta,
        )
