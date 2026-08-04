"""Multilevel terminal SMC on a score prior, wired to the CT Poisson reward.

Algorithm bodies are ported verbatim from the SMC notebook: the multiscale
Haar-type level expansion, the base/increment reverse-SDE simulators, the
first-order surrogate L_m, and branch/kill resampling. The only structural
change is that notebook globals became fields on ``SMCConfig``, and the reward
oracle is now the ``CTForward`` object rather than ``reward_mri``.

    S_m(t)   terminal partial sums, m = 0 .. num_smc_levels-1
    L_m(x)   = r(x) + c_m * eta_m * <grad r(x), mu_tail_m>
    log w_m  = L_m(S~_m) - L_{m-1}(S_{m-1})

Selection happens only at terminal time T1, matching the manuscript.
"""

from __future__ import annotations

import gc
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

__all__ = ["SMCConfig", "MultilevelSMC", "run_multilevel_smc_scoreprior",
           "load_tail_statistics_npz"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SMCConfig:
    img_size: int = 64

    # Reverse-SDE window.
    t0: float = 0.03      # REVERSE_T_START in your notebook, NOT 1e-3
    t1: float = 1.0
    dsm_t_min: float = 1e-3
    dsm_t_max: float = 0.98
    base_noise_mode: str = "notebook_constant"   # or "sqrt_2_over_t"
    sample_clamp: Optional[float] = 4.0

    # Multiscale level structure.
    num_smc_levels: int = 5
    num_ref_levels: int = 8
    n_steps_ref_by_level: List[int] = field(
        default_factory=lambda: [100, 120, 140, 160, 180, 200, 220, 240])
    n_steps_smc_by_level: Optional[List[int]] = None   # defaults to ref[:num_smc_levels]

    # Surrogate: none | inv_m | inv_sqrt_m
    surrogate_mode: str = "inv_m"
    tail_correction_extra_scale: float = 1.0

    # Tail statistics.
    n_ref_tail: int = 256
    tail_chunk_size: int = 32
    use_monotone_gap: bool = True
    tail_stats_path: str = ""      # "" -> recompute

    # Likelihood tempering.  At CT scale r is order 1e5-1e8 and raw weights
    # degenerate to ESS=1; see calibrate_likelihood_strength().
     # Exact full likelihood: no calibration and no global tempering.
    likelihood_strength: float = 1.0
    auto_calibrate_strength: bool = False

# Retained only for optional experiments in which calibration is
# explicitly re-enabled.
    target_ess_fraction: float = 0.5

    # Particles.
    n_particles: int = 256
    target_population: Optional[int] = None    # defaults to n_particles
    seed: int = 0
    surrogate_chunk_size: int = 8

    def __post_init__(self):
        if self.n_steps_smc_by_level is None:
            self.n_steps_smc_by_level = list(
                self.n_steps_ref_by_level[:self.num_smc_levels])
        if self.target_population is None:
            self.target_population = self.n_particles
        if len(self.n_steps_ref_by_level) < self.num_ref_levels:
            raise ValueError("n_steps_ref_by_level shorter than num_ref_levels")


# ---------------------------------------------------------------------------
# Small helpers (verbatim)
# ---------------------------------------------------------------------------

def make_generator(seed: int, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def randn_seeded(shape, seed: int, device, dtype=torch.float32):
    return torch.randn(shape, generator=make_generator(seed, device),
                       device=device, dtype=dtype)


def clear_device_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def trace_var_per_dim_from_sums(sum_x, sum_x2, n):
    mean = sum_x / float(n)
    mean2 = sum_x2 / float(n)
    return torch.clamp(mean2 - mean ** 2, min=0.0).mean()


def normalize_logweights_torch(logw: torch.Tensor) -> torch.Tensor:
    return torch.exp(logw - torch.logsumexp(logw, dim=0))


def ess_from_logweights_torch(logw: torch.Tensor) -> torch.Tensor:
    w = normalize_logweights_torch(logw)
    return 1.0 / torch.sum(w ** 2)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class MultilevelSMC:
    """Holds the score model, the CT oracle, and the level machinery."""

    def __init__(self, model: nn.Module, ct, y_obs: torch.Tensor,
                 cfg: SMCConfig, device):
        self.model = model.eval()
        self.ct = ct
        self.y_obs = y_obs
        self.cfg = cfg
        self.device = device
        self.mu_tail_by_level = None
        self.eta_by_level = None
        self.tail_info = None

    # -- score / drift (verbatim) -----------------------------------------

    @torch.no_grad()
    def score_apply(self, tau_forward, x):
        c = self.cfg
        if not torch.is_tensor(tau_forward):
            tau_forward = torch.full((x.shape[0],), float(tau_forward),
                                     device=x.device, dtype=x.dtype)
        elif tau_forward.ndim == 0:
            tau_forward = torch.full((x.shape[0],), float(tau_forward.item()),
                                     device=x.device, dtype=x.dtype)
        return self.model(x, torch.clamp(tau_forward.to(x.device, x.dtype),
                                         min=c.dsm_t_min, max=c.dsm_t_max))

    @torch.no_grad()
    def base_drift(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """Manuscript SDE drift (x + 2 score)/t; level noise is separate."""
        c = self.cfg
        t_safe = torch.clamp(torch.as_tensor(t, device=y.device, dtype=y.dtype), min=1e-4)
        score_time = torch.clamp(1.0 - t_safe, min=c.dsm_t_min, max=c.dsm_t_max)
        return (y + 2.0 * self.score_apply(score_time, y)) / t_safe

    @torch.no_grad()
    def increment_drift(self, t, y, x_base):
        return self.base_drift(t, x_base + y) - self.base_drift(t, x_base)

    # -- multiscale white-noise expansion (verbatim) ----------------------

    def psi_pair(self, t, n, k):
        d = 2.0 ** (-(n - 1))
        I0, I1, I2 = (k - 1) * d, k * d, (k + 1) * d
        scale = 2.0 ** ((n - 2) / 2.0)
        tf = float(t)
        val = torch.zeros((2,), device=self.device)
        if I0 < tf < I1:
            val[0] = scale
        elif I1 < tf < I2:
            val[1] = -scale
        return val

    def white_level_coeffs(self, t, n_level):
        if n_level == 1:
            return torch.ones((1,), device=self.device)
        vals = [self.psi_pair(t, n_level, 2 * kk + 1)
                for kk in range(2 ** (n_level - 2))]
        return torch.cat(vals, dim=0) * (2.0 ** (-(n_level) / 2.0))

    def generate_eta_tensor(self, max_n, batch_size, image_shape, seed):
        return randn_seeded((batch_size, 2 ** (max_n - 1), *image_shape),
                            seed=seed, device=self.device)

    def white_level_vector(self, t, n_level, eta_batch):
        t_safe = torch.clamp(torch.as_tensor(t, device=self.device,
                                             dtype=eta_batch.dtype), min=1e-4)
        coeffs = self.white_level_coeffs(float(t_safe), n_level).to(dtype=eta_batch.dtype)
        noise = torch.tensordot(coeffs, eta_batch, dims=([0], [1]))
        return noise * torch.sqrt(2.0 / t_safe)

    # -- path simulators (verbatim) ---------------------------------------

    @torch.no_grad()
    def interpolate_paths(self, paths, target_n_steps):
        old_n = int(paths.shape[0] - 1)
        target_n_steps = int(target_n_steps)
        if old_n == target_n_steps:
            return paths
        pos = torch.linspace(0.0, float(old_n), target_n_steps + 1, device=paths.device)
        lo = torch.floor(pos).long().clamp(0, old_n)
        hi = torch.ceil(pos).long().clamp(0, old_n)
        w = (pos - lo.to(dtype=paths.dtype)).view(-1, 1, 1, 1, 1)
        return (1.0 - w) * paths[lo] + w * paths[hi]

    @torch.no_grad()
    def simulate_base_level(self, seed, batch_size, n_steps):
        c = self.cfg
        n_steps = int(n_steps)
        H = c.img_size
        x = randn_seeded((batch_size, 1, H, H), seed=seed, device=self.device)
        omega0 = randn_seeded((batch_size, 1, H, H), seed=seed + 1, device=self.device)
        ts = torch.linspace(float(c.t0), float(c.t1), n_steps + 1, device=self.device)
        dt = float(c.t1 - c.t0) / float(n_steps)
        xs = [x.clone()]
        for k in range(n_steps):
            t = ts[k]
            drift = self.base_drift(float(t.item()), x)
            if c.base_noise_mode == "sqrt_2_over_t":
                noise = torch.sqrt(2.0 / torch.clamp(t, min=1e-4)) * omega0
            else:
                noise = omega0
            x = x + dt * (drift + noise)
            if c.sample_clamp is not None:
                x = torch.clamp(x, -c.sample_clamp, c.sample_clamp)
            xs.append(x.clone())
        return torch.stack(xs, dim=0)

    @torch.no_grad()
    def simulate_increment_level(self, seed, noise_level, current_paths, n_steps,
                                 return_base_on_grid=False):
        c = self.cfg
        n_steps = int(n_steps)
        batch_size = int(current_paths.shape[1])
        image_shape = tuple(current_paths.shape[2:])
        current_on_grid = self.interpolate_paths(current_paths, n_steps)
        eta_batch = self.generate_eta_tensor(noise_level, batch_size, image_shape, seed)
        y = torch.zeros((batch_size, *image_shape), device=self.device,
                        dtype=current_paths.dtype)
        ts = torch.linspace(float(c.t0), float(c.t1), n_steps + 1, device=self.device)
        dt = float(c.t1 - c.t0) / float(n_steps)
        ys = [y.clone()]
        for k in range(n_steps):
            t = ts[k]
            x_base = current_on_grid[k]
            drift = self.increment_drift(float(t.item()), y, x_base)
            noise = self.white_level_vector(float(t.item()), noise_level, eta_batch)
            y = y + dt * (drift + noise)
            if c.sample_clamp is not None:
                total = torch.clamp(x_base + y, -c.sample_clamp, c.sample_clamp)
                y = total - x_base
            ys.append(y.clone())
        inc_paths = torch.stack(ys, dim=0)
        return (inc_paths, current_on_grid) if return_base_on_grid else inc_paths

    @torch.no_grad()
    def simulate_prior_hierarchy(self, seed, batch_size, num_levels, n_steps_by_level):
        c = self.cfg
        steps = [int(s) for s in list(n_steps_by_level)]
        paths_by_level = []
        current_paths = self.simulate_base_level(seed, batch_size, n_steps=steps[0])
        paths_by_level.append(current_paths)
        for ell in range(1, int(num_levels)):
            inc_paths, current_on_grid = self.simulate_increment_level(
                seed + 1000 * ell, ell + 1, current_paths,
                n_steps=steps[ell], return_base_on_grid=True)
            current_paths = current_on_grid + inc_paths
            if c.sample_clamp is not None:
                current_paths = torch.clamp(current_paths, -c.sample_clamp, c.sample_clamp)
            paths_by_level.append(current_paths)
        return paths_by_level

    # -- surrogate ---------------------------------------------------------

    def tail_correction_multiplier(self) -> float:
    """Return c_l in

        L_l(x)
        =
        r(x)
        +
        c_l * eta_l * <grad r(x), mu_tail_l>.

    Mode meanings
    -------------
    none / unscaled:
        c_l = 1

    inv_m:
        c_l = 1 / m

    inv_sqrt_m:
        c_l = 1 / sqrt(m)

    off / no_tail:
        c_l = 0
    """
    c = self.cfg
    mode = str(c.surrogate_mode).strip().lower()
    m = float(self.ct.obs_dim)

    if mode in {"none", "unscaled"}:
        # none = inv_m * m = 1
        base = 1.0

    elif mode == "inv_m":
        base = 1.0 / m 

    elif mode == "inv_sqrt_m":
        base = 1.0 / math.sqrt(m)

    elif mode in {"off", "no_tail", "full_likelihood"}:
        base = 0.0

    else:
        raise ValueError(
            f"Unknown surrogate_mode={mode!r}. "
            "Use 'none', 'inv_m', 'inv_sqrt_m', or 'off'."
        )

    return float(c.tail_correction_extra_scale) * base

    def reward_and_grad(
    self,
    x,
    strength=None,
):
    """Full summed Poisson reward and its image gradient."""

    # Ignore any previously calibrated value.
    likelihood_strength = 1.0

    x = (
        x.detach()
        .requires_grad_(True)
    )

    with torch.enable_grad():
        reward = self.ct.reward(
            x,
            self.y_obs,

            # Sum over all independent Poisson coordinates.
            normalize_by_obs_dim=False,

            # No calibration or likelihood power.
            likelihood_strength=likelihood_strength,
        )

        gradient = torch.autograd.grad(
            reward.sum(),
            x,
        )[0]

    return (
        reward.detach(),
        gradient.detach(),
    )


   def L_surrogate(
    self,
    level: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the level-dependent likelihood surrogate.

    Requested unscaled mode:

        L_l(x)
        =
        r(x)
        +
        eta_l <grad r(x), mu_tail_l>.

    The base reward is the full summed Poisson log-likelihood.
    """
    mode = str(
        self.cfg.surrogate_mode
    ).strip().lower()

    # Optional explicit no-tail mode.
    if mode in {
        "off",
        "no_tail",
        "full_likelihood",
    }:
        values = []

        for xs in x.split(
            int(self.cfg.surrogate_chunk_size),
            dim=0,
        ):
            values.append(
                self.ct.reward(
                    xs,
                    self.y_obs,

                    # Full product likelihood.
                    normalize_by_obs_dim=False,

                    # No likelihood tempering.
                    likelihood_strength=1.0,
                )
            )

        return torch.cat(values, dim=0)

    # none, inv_m, and inv_sqrt_m all require real tail statistics.
    if (
        self.mu_tail_by_level is None
        or self.eta_by_level is None
    ):
        raise RuntimeError(
            "Tail statistics are required for "
            f"surrogate_mode={mode!r}, but were not loaded."
        )

    mu = self.mu_tail_by_level[level].to(
        device=x.device,
        dtype=x.dtype,
    )

    if mu.ndim == x.ndim - 1:
        mu = mu.unsqueeze(0)

    elif mu.ndim == x.ndim - 2:
        mu = (
            mu.unsqueeze(0)
            .unsqueeze(0)
        )

    eta = torch.as_tensor(
        self.eta_by_level[level],
        device=x.device,
        dtype=x.dtype,
    )

    correction_multiplier = torch.as_tensor(
        self.tail_correction_multiplier(),
        device=x.device,
        dtype=x.dtype,
    )

    values = []

    for xs in x.split(
        int(self.cfg.surrogate_chunk_size),
        dim=0,
    ):
        reward, reward_gradient = (
            self.reward_and_grad(xs)
        )

        tail_inner_product = torch.sum(
            reward_gradient * mu,
            dim=tuple(
                range(
                    1,
                    reward_gradient.ndim,
                )
            ),
        )

        values.append(
            reward
            + correction_multiplier
            * eta
            * tail_inner_product
        )

    return torch.cat(values, dim=0)
    # -- resampling (verbatim) --------------------------------------------

    def stochastic_round_resample(self, paths, logw, target_N, seed):
        """Unbiased branch/kill resampling of whole terminal paths."""
        N_current = int(paths.shape[1])
        w = normalize_logweights_torch(logw.detach())
        gen = make_generator(seed, self.device)
        U = torch.rand((N_current,), device=self.device, generator=gen)
        counts = torch.floor(float(target_N) * w + U).to(torch.int64)
        idx = torch.repeat_interleave(torch.arange(N_current, device=self.device), counts)
        if idx.numel() == 0:
            idx = torch.multinomial(w, num_samples=int(target_N), replacement=True)
        return paths[:, idx, ...], counts, idx

    # -- tail statistics ---------------------------------------------------

    @torch.no_grad()
    def build_tail_statistics(self, verbose=True):
        """Streaming Algorithm-2 tail statistics. R_m = S_ref - S_m."""
        c = self.cfg
        shape = (1, 1, c.img_size, c.img_size)
        z = lambda: torch.zeros(shape, dtype=torch.float64)
        sum_ref, sum_ref2 = z(), z()
        sum_level = [z() for _ in range(c.num_smc_levels)]
        sum_level2 = [z() for _ in range(c.num_smc_levels)]
        sum_tail = [z() for _ in range(c.num_smc_levels)]
        sum_tail2 = [z() for _ in range(c.num_smc_levels)]

        count = 0
        for start in range(0, int(c.n_ref_tail), int(c.tail_chunk_size)):
            bs = min(int(c.tail_chunk_size), int(c.n_ref_tail) - int(start))
            paths_ref = self.simulate_prior_hierarchy(
                seed=c.seed + 700000 + int(start), batch_size=bs,
                num_levels=c.num_ref_levels, n_steps_by_level=c.n_steps_ref_by_level)
            finals = [p[-1].detach().cpu().to(torch.float64) for p in paths_ref]
            x_ref = finals[-1]
            sum_ref += x_ref.sum(dim=0, keepdim=True)
            sum_ref2 += (x_ref ** 2).sum(dim=0, keepdim=True)
            for m in range(c.num_smc_levels):
                xm = finals[m]
                tail = x_ref - xm
                sum_level[m] += xm.sum(dim=0, keepdim=True)
                sum_level2[m] += (xm ** 2).sum(dim=0, keepdim=True)
                sum_tail[m] += tail.sum(dim=0, keepdim=True)
                sum_tail2[m] += (tail ** 2).sum(dim=0, keepdim=True)
            count += bs
            del paths_ref, finals, x_ref
            clear_device_cache()
            if verbose:
                print(f"  tail stats {count}/{c.n_ref_tail}", end="\r")

        var_ref = trace_var_per_dim_from_sums(sum_ref, sum_ref2, count)
        mu_tails, tail_vars, level_vars, gap_vars = [], [], [], []
        for m in range(c.num_smc_levels):
            mu_tails.append((sum_tail[m] / float(count)).to(torch.float32).detach())
            var_m = trace_var_per_dim_from_sums(sum_level[m], sum_level2[m], count)
            var_tail = trace_var_per_dim_from_sums(sum_tail[m], sum_tail2[m], count)
            level_vars.append(float(var_m.cpu()))
            tail_vars.append(float(var_tail.cpu()))
            gap_vars.append(float(torch.clamp(var_ref - var_m, min=0.0).cpu()))

        if c.use_monotone_gap:
            gap_vars = [float(v) for v in
                        np.minimum.accumulate(np.asarray(gap_vars, dtype=np.float64))]

        eta = [float(math.sqrt(max(gap_vars[m], 0.0) / max(tail_vars[m], 1e-12)))
               for m in range(c.num_smc_levels)]

        info = dict(eta=eta, tail_vars=tail_vars, level_vars=level_vars,
                    gap_vars=gap_vars, var_ref=float(var_ref.cpu()),
                    n_ref=int(count), num_smc_levels=int(c.num_smc_levels),
                    source="recomputed")
        return mu_tails, eta, info

    def prepare_tail_statistics(
    self,
    verbose=True,
):
    """Load the actual tail statistics for all tail-correction modes."""

    c = self.cfg
    mode = str(
        c.surrogate_mode
    ).strip().lower()

    # Only an explicit "off" mode skips the tail statistics.
    if mode in {
        "off",
        "no_tail",
        "full_likelihood",
    }:
        shape = (
            1,
            1,
            c.img_size,
            c.img_size,
        )

        self.mu_tail_by_level = [
            torch.zeros(
                shape,
                dtype=torch.float32,
            )
            for _ in range(
                c.num_smc_levels
            )
        ]

        self.eta_by_level = [
            0.0
            for _ in range(
                c.num_smc_levels
            )
        ]

        self.tail_info = {
            "source": "tail_disabled",
            "n_ref": 0,
        }

        return

    # none, inv_m, and inv_sqrt_m all load real mu and eta.
    if c.tail_stats_path:
        path = pathlib.Path(
            c.tail_stats_path
        ).expanduser()

        if path.exists():
            if verbose:
                print(
                    "loading tail stats:",
                    path,
                )

            mu, eta, info = (
                load_tail_statistics_npz(
                    path,
                    c.num_smc_levels,
                )
            )

            self.mu_tail_by_level = mu
            self.eta_by_level = eta
            self.tail_info = info

            return

        print(
            "tail stats not found, recomputing:",
            path,
        )

    start_time = time.time()

    mu, eta, info = (
        self.build_tail_statistics(
            verbose=verbose,
        )
    )

    if verbose:
        print(
            "\nbuilt tail statistics in "
            f"{time.time() - start_time:.1f}s"
        )

    self.mu_tail_by_level = mu
    self.eta_by_level = eta
    self.tail_info = info
    # -- main loop ---------------------------------------------------------

    def run(self, n_particles=None, verbose=True):
        c = self.cfg
        n_particles = int(n_particles or c.n_particles)
        if self.mu_tail_by_level is None:
            self.prepare_tail_statistics(verbose=verbose)
        if c.auto_calibrate_strength:
            self.calibrate_likelihood_strength(verbose=verbose)

        steps = [int(s) for s in c.n_steps_smc_by_level]
        out = dict(samples_by_level=[], ess_by_level=[], pop_by_level=[],
                   logw_by_level=[], unique_ancestors=[])

        if verbose:
            print("=" * 72)
            print("MULTILEVEL SMC  levels:", c.num_smc_levels, " steps:", steps)
            print("eta_by_level:", [f"{e:.4f}" for e in self.eta_by_level])
            print("likelihood_strength:", f"{c.likelihood_strength:.3e}")
            print("surrogate:", c.surrogate_mode,
                  " multiplier:", f"{self.tail_correction_multiplier():.3e}",
                  " m:", self.ct.obs_dim)
            print("=" * 72)

        current_paths = self.simulate_base_level(c.seed + 3000, n_particles,
                                                 n_steps=steps[0])
        # Persistent ancestor label per particle, carried through every
        # resampling.  Used by the top-k MAP display to avoid showing the same
        # lineage five times.
        ancestors = torch.arange(n_particles, device=self.device)
        L_new = self.L_surrogate(0, current_paths[-1])
        logw = L_new
        ess = ess_from_logweights_torch(logw)
        current_paths, counts, idx = self.stochastic_round_resample(
            current_paths, logw, c.target_population, seed=c.seed + 3100)
        ancestors = ancestors[idx]

        out.setdefault("ancestors_by_level", []).append(ancestors.cpu().numpy().copy())
        out["samples_by_level"].append(current_paths[-1].detach())
        out["ess_by_level"].append(float(ess))
        out["pop_by_level"].append(int(current_paths.shape[1]))
        out["logw_by_level"].append(logw.detach().cpu().numpy())
        out["unique_ancestors"].append(int(torch.unique(idx).numel()))
        if verbose:
            print(f"level 1/{c.num_smc_levels}: N={current_paths.shape[1]} "
                  f"ESS={float(ess):.1f}/{len(logw)} "
                  f"unique={int(torch.unique(idx).numel())}")

        for ell in range(1, c.num_smc_levels):
            inc_paths, current_on_grid = self.simulate_increment_level(
                seed=c.seed + 4000 + ell, noise_level=ell + 1,
                current_paths=current_paths, n_steps=steps[ell],
                return_base_on_grid=True)
            proposed = current_on_grid + inc_paths
            if c.sample_clamp is not None:
                proposed = torch.clamp(proposed, -c.sample_clamp, c.sample_clamp)

            L_new = self.L_surrogate(ell, proposed[-1])
            L_old = self.L_surrogate(ell - 1, current_paths[-1])
            logw = L_new - L_old
            ess = ess_from_logweights_torch(logw)
            proposed, counts, idx = self.stochastic_round_resample(
                proposed, logw, c.target_population, seed=c.seed + 4100 + ell)
            ancestors = ancestors[idx]
            current_paths = proposed.detach()

            out["ancestors_by_level"].append(ancestors.cpu().numpy().copy())
            out["samples_by_level"].append(current_paths[-1].detach())
            out["ess_by_level"].append(float(ess))
            out["pop_by_level"].append(int(current_paths.shape[1]))
            out["logw_by_level"].append(logw.detach().cpu().numpy())
            out["unique_ancestors"].append(int(torch.unique(idx).numel()))
            if verbose:
                print(f"level {ell+1}/{c.num_smc_levels}: N={current_paths.shape[1]} "
                      f"ESS={float(ess):.1f}/{len(logw)} "
                      f"unique={int(torch.unique(idx).numel())}")
            clear_device_cache()

        out["samples"] = out["samples_by_level"][-1]
        out["ancestors"] = out["ancestors_by_level"][-1]
        out["eta_by_level"] = list(self.eta_by_level)
        out["tail_info"] = self.tail_info
        return out


# ---------------------------------------------------------------------------
# npz I/O
# ---------------------------------------------------------------------------

def load_tail_statistics_npz(path, num_smc_levels: int):
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Tail-stats npz does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    if "mu_tail_by_level" not in data or "eta_by_level" not in data:
        raise KeyError(f"{path} must contain mu_tail_by_level and eta_by_level")
    mu_np = np.asarray(data["mu_tail_by_level"])
    eta_np = np.asarray(data["eta_by_level"], dtype=np.float64).reshape(-1)
    if mu_np.shape[0] < num_smc_levels or eta_np.shape[0] < num_smc_levels:
        raise ValueError(
            f"{path} has {mu_np.shape[0]} saved levels but num_smc_levels="
            f"{num_smc_levels}. Lower num_smc_levels or rebuild the stats.")
    mu = [torch.as_tensor(mu_np[m], dtype=torch.float32)
          for m in range(num_smc_levels)]
    eta = [float(e) for e in eta_np[:num_smc_levels]]
    info = {k: data[k] for k in data.files if k not in
            ("mu_tail_by_level", "eta_by_level")}
    info["source"] = "loaded_npz"
    info["path"] = str(path)
    return mu, eta, info


# ---------------------------------------------------------------------------
# Entry point matching the notebook's method contract
# ---------------------------------------------------------------------------

def run_multilevel_smc_scoreprior(model, ct, y_obs, n_particles=256, seed=0,
                                  cfg: Optional[SMCConfig] = None, verbose=True):
    """Returns a dict; ``result["samples"]`` is (N,1,H,W) at the finest level."""
    device = next(model.parameters()).device
    if cfg is None:
        cfg = SMCConfig(img_size=ct.cfg.img_size, n_particles=int(n_particles),
                        seed=int(seed))
    else:
        # n_particles from the call site wins; keep target_population tied to it
        # unless the caller deliberately set a different value.
        if cfg.target_population == cfg.n_particles:
            cfg.target_population = int(n_particles)
        cfg.n_particles = int(n_particles)
        cfg.seed = int(seed)
    smc = MultilevelSMC(model, ct, y_obs, cfg, device)
    return smc.run(n_particles=n_particles, verbose=verbose)
