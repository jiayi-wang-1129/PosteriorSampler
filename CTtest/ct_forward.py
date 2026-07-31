"""Shared LoDoPaB-style parallel-beam CT forward model (Poisson transmission only).

This module is imported by *both* the SMC notebook and the baselines notebook so
that the two runs provably share the same posterior.  Nothing here reads
notebook globals: everything lives on a ``CTConfig`` / ``CTForward`` pair, and
``CTForward.fingerprint()`` returns a hash that both notebooks can assert on.

Model
-----
    a(x)     = s * link(g * (x + c))                 nonnegative attenuation
    z(x)     = P_Omega A_radon a(x)                  masked line integrals
    lambda   = N0 * exp(-z) + b                      Beer-Lambert transmission
    y_i      ~ Poisson(lambda_i)                     independent counts

The reward r(x) = sum_i [ y_i log lambda_i - lambda_i ] drops the log(y_i!)
constant.  Its gradient is obtained by autodiff through the projector, never by
an explicit adjoint.

Code in ``mask``/``sinogram``/``backproject`` is ported verbatim from the
baselines notebook so that observations are bit-identical across the two files.
"""

from __future__ import annotations

import hashlib
import json
import warnings
import math
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["CTConfig", "CTForward", "ct_config_from_env"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CTConfig:
    """All CT-specific settings.  Serialisable, hashable, notebook-independent."""

    img_size: int = 64

    # Ray-angle acquisition grid.
    num_angles: int = 180
    num_detectors: int = 128
    angle_max_degrees: float = 180.0

    # Photon statistics.
    n0_photons: float = 4096.0
    background_counts: float = 0.0

    # Ray mask on the (angle, detector) grid.
    # vd_poisson | random | full
    mask_type: str = "vd_poisson"
    mask_fraction: float = 0.8              # vd_poisson / random only
    mask_center_fraction: float = 0.12      # vd_poisson only
    mask_seed: int = 0

    # Attenuation link a(x) = scale * link(gain * (x + offset)).
    attenuation_link: str = "softplus"       # softplus | sigmoid | positive_shift | affine
    attenuation_scale: float = 0.05
    attenuation_offset: float = 2.0
    attenuation_gain: float = 1.0

    # Projector.
    pixel_size: Optional[float] = None       # default: 2.0 / img_size
    angle_chunk: int = 20

    # Numerics.
    exponent_clamp: float = 30.0
    log_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.pixel_size is None:
            self.pixel_size = 2.0 / float(self.img_size)
        self.mask_type = str(self.mask_type).strip().lower()
        self.attenuation_link = str(self.attenuation_link).strip().lower()

    # -- derived quantities ------------------------------------------------

    @property
    def full_grid_size(self) -> int:
        return int(self.num_angles) * int(self.num_detectors)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def ct_config_from_env(env_str, env_int, env_float, img_size: int) -> CTConfig:
    """Build a CTConfig from the notebook's existing env_* helpers.

    Keeps the CT_* environment-variable names used by the baselines notebook so
    that existing launch scripts keep working.
    """
    return CTConfig(
        img_size=int(img_size),
        num_angles=env_int("CT_NUM_ANGLES", 180),
        num_detectors=env_int("CT_NUM_DETECTORS", 128),
        angle_max_degrees=env_float("CT_ANGLE_MAX_DEGREES", 180.0),
        n0_photons=env_float("CT_N0_PHOTONS", 4096.0),
        background_counts=env_float("CT_BACKGROUND_COUNTS", 0.0),
        mask_type=env_str("CT_MASK_TYPE", "vd_poisson"),
        mask_fraction=env_float("CT_MASK_FRACTION", 0.8),
        mask_center_fraction=env_float("CT_MASK_CENTER_FRACTION", 0.12),
        mask_seed=env_int("CT_MASK_SEED", 0),
        attenuation_link=env_str("CT_ATTENUATION_LINK", "softplus"),
        attenuation_scale=env_float("CT_ATTENUATION_SCALE", 0.05),
        attenuation_offset=env_float("CT_ATTENUATION_OFFSET", 2.0),
        attenuation_gain=env_float("CT_ATTENUATION_GAIN", 1.0),
        pixel_size=env_float("CT_PIXEL_SIZE", 2.0 / float(img_size)),
        angle_chunk=env_int("CT_ANGLE_CHUNK", 20),
    )


# ---------------------------------------------------------------------------
# Ray masks  (ported verbatim from the baselines notebook)
# ---------------------------------------------------------------------------

def _vd_fraction_mask(H, W, keep_fraction, center_fraction, seed, device) -> torch.Tensor:
    """Variable-density ray mask with an *exact* target keep fraction."""
    H, W = int(H), int(W)
    keep_fraction = float(keep_fraction)
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"mask_fraction must be in (0,1], got {keep_fraction}")
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[-1:1:complex(H), -1:1:complex(W)]
    rr = np.sqrt(xx ** 2 + yy ** 2)
    weights = np.exp(-(rr / 0.50) ** 2).reshape(-1)
    weights = weights / np.maximum(weights.sum(), 1e-12)

    target = max(1, int(round(keep_fraction * H * W)))
    mask = np.zeros(H * W, dtype=bool)

    low_h = max(1, int(round(H * float(center_fraction))))
    low_w = max(1, int(round(W * float(center_fraction))))
    hs = H // 2 - low_h // 2
    ws = W // 2 - low_w // 2
    protected = np.zeros((H, W), dtype=bool)
    protected[hs:hs + low_h, ws:ws + low_w] = True
    mask[protected.reshape(-1)] = True

    current = int(mask.sum())
    if current < target:
        candidates = np.flatnonzero(~mask)
        p = weights[candidates]
        p = p / np.maximum(p.sum(), 1e-12)
        chosen = rng.choice(candidates, size=min(target - current, len(candidates)),
                            replace=False, p=p)
        mask[chosen] = True
    elif current > target:
        candidates = np.flatnonzero(mask & ~protected.reshape(-1))
        drop = rng.choice(candidates, size=min(current - target, len(candidates)),
                          replace=False)
        mask[drop] = False

    return torch.as_tensor(mask.reshape(H, W), dtype=torch.bool, device=device)


def _random_mask(H, W, keep_fraction, seed, device) -> torch.Tensor:
    H, W = int(H), int(W)
    rng = np.random.default_rng(int(seed))
    target = max(1, int(round(float(keep_fraction) * H * W)))
    flat = np.zeros(H * W, dtype=bool)
    flat[rng.choice(H * W, size=target, replace=False)] = True
    return torch.as_tensor(flat.reshape(H, W), dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

class CTForward:
    """Holds the mask + geometry and exposes the (r, grad r) oracle inputs."""

    def __init__(self, cfg: CTConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.mask = self._build_mask()
        self.obs_indices = torch.nonzero(self.mask.reshape(-1), as_tuple=False).flatten().to(device)
        self.obs_dim = int(self.mask.sum().item())
        self._angles = self._build_angles()
        self._z_ref: Optional[torch.Tensor] = None

    # -- construction ------------------------------------------------------

    def _build_mask(self) -> torch.Tensor:
        c, dev = self.cfg, self.device
        A, D = int(c.num_angles), int(c.num_detectors)
        mt = c.mask_type
        if mt in {"full", "none", "all"}:
            return torch.ones((A, D), dtype=torch.bool, device=dev)
        if mt in {"vd_poisson", "variable_density", "low_frequency", "lowfreq"}:
            return _vd_fraction_mask(A, D, c.mask_fraction, c.mask_center_fraction, c.mask_seed, dev)
        if mt in {"random", "uniform_random"}:
            return _random_mask(A, D, c.mask_fraction, c.mask_seed, dev)
        raise ValueError(f"Unknown mask_type={mt}; use vd_poisson, random, or full.")

    def _build_angles(self) -> torch.Tensor:
        max_angle = math.radians(float(self.cfg.angle_max_degrees))
        # endpoint=False analogue of np.linspace
        return torch.linspace(0.0, max_angle, int(self.cfg.num_angles) + 1,
                              device=self.device, dtype=torch.float32)[:-1]

    def angles(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return self._angles.to(device=device or self.device, dtype=dtype)

    # -- link a(x) ---------------------------------------------------------

    def attenuation(self, x: torch.Tensor) -> torch.Tensor:
        """Nonnegative attenuation image a(x); differentiable."""
        c = self.cfg
        if x.ndim == 3:
            x = x[:, None]
        t = float(c.attenuation_gain) * (x + float(c.attenuation_offset))
        link = c.attenuation_link
        if link == "softplus":
            return float(c.attenuation_scale) * F.softplus(t)
        if link == "affine":
            # Exactly-concave variant: r is then concave in x.  Relies on the
            # exponent clamp in rate() for the overflow guard.
            return float(c.attenuation_scale) * t
        raise ValueError(f"Unknown attenuation_link={link}")

    # -- projector ---------------------------------------------------------

    def sinogram(self, x: torch.Tensor, angle_chunk: Optional[int] = None) -> torch.Tensor:
        """Differentiable parallel-beam projector -> (B, num_angles, num_detectors).

        Rotates the attenuation image and sums one spatial axis.  Matrix-free
        and adequate for 64x64 synthetic work; verify with ``check_gradient``.
        """
        c = self.cfg
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, c.img_size, c.img_size)
        if x.ndim == 3:
            x = x[:, None]
        x = x.to(torch.float32)
        B, H, W = int(x.shape[0]), int(x.shape[-2]), int(x.shape[-1])
        n_detectors = int(c.num_detectors)
        chunk = int(angle_chunk if angle_chunk is not None else c.angle_chunk)
        angles = self.angles(device=x.device, dtype=x.dtype)

        atten = self.attenuation(x)
        outs = []
        for start in range(0, int(angles.numel()), chunk):
            th = angles[start:start + chunk]
            A = int(th.numel())
            cs, sn = torch.cos(th), torch.sin(th)
            mats = torch.zeros((A, 2, 3), device=x.device, dtype=x.dtype)
            mats[:, 0, 0] = cs
            mats[:, 0, 1] = -sn
            mats[:, 1, 0] = sn
            mats[:, 1, 1] = cs
            mats = mats[None].expand(B, A, 2, 3).reshape(B * A, 2, 3)

            imgs = atten[:, None].expand(B, A, 1, H, W).reshape(B * A, 1, H, W)
            grid = F.affine_grid(mats, size=imgs.shape, align_corners=False)
            rot = F.grid_sample(imgs, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)

            proj = rot.sum(dim=2).squeeze(1) * float(c.pixel_size)   # (B*A, W)
            if n_detectors != W:
                proj = F.interpolate(proj[:, None, :], size=n_detectors,
                                     mode="linear", align_corners=False).squeeze(1)
            outs.append(proj.reshape(B, A, n_detectors))
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """h(x) = P_Omega A a(x), shape (B, obs_dim).  Differentiable."""
        return self.sinogram(x)[:, self.mask].float()

    # -- transmission statistics ------------------------------------------

    def rate(self, line_integrals: torch.Tensor) -> torch.Tensor:
        """lambda = N0 exp(-z) + b, with the exponent clamped for safety."""
        c = self.cfg
        z = torch.clamp(line_integrals, min=-c.exponent_clamp, max=c.exponent_clamp)
        return float(c.n0_photons) * torch.exp(-z) + float(c.background_counts)

    def line_integrals_from_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """Approximate inverse z = -log((y-b)/N0).  Display/seed use only."""
        c = self.cfg
        y = torch.clamp(counts.to(torch.float32) - float(c.background_counts), min=1e-6)
        z = -torch.log(torch.clamp(y / max(float(c.n0_photons), 1e-12), min=1e-12))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # -- reward ------------------------------------------------------------

    def set_reference_from_counts(self, y_obs: torch.Tensor) -> None:
        """Set the centring reference directly from the observed counts.

        z_ref = -log(y/N0) puts the reference exactly at the data, which is the
        best available choice and needs no image.  Called automatically by
        ``simulate_observation``; call it yourself if you load y_obs from disk.
        """
        with torch.no_grad():
            self._z_ref = self.line_integrals_from_counts(y_obs).reshape(-1).detach()

    def set_reference(self, x_ref: torch.Tensor) -> None:
        """Fix the centering reference z_ref = h(x_ref) used by ``reward``.

        Only *differences* of r matter to SMC (an additive constant cancels in
        log w_n), so centering is free and it is the difference between a usable
        reward and a numerically meaningless one -- see ``reward``.
        """
        with torch.no_grad():
            self._z_ref = self.forward(x_ref.reshape(1, 1, self.cfg.img_size,
                                                     self.cfg.img_size))[0].detach()

    def reward(self, x: torch.Tensor, y_obs: torch.Tensor,
               normalize_by_obs_dim: bool = False,
               likelihood_strength: float = 1.0,
               likelihood_type: str = "poisson_direct",   # kept for API compat; must be this
               center: bool = True,
               accumulate_float64: bool = True,
               out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """r(x) = sum_i [ y_i log lambda_i - lambda_i ], batched over particles.

        Two numerical safeguards, both on by default and both necessary at
        m ~ 1e4 rays:

        ``center``  subtracts the constant r(x_ref) *analytically, per ray*.
            Uncentred, r is ~5e8 while the spread across particles is ~1e0-1e2,
            so in float32 the rounding error exceeds the signal by 10x and the
            SMC weights become noise.  Because lambda = N0 exp(-z), the centred
            integrand collapses to  y_i (z_ref,i - z_i) - N0 (e^-z_i - e^-z_ref,i),
            which is exact, not an approximation.

        ``accumulate_float64``  performs the reduction over rays in float64.
            The projector stays in float32; only the sum is promoted.
        """
        hx = self.forward(x)
        c = self.cfg
        y = y_obs[None, :]
        if accumulate_float64:
            hx, y = hx.double(), y.double()

        if likelihood_type == "poisson_direct":
            n0 = float(c.n0_photons)
            b = float(c.background_counts)
            z = torch.clamp(hx, min=-c.exponent_clamp, max=c.exponent_clamp)
            lam = n0 * torch.exp(-z) + b
            if center:
                if self._z_ref is None:
                    warnings.warn(
                        "CTForward.reward(center=True) called with no reference set; "
                        "falling back to z_ref=0. Call set_reference_from_counts(y_obs) "
                        "for full precision -- see the float32 note in the module "
                        "docstring.", RuntimeWarning, stacklevel=2)
                    self._z_ref = torch.zeros(self.obs_dim, device=hx.device,
                                              dtype=torch.float32)
                z_ref = self._z_ref.to(device=hx.device, dtype=hx.dtype)[None, :]
                lam_ref = n0 * torch.exp(-z_ref) + b
                if b == 0.0:
                    # exact: log lam - log lam_ref = z_ref - z
                    log_ratio = z_ref - z
                else:
                    log_ratio = torch.log(lam + c.log_eps) - torch.log(lam_ref + c.log_eps)
                per_coord = y * log_ratio - (lam - lam_ref)
            else:
                per_coord = y * torch.log(lam + c.log_eps) - lam

        else:
            raise ValueError(
                f"This module is Poisson-only; got likelihood_type={likelihood_type}."
            )

        agg = per_coord.mean(dim=1) if normalize_by_obs_dim else per_coord.sum(dim=1)
        return (float(likelihood_strength) * agg).to(out_dtype)

    # -- observation simulation -------------------------------------------

    @torch.no_grad()
    def simulate_observation(self, x_true: torch.Tensor, sample_counts: bool = True,
                             generator: Optional[torch.Generator] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (y_obs, lambda_true) for a single ground-truth image."""
        z_clean = self.forward(x_true)[0]
        lam = self.rate(z_clean)
        if sample_counts:
            y = torch.poisson(lam, generator=generator).to(torch.float32)
        else:
            y = torch.round(lam).to(torch.float32)
        self.set_reference_from_counts(y)   # centring is mandatory; never leave it unset
        return y, lam

    # -- backprojection seed ----------------------------------------------

    @torch.no_grad()
    def scatter_to_grid(self, values: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((int(self.cfg.num_angles), int(self.cfg.num_detectors)),
                           dtype=torch.float32, device=values.device)
        full[self.mask] = values.reshape(-1).to(torch.float32)
        return full

    @torch.no_grad()
    def backproject(self, sino_full: torch.Tensor, normalise: bool = True) -> torch.Tensor:
        """Unfiltered backprojection -> (B,1,H,W), robust-normalised to [-1,1]."""
        c = self.cfg
        if sino_full.ndim == 2:
            sino_full = sino_full[None]
        sino_full = sino_full.to(torch.float32)
        B, A, D = sino_full.shape
        H = W = int(c.img_size)
        angles = self.angles(device=sino_full.device, dtype=sino_full.dtype)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=sino_full.device, dtype=sino_full.dtype),
            torch.linspace(-1.0, 1.0, W, device=sino_full.device, dtype=sino_full.dtype),
            indexing="ij",
        )
        out = torch.zeros((B, H, W), dtype=sino_full.dtype, device=sino_full.device)
        chunk = int(c.angle_chunk)
        for start in range(0, A, chunk):
            th = angles[start:start + chunk]
            Ac = int(th.numel())
            cs = torch.cos(th)[:, None, None]
            sn = torch.sin(th)[:, None, None]
            det_coord = xx[None] * cs + yy[None] * sn
            grid = torch.stack([det_coord, torch.zeros_like(det_coord)], dim=-1)
            grid = grid[None].expand(B, Ac, H, W, 2).reshape(B * Ac, H, W, 2)
            inp = sino_full[:, start:start + Ac, :].reshape(B * Ac, 1, 1, D)
            vals = F.grid_sample(inp, grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)
            out = out + vals.reshape(B, Ac, H, W).sum(dim=1)

        out = out * (math.pi / max(float(A), 1.0))
        if not normalise:
            return out[:, None]
        # Robust affine map to [-1,1].  The old 1%/99.5% window clipped hard at
        # both ends, which is fine for the smooth unfiltered BP but destroys a
        # ramp-filtered image: FBP legitimately produces sharp over/undershoot
        # and negative streaks, and clipping them cost ~0.5 in correlation.
        flat = out.flatten(1)
        lo = torch.quantile(flat, 0.001, dim=1).view(B, 1, 1)
        hi = torch.quantile(flat, 0.999, dim=1).view(B, 1, 1)
        out = (out - lo) / torch.clamp(hi - lo, min=1e-6)
        return torch.clamp(2.0 * out - 1.0, -1.0, 1.0)[:, None]

    @torch.no_grad()
    def ramp_filter(self, sino: torch.Tensor, window: str = "hann") -> torch.Tensor:
        """Ramp-filter projections along the detector axis (the FBP filter).

        Plain backprojection reconstructs the object convolved with 1/|r|, which
        is why unfiltered BP looks badly blurred even when the data is complete
        and heavily overdetermined.  The ramp |w| undoes that.  ``window``
        tapers high frequencies, which matters because the ramp amplifies noise
        linearly and CT counts are noisy.
        """
        if sino.ndim == 2:
            sino = sino[None]
        D = int(sino.shape[-1])
        n = int(2 ** math.ceil(math.log2(max(2 * D, 2))))
        s = F.pad(sino.float(), (0, n - D))

        f = torch.fft.rfftfreq(n, device=sino.device)
        filt = 2.0 * f                                    # ramp
        if window == "hann":
            filt = filt * (0.5 + 0.5 * torch.cos(math.pi * f / f.max().clamp(min=1e-12)))
        elif window == "hamming":
            filt = filt * (0.54 + 0.46 * torch.cos(math.pi * f / f.max().clamp(min=1e-12)))
        elif window not in (None, "none", "ramp"):
            raise ValueError(f"Unknown window={window}")

        out = torch.fft.irfft(torch.fft.rfft(s, dim=-1) * filt, n=n, dim=-1)
        return out[..., :D]

    @torch.no_grad()
    def fbp(self, sino_full: torch.Tensor, window: str = "hann",
            normalise: bool = True) -> torch.Tensor:
        """Filtered backprojection: ramp filter, then backproject."""
        return self.backproject(self.ramp_filter(sino_full, window=window),
                                normalise=normalise)

    @torch.no_grad()
    def fbp_from_counts(self, y_count: torch.Tensor, window: str = "hann",
                        fill_missing: bool = True) -> torch.Tensor:
        """FBP reconstruction from Poisson counts on the observed rays.

        Unobserved rays are filled by interpolation along the detector axis
        rather than left at zero.  Zeros are not missing data to a ramp filter --
        they read as 'no attenuation here' and streak badly.
        """
        z = self.line_integrals_from_counts(y_count)
        full = torch.zeros((int(self.cfg.num_angles), int(self.cfg.num_detectors)),
                           dtype=torch.float32, device=y_count.device)
        full[self.mask] = z.reshape(-1).to(torch.float32)
        if fill_missing and not bool(self.mask.all()):
            full = self._fill_missing_rays(full)
        return self.fbp(full, window=window)

    @torch.no_grad()
    def _fill_missing_rays(self, full: torch.Tensor) -> torch.Tensor:
        """Linear interpolation across masked-out detector bins, per angle."""
        out = full.clone()
        m = self.mask
        A, D = out.shape
        cols = torch.arange(D, device=out.device, dtype=torch.float32)
        for a in range(A):
            known = m[a]
            if bool(known.all()) or not bool(known.any()):
                continue
            xk = cols[known]
            yk = out[a][known]
            idx = torch.searchsorted(xk, cols.clamp(xk[0], xk[-1])).clamp(1, len(xk) - 1)
            x0, x1 = xk[idx - 1], xk[idx]
            y0, y1 = yk[idx - 1], yk[idx]
            w = (cols.clamp(xk[0], xk[-1]) - x0) / (x1 - x0).clamp(min=1e-6)
            interp = y0 + w * (y1 - y0)
            out[a] = torch.where(known, out[a], interp)
        return out

    @torch.no_grad()
    def backproject_from_counts(self, y_count: torch.Tensor) -> torch.Tensor:
        z = self.line_integrals_from_counts(y_count)
        return self.backproject(self.scatter_to_grid(z))

    @torch.no_grad()
    def display_sinogram(self, y_vec: torch.Tensor, values_are_counts: bool = True) -> torch.Tensor:
        vals = torch.log1p(torch.clamp(y_vec.reshape(-1).float(), min=0.0)) \
            if values_are_counts else y_vec.reshape(-1).float()
        return self.scatter_to_grid(vals)

    # -- measurement-space residuals --------------------------------------

    @torch.no_grad()
    def measurement_residuals(self, x: torch.Tensor, y_obs: torch.Tensor) -> dict:
        """Per-sample residuals between predicted counts and the data.

        Image-space RMSE says how close a reconstruction is to a truth you will
        not have on real data.  These say whether a sample *explains the
        measurements*, which is checkable at run time and is the quantity the
        likelihood actually constrains.

        Because the noise is Poisson with a known rate, the residuals are
        self-normalising and calibrated:

            pearson_rms  sqrt(mean_i (y_i - lam_i)^2 / lam_i)
            chi2_per_ray mean_i (y_i - lam_i)^2 / lam_i        (= pearson_rms^2)
            deviance     2/m * sum_i [y_i log(y_i/lam_i) - (y_i - lam_i)]

        All three have expectation ~1 for a sample drawn from the true
        posterior.  Reading:

            ~ 1    explains the data at the noise level
            >> 1   underfits -- the sample is inconsistent with the counts
            << 1   overfits the noise realisation

        Also returned, unnormalised, for reference:
            sino_rmse    RMSE of line integrals z against z(x_true)-free data
            count_rmse   RMSE of predicted counts against y
        """
        if x.ndim == 3:
            x = x[:, None]
        y = y_obs.reshape(1, -1).double()
        z = self.forward(x).double()
        lam = self.rate(z).clamp(min=1e-12)

        resid = y - lam
        chi2 = (resid ** 2 / lam).mean(dim=1)

        # Poisson deviance; the y log(y/lam) term is 0 at y = 0 by convention.
        ylog = torch.where(y > 0, y * torch.log(y.clamp(min=1e-12) / lam),
                           torch.zeros_like(y))
        dev = (2.0 * (ylog - resid)).mean(dim=1)

        return dict(
            pearson_rms=torch.sqrt(chi2).float(),
            chi2_per_ray=chi2.float(),
            deviance_per_ray=dev.float(),
            count_rmse=torch.sqrt((resid ** 2).mean(dim=1)).float(),
            sino_rmse=torch.sqrt(((z - self.line_integrals_from_counts(y_obs)
                                   .reshape(1, -1).double()) ** 2).mean(dim=1)).float(),
        )

    @torch.no_grad()
    def residual_summary(self, x: torch.Tensor, y_obs: torch.Tensor) -> dict:
        """measurement_residuals reduced to scalars across the sample cloud."""
        r = self.measurement_residuals(x, y_obs)
        out = {}
        for k, v in r.items():
            out[f"{k}_mean"] = float(v.mean())
            if v.numel() > 1:
                out[f"{k}_std"] = float(v.std())
        return out

    # -- diagnostics -------------------------------------------------------

    @torch.no_grad()
    def regime_report(self, x_true: torch.Tensor, y_obs: torch.Tensor) -> dict:
        """The numbers the paper needs to substantiate a low-count claim."""
        z = self.forward(x_true)[0]
        lam = self.rate(z)
        a = self.attenuation(x_true)
        return {
            "obs_dim_m": self.obs_dim,
            "grid": (int(self.cfg.num_angles), int(self.cfg.num_detectors)),
            "observed_fraction": self.obs_dim / float(self.cfg.full_grid_size),
            "a_min": float(a.min()), "a_max": float(a.max()),
            "z_min": float(z.min()), "z_mean": float(z.mean()), "z_max": float(z.max()),
            "lambda_min": float(lam.min()), "lambda_mean": float(lam.mean()),
            "lambda_max": float(lam.max()),
            "frac_zero_counts": float((y_obs == 0).float().mean()),
            "frac_counts_below_10": float((y_obs < 10).float().mean()),
        }

    def check_gradient(self, x: torch.Tensor, y_obs: torch.Tensor,
                       eps: float = 1e-2, n_directions: int = 5,
                       **reward_kwargs) -> dict:
        """Finite-difference vs autodiff directional derivative of r.

        Two things this gets right that a naive check does not:

        * The finite difference is taken in **float64**.  In float32 this check
          is meaningless for CT -- one ulp of r exceeds the true directional
          difference, so a *correct* gradient reports 100% error.
        * It averages over several random directions and reports the median.
          A single unlucky draw nearly orthogonal to grad r gives a tiny |ad|
          and a large relative error even when the gradient is exact.
        """
        reward_kwargs.setdefault("center", True)
        reward_kwargs.setdefault("accumulate_float64", True)
        rels, pairs = [], []
        x_req = x.detach().clone().requires_grad_(True)
        r = self.reward(x_req, y_obs, out_dtype=torch.float64, **reward_kwargs)
        g = torch.autograd.grad(r.sum(), x_req)[0].double()
        for _ in range(int(n_directions)):
            d = torch.randn_like(x)
            d = d / (torch.linalg.vector_norm(d) + 1e-12)
            with torch.no_grad():
                fd = ((self.reward(x + eps * d, y_obs, out_dtype=torch.float64, **reward_kwargs)
                       - self.reward(x - eps * d, y_obs, out_dtype=torch.float64, **reward_kwargs))
                      / (2.0 * eps)).reshape(())
            ad = torch.sum(g * d.double())
            rels.append(float(torch.abs(fd - ad) / (torch.abs(fd) + torch.abs(ad) + 1e-12)))
            pairs.append((float(fd), float(ad)))
        rels_sorted = sorted(rels)
        return {"median_relative_error": rels_sorted[len(rels_sorted) // 2],
                "max_relative_error": rels_sorted[-1],
                "grad_norm": float(g.norm()),
                "pairs_fd_ad": pairs}

    # -- cross-notebook consistency ---------------------------------------

    def fingerprint(self, y_obs: Optional[torch.Tensor] = None,
                    x_true: Optional[torch.Tensor] = None) -> str:
        """Hash of config + mask + (optionally) the data.

        Assert this is equal in the SMC and baselines notebooks.  If it differs,
        the two are not sampling the same posterior and the comparison is void.
        """
        h = hashlib.sha256()
        h.update(self.cfg.to_json().encode())
        h.update(self.mask.detach().cpu().numpy().tobytes())
        if x_true is not None:
            h.update(np.ascontiguousarray(
                x_true.detach().cpu().numpy().astype(np.float32)).tobytes())
        if y_obs is not None:
            h.update(np.ascontiguousarray(
                y_obs.detach().cpu().numpy().astype(np.float32)).tobytes())
        return h.hexdigest()[:16]

    def summary(self) -> str:
        c = self.cfg
        return (
            f"CT parallel-beam | {c.num_angles} angles x {c.num_detectors} detectors "
            f"over [0,{c.angle_max_degrees}) deg\n"
            f"  mask={c.mask_type}  m={self.obs_dim}/{c.full_grid_size} "
            f"({self.obs_dim / c.full_grid_size:.3f})\n"
            f"  link={c.attenuation_link} (s,g,c)="
            f"({c.attenuation_scale},{c.attenuation_gain},{c.attenuation_offset})  "
            f"pixel_size={c.pixel_size:.5f}\n"
            f"  N0={c.n0_photons}  background={c.background_counts}"
        )
