"""Quantifying posterior-sampling quality.

Four axes, because no single number separates a good sampler from a bad one.
A method can ace any one of these while failing badly:

  1. DATA FIT       do the samples explain the counts, at the right level?
                    Target chi2_per_ray = 1, NOT 0.  Below 1 means the sample
                    has absorbed the noise realisation.
  2. PRIOR TYPICALITY   do the samples look like draws from the prior, or has
                    the likelihood dragged them off the learned manifold?
  3. DIVERSITY      is the cloud exploring the posterior, or is it N copies of
                    one MAP estimate?
  4. CALIBRATION    is the reported spread honest -- does posterior std match
                    actual error?  This is the one that catches a sampler that
                    looks good on 1-3 and is still overconfident.

Axes 1-4 are per-observation diagnostics.  For a decisive answer, use
``simulation_based_calibration``: repeat over many (x*, y) draws and check that
the rank of the truth among the samples is uniform.  That is the only test here
that can actually certify a sampler rather than fail to reject it.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Callable

import numpy as np
import torch


__all__ = ["data_fit", "prior_typicality", "diversity", "calibration",
           "posterior_report", "simulation_based_calibration", "format_report"]


# ---------------------------------------------------------------------------
# 1. Data fit  -- calibration against the known Poisson noise level
# ---------------------------------------------------------------------------

@torch.no_grad()
def data_fit(ct, samples: torch.Tensor, y_obs: torch.Tensor) -> Dict[str, float]:
    """Is the measurement residual at the level the noise model predicts?

    Under the true posterior, m * chi2_per_ray ~ chi2_m, so

        E[chi2_per_ray] = 1,   SD[chi2_per_ray] = sqrt(2/m).

    ``chi2_z`` expresses the deviation in units of that SD, so it is comparable
    across geometries.  |chi2_z| < 2 or so is consistent with correct sampling;
    large positive means underfitting, large negative means the sample has
    fitted the noise.
    """
    r = ct.measurement_residuals(samples, y_obs)
    chi2 = r["chi2_per_ray"].double()
    m = float(ct.obs_dim)
    sd = math.sqrt(2.0 / m)
    return {
        "chi2_per_ray": float(chi2.mean()),
        "chi2_z": float((chi2.mean() - 1.0) / sd),
        "chi2_spread_ratio": float(chi2.std() / sd) if chi2.numel() > 1 else float("nan"),
        "pearson_rms": float(r["pearson_rms"].mean()),
        "deviance_per_ray": float(r["deviance_per_ray"].mean()),
        "expected_sd": sd,
    }


# ---------------------------------------------------------------------------
# 2. Prior typicality -- are the samples on the learned manifold?
# ---------------------------------------------------------------------------

@torch.no_grad()
def prior_typicality(model, samples: torch.Tensor,
                     reference: torch.Tensor,
                     t_values=(0.1, 0.3, 0.5),
                     n_noise: int = 4,
                     seed: int = 0,
                     chunk_size: int = 16) -> Dict[str, float]:
    """Compare the DSM residual of the samples against genuine prior draws.

    A score model gives no normalised density, so instead we use the quantity it
    was trained to minimise.  For x on the data manifold the denoising residual
    at forward time t is small and predictable; for off-manifold x it is not.

    ``reference`` should be real draws from the training distribution (e.g.
    ``common.sample_cylinder_image``).  The returned z-score says how many
    reference standard deviations the samples sit above the reference mean.
    Near 0 is in-distribution; large positive means the likelihood has pulled
    the samples off the prior manifold.
    """
    device = samples.device
    g = torch.Generator(device=device).manual_seed(int(seed))

    def dsm_residual(x):
        """Mean squared score-matching residual per image, averaged over t.

        Evaluation is chunked because a 256/512-image score pass can exhaust
        one ROCm device even though gradients are disabled.
        """
        outs = []
        for start in range(0, int(x.shape[0]), int(chunk_size)):
            xb = x[start:start + int(chunk_size)]
            tot = torch.zeros(xb.shape[0], device=device, dtype=torch.float64)
            for t_val in t_values:
                for _ in range(int(n_noise)):
                    t = torch.full((xb.shape[0],), float(t_val), device=device)
                    sigma = torch.sqrt(torch.clamp(t * (2.0 - t), min=1e-8)).view(-1, 1, 1, 1)
                    mean = (1.0 - t).view(-1, 1, 1, 1) * xb
                    eps = torch.randn(xb.shape, device=device, generator=g)
                    xt = mean + sigma * eps
                    pred = model(xt, t)
                    target = -eps / sigma
                    sq = ((pred - target) ** 2) * sigma ** 2
                    tot += sq.flatten(1).mean(dim=1).double()
            outs.append(tot / (len(t_values) * int(n_noise)))
        return torch.cat(outs, dim=0)

    d_s = dsm_residual(samples)
    d_r = dsm_residual(reference)
    ref_mu, ref_sd = float(d_r.mean()), float(d_r.std().clamp(min=1e-12))
    return {
        "dsm_residual": float(d_s.mean()),
        "dsm_residual_reference": ref_mu,
        "dsm_z": (float(d_s.mean()) - ref_mu) / ref_sd,
        "range_violation": float((samples.abs() > 1.0).float().mean()),
    }


# ---------------------------------------------------------------------------
# 3. Diversity -- is the cloud exploring, or collapsed?
# ---------------------------------------------------------------------------

@torch.no_grad()
def diversity(samples: torch.Tensor, max_pairs: int = 4096,
              seed: int = 0) -> Dict[str, float]:
    """Spread and effective dimensionality of a sample cloud.

    Pairwise distances are subsampled rather than materialising an N x N
    distance matrix.  The covariance Gram matrix is evaluated on CPU float64;
    this avoids large/fragile ROCm double-precision ``cdist`` kernels.
    """
    n = int(samples.shape[0])
    if n < 2:
        return {k: float("nan") for k in
                ("mean_pairwise_rmse", "min_pairwise_rmse",
                 "participation_ratio", "n_effective",
                 "pixel_std", "distinct_fraction")}

    flat = samples.detach().reshape(n, -1).float().cpu()
    d = int(flat.shape[1])

    # Random unique-ish pairs; exhaustive when the cloud is small.
    total_pairs = n * (n - 1) // 2
    if total_pairs <= int(max_pairs):
        ii, jj = torch.triu_indices(n, n, offset=1)
    else:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        ii = torch.randint(0, n, (int(max_pairs) * 2,), generator=gen)
        jj = torch.randint(0, n, (int(max_pairs) * 2,), generator=gen)
        keep = ii != jj
        ii, jj = ii[keep][:int(max_pairs)], jj[keep][:int(max_pairs)]
    pair = torch.sqrt(torch.mean((flat[ii] - flat[jj]) ** 2, dim=1))

    # Covariance spectrum via N x N Gram matrix.
    c = (flat - flat.mean(0, keepdim=True)).double()
    gram = (c @ c.T) / max(n - 1, 1)
    ev = torch.linalg.eigvalsh(gram).clamp(min=0)
    s1, s2 = float(ev.sum()), float((ev ** 2).sum())
    pr = (s1 ** 2 / s2) if s2 > 1e-24 else 0.0

    # Exact/near duplicates, using a sparse rounded content signature.
    arr = flat.numpy()[:, ::97]
    keys = {tuple(np.round(r, 7)) for r in arr}
    return {
        "mean_pairwise_rmse": float(pair.mean()),
        "min_pairwise_rmse": float(pair.min()),
        "participation_ratio": pr,
        "n_effective": pr,
        "pixel_std": float(samples.detach().float().std(0, unbiased=False).mean().cpu()),
        "distinct_fraction": len(keys) / n,
    }


# ---------------------------------------------------------------------------
# 4. Calibration -- is the reported uncertainty honest?
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibration(samples: torch.Tensor, x_true: torch.Tensor,
                levels=(0.5, 0.9)) -> Dict[str, float]:
    """Does the posterior spread match the actual error?

    ``contraction_ratio`` = RMS(posterior sd) / RMS(mean - truth).  Near 1 is
    honest; well below 1 is overconfident (the usual failure); well above 1 is
    an uninformative posterior.  This is the single most diagnostic number here,
    because a sampler can look fine on data fit and diversity and still report
    a spread that has nothing to do with its error.

    Coverage is per-pixel and therefore optimistic -- it ignores spatial
    correlation -- but a large shortfall is still meaningful.
    """
    n = int(samples.shape[0])
    if n < 2:
        return {k: float("nan") for k in ("contraction_ratio", "coverage_50",
                                          "coverage_90", "rmse_of_mean")}
    mean = samples.mean(0, keepdim=True)
    sd = samples.std(0, keepdim=True)
    err = (mean - x_true)
    out = {
        "contraction_ratio": float(sd.pow(2).mean().sqrt() / err.pow(2).mean().sqrt().clamp(min=1e-12)),
        "rmse_of_mean": float(err.pow(2).mean().sqrt()),
    }
    for lv in levels:
        lo = torch.quantile(samples, (1 - lv) / 2, dim=0, keepdim=True)
        hi = torch.quantile(samples, 1 - (1 - lv) / 2, dim=0, keepdim=True)
        out[f"coverage_{int(lv*100)}"] = float(
            ((x_true >= lo) & (x_true <= hi)).float().mean())
    return out


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

@torch.no_grad()
def posterior_report(ct, model, samples, y_obs, x_true, prior_reference,
                     seed: int = 0) -> Dict[str, float]:
    out = {}
    out.update(data_fit(ct, samples, y_obs))
    out.update(prior_typicality(model, samples, prior_reference, seed=seed))
    out.update(diversity(samples))
    out.update(calibration(samples, x_true))
    return out


_TARGETS = [
    ("chi2_per_ray",        1.0,  "= 1  (< 1 means the noise was fitted)"),
    ("chi2_z",              0.0,  "|z| < ~3"),
    ("dsm_z",               0.0,  "~0  (large = off the prior manifold)"),
    ("participation_ratio", None, "large  (1.0 = collapsed)"),
    ("distinct_fraction",   1.0,  "-> 1"),
    ("contraction_ratio",   1.0,  "= 1  (< 1 = overconfident)"),
    ("coverage_90",         0.90, "-> 0.90"),
]


def format_report(reports: Dict[str, Dict[str, float]]) -> str:
    """Side-by-side table of the headline numbers with their targets."""
    names = list(reports)
    w = max(len(n) for n in names) + 2
    lines = [f"{'metric':<22}" + "".join(f"{n:>{w}}" for n in names) + "   target",
             "-" * (22 + w * len(names) + 12)]
    for key, _tgt, note in _TARGETS:
        row = f"{key:<22}"
        for n in names:
            v = reports[n].get(key, float("nan"))
            row += f"{v:>{w}.4g}"
        lines.append(row + f"   {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The decisive test: simulation-based calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulation_based_calibration(ct, sampler: Callable, prior_draw: Callable,
                                 n_trials: int = 20, n_samples: int = 32,
                                 statistic: Optional[Callable] = None,
                                 seed: int = 0) -> Dict[str, object]:
    """Rank statistic over repeated (x*, y) draws.

    The only test here that can certify rather than merely fail to reject.  If
    x* ~ prior, y ~ p(y|x*), and x_1..x_N ~ q(.|y), then when q is the true
    posterior the rank of T(x*) among {T(x_i)} is uniform on {0..N}.  Deviation
    is diagnostic:

        U-shaped   posterior too narrow (overconfident)
        peaked     posterior too wide
        skewed     biased

    Costs n_trials full sampler runs, so run it once at reduced resolution
    rather than at final settings.  ``statistic`` defaults to the image mean;
    a few different T's probe different failure modes.

    Returns ranks plus a chi-square uniformity p-value.
    """
    if statistic is None:
        def statistic(x):
            return x.reshape(x.shape[0], -1).mean(dim=1)

    ranks = []
    for k in range(int(n_trials)):
        x_star = prior_draw(seed + k)
        y_k, _ = ct.simulate_observation(
            x_star, generator=torch.Generator(device=x_star.device).manual_seed(seed + 1000 + k))
        xs = sampler(y_k, n_samples, seed + 2000 + k)
        t_star = float(statistic(x_star)[0])
        t_s = statistic(xs).cpu().numpy()
        ranks.append(int((t_s < t_star).sum()))

    ranks = np.asarray(ranks)
    n_bins = min(10, int(n_samples) + 1)
    hist, _ = np.histogram(ranks, bins=n_bins, range=(0, int(n_samples) + 1))
    expected = len(ranks) / n_bins
    chi2 = float(((hist - expected) ** 2 / max(expected, 1e-12)).sum())
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(1.0 - chi2_dist.cdf(chi2, n_bins - 1))
    except ImportError:
        p = float("nan")
    return {"ranks": ranks, "histogram": hist, "chi2": chi2,
            "p_value": p, "n_trials": len(ranks), "n_samples": int(n_samples)}
