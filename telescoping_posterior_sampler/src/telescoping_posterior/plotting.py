from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import numpy as np


def _to_numpy(x):
    """Convert JAX arrays (or arraylikes) to NumPy for matplotlib."""
    try:
        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def plot_final_samples(
    *,
    prior_samples,
    generated_samples,
    selected_samples,
    observation: Optional[Sequence[float]] = None,
    title: str = "",
    lims: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    savepath: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Scatter plot of prior vs generated vs selected samples.

    Args:
        prior_samples: (N,2)
        generated_samples: (M,2)
        selected_samples: (K,2)
        observation: optional (2,) point to plot as a star
        title: plot title
        lims: ((xmin,xmax),(ymin,ymax))
        savepath: optional .png path
        show: whether to call plt.show()
    """
    import matplotlib.pyplot as plt

    prior = _to_numpy(prior_samples)
    gen = _to_numpy(generated_samples)
    sel = _to_numpy(selected_samples)

    plt.figure(figsize=(8, 6))
    plt.scatter(prior[:, 0], prior[:, 1], alpha=0.25, label="prior")
    plt.scatter(gen[:, 0], gen[:, 1], alpha=0.25, label="generated")
    plt.scatter(sel[:, 0], sel[:, 1], alpha=0.45, label="selected")

    if observation is not None:
        obs = np.asarray(observation, dtype=float)
        plt.scatter(obs[0], obs[1], marker="*", s=250, label="observation")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    if lims is not None:
        (xmin, xmax), (ymin, ymax) = lims
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_level_particle_counts(
    *,
    selected_trajectories,
    savepath: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot number of particles after resampling vs refinement level."""
    import matplotlib.pyplot as plt

    counts = [int(traj.shape[1]) for traj in selected_trajectories]
    levels = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(levels, counts, marker="o")
    plt.xlabel("level")
    plt.ylabel("#particles after resampling")
    plt.grid(True)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
