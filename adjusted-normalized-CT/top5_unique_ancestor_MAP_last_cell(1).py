# ============================================================
# Top-five DISTINCT MAP particles for each SMC level (levels 1--5)
# One highest-reward particle per unique surviving ancestor/lineage.
# No particle is ever repeated to fill an empty MAP slot.
# ============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

NUM_MAP_PARTICLES = int(globals().get("NUM_MAP_PARTICLES", 5))
MAP_REWARD_CHUNK_SIZE = int(globals().get("MAP_REWARD_CHUNK_SIZE", 16))
MAP_DISTINCT_ATOL = float(globals().get("MAP_DISTINCT_ATOL", 1e-7))

if "smc_out" not in globals():
    raise NameError("smc_out not found. Run the SMC method first.")

# Avoid using `a or b` here: tensors cannot safely be converted to bool.
samples_by_level = None
samples_key = None
for key in ("samples_by_level", "level_samples", "particles_by_level"):
    value = smc_out.get(key, None)
    if value is not None:
        samples_by_level = value
        samples_key = key
        break

if samples_by_level is None:
    raise KeyError(
        "Could not find per-level samples in smc_out. "
        "Expected one of: 'samples_by_level', 'level_samples', or "
        "'particles_by_level'."
    )

ancestors_by_level = smc_out.get("ancestors_by_level", None)


def _as_image_batch(samples):
    """Convert samples to (N,1,H,W) without changing their device."""
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    if samples.ndim == 3:
        samples = samples[:, None, :, :]
    if samples.ndim != 4:
        raise ValueError(
            "Expected level samples with shape (N,H,W) or (N,1,H,W); "
            f"received {tuple(samples.shape)}."
        )
    return samples


def _map_content_key(image, atol=MAP_DISTINCT_ATOL):
    """Near-exact image signature used only to prevent duplicate displays."""
    array = image.detach().to(torch.float32).cpu().reshape(-1).numpy()
    if atol > 0:
        array = np.round(array / atol).astype(np.int64)
    return array.tobytes()


@torch.no_grad()
def _reward_in_chunks(samples, ct, y_obs, chunk_size=MAP_REWARD_CHUNK_SIZE):
    """Evaluate the same CT reward used for SMC without a large forward batch."""
    reward_parts = []
    reward_device = y_obs.device

    for sample_chunk in samples.split(max(1, int(chunk_size)), dim=0):
        reward_parts.append(
            ct.reward(sample_chunk.to(reward_device), y_obs)
            .detach()
            .to("cpu", dtype=torch.float64)
        )

    return torch.cat(reward_parts, dim=0)


@torch.no_grad()
def topk_map_from_unique_ancestors(
    samples,
    ct,
    y_obs,
    ancestor_ids=None,
    k=NUM_MAP_PARTICLES,
):
    """Rank particles by reward and retain at most one per unique lineage.

    The first encountered member of a lineage is its highest-reward member,
    because candidates are traversed in descending reward order. Exact or
    near-exact duplicate images are also rejected. If fewer than k distinct
    lineages/images exist, fewer than k particles are returned; no repetition
    or padding is performed.
    """
    samples = _as_image_batch(samples)
    n_candidates = int(samples.shape[0])
    if n_candidates == 0:
        raise ValueError("The level sample cloud is empty.")

    rewards = _reward_in_chunks(samples, ct, y_obs)
    order = torch.argsort(rewards, descending=True).tolist()

    ancestors = None
    if ancestor_ids is not None:
        ancestors = np.asarray(ancestor_ids).reshape(-1)
        if len(ancestors) != n_candidates:
            print(
                f"  Warning: ancestor_ids has length {len(ancestors)}, "
                f"but this level has {n_candidates} particles. "
                "Falling back to image-content distinctness."
            )
            ancestors = None

    selected_indices = []
    selected_ancestors = []
    seen_ancestors = set()
    seen_images = set()

    for candidate_index in order:
        ancestor = (
            int(ancestors[candidate_index])
            if ancestors is not None
            else None
        )

        # At most one selected particle from each surviving root lineage.
        if ancestor is not None and ancestor in seen_ancestors:
            continue

        # Also guarantee that the plotted images themselves are different.
        image_key = _map_content_key(samples[candidate_index])
        if image_key in seen_images:
            # Do not mark this ancestor as used: a lower-reward descendant from
            # the same lineage may still be a genuinely different image.
            continue

        selected_indices.append(candidate_index)
        selected_ancestors.append(ancestor)
        seen_images.add(image_key)
        if ancestor is not None:
            seen_ancestors.add(ancestor)

        if len(selected_indices) >= int(k):
            break

    if not selected_indices:
        # This should be unreachable for a nonempty cloud, but keeps the cell
        # robust to an unusual signature failure.
        selected_indices = [int(order[0])]
        selected_ancestors = [
            int(ancestors[order[0]]) if ancestors is not None else None
        ]

    sample_index = torch.as_tensor(
        selected_indices,
        dtype=torch.long,
        device=samples.device,
    )
    reward_index = torch.as_tensor(selected_indices, dtype=torch.long)

    meta = {
        "n_candidates": n_candidates,
        "n_requested": int(k),
        "n_selected": len(selected_indices),
        "n_distinct": len(selected_indices),
        "n_displayed": len(selected_indices),
        "repeated_single": False,
        "used_ancestor_labels": ancestors is not None,
        "n_unique_ancestors": (
            int(len(np.unique(ancestors))) if ancestors is not None else None
        ),
        "n_selected_ancestors": (
            int(len({a for a in selected_ancestors if a is not None}))
            if ancestors is not None
            else None
        ),
        "selected_indices": selected_indices,
        "selected_ancestor_ids": selected_ancestors,
    }

    return rewards[reward_index], samples[sample_index], meta


level_map_results = {}

print("=" * 96)
print("TOP-5 DISTINCT MAP PARTICLES FOR EACH SMC LEVEL")
print(f"Particle source: smc_out[{samples_key!r}] (post-resampling population)")
print("Selection rule: highest reward, at most one image per unique ancestor")
print("=" * 96)

for ell, level_samples in enumerate(samples_by_level, start=1):
    if level_samples is None:
        continue

    level_samples = _as_image_batch(level_samples)
    if int(level_samples.shape[0]) < 1:
        continue

    ancestor_ids = None
    if ancestors_by_level is not None and len(ancestors_by_level) >= ell:
        ancestor_ids = ancestors_by_level[ell - 1]

    rw, mp, meta = topk_map_from_unique_ancestors(
        level_samples,
        ct,
        y_obs,
        ancestor_ids=ancestor_ids,
        k=NUM_MAP_PARTICLES,
    )

    level_map_results[ell] = (rw, mp, meta)

    if meta["used_ancestor_labels"]:
        print(
            f"SMC level {ell:>2d}: selected {meta['n_selected']}/"
            f"{NUM_MAP_PARTICLES} distinct MAP particles from "
            f"{meta['n_unique_ancestors']} available unique ancestors "
            f"({meta['n_candidates']} stored particles)."
        )
    else:
        print(
            f"SMC level {ell:>2d}: selected {meta['n_selected']}/"
            f"{NUM_MAP_PARTICLES} content-distinct MAP particles from "
            f"{meta['n_candidates']} stored particles; ancestor labels "
            "were unavailable."
        )

    if meta["n_selected"] < NUM_MAP_PARTICLES:
        print(
            "   Fewer than five distinct eligible particles exist at this "
            "level; empty plot slots will remain blank (no repetition)."
        )

    for j in range(meta["n_selected"]):
        ancestor = meta["selected_ancestor_ids"][j]
        ancestor_text = f" | ancestor = {ancestor}" if ancestor is not None else ""
        print(
            f"   rank {j + 1:>2d} | reward = {float(rw[j]):.6g}"
            f"{ancestor_text}"
        )

# ------------------------------------------------------------
# Plot truth + up to five distinct MAP particles for each level
# ------------------------------------------------------------
if level_map_results:
    nrow = len(level_map_results)
    fig, ax = plt.subplots(
        nrow,
        NUM_MAP_PARTICLES + 1,
        figsize=(2.15 * (NUM_MAP_PARTICLES + 1), 2.45 * nrow),
        squeeze=False,
    )

    level_items = sorted(level_map_results.items(), key=lambda item: item[0])

    for row, (ell, (rw, mp, meta)) in enumerate(level_items):
        ax[row, 0].imshow(
            x_true[0, 0].detach().cpu(),
            cmap="gray",
            vmin=-1,
            vmax=1,
        )
        ax[row, 0].set_ylabel(
            f"SMC L{ell}\n{meta['n_selected']} distinct",
            fontsize=9,
        )
        ax[row, 0].set_title("truth" if row == 0 else "", fontsize=9)

        for rank in range(NUM_MAP_PARTICLES):
            axis = ax[row, rank + 1]

            if rank >= meta["n_selected"]:
                axis.set_visible(False)
                continue

            axis.imshow(
                mp[rank, 0].detach().cpu(),
                cmap="gray",
                vmin=-1,
                vmax=1,
            )

            ancestor = meta["selected_ancestor_ids"][rank]
            ancestor_line = (
                f"\nanc={ancestor}" if ancestor is not None else ""
            )
            axis.set_title(
                f"MAP {rank + 1}{ancestor_line}\nr={float(rw[rank]):.4g}",
                fontsize=8,
            )

        for axis in ax[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle(
        "Top distinct MAP particles by SMC level "
        "(one per surviving ancestor)",
        fontsize=11,
        y=1.002,
    )
    fig.tight_layout()

    if getattr(cfg, "save_figures", False):
        figure_path = OUT / "smc_levels_top5_unique_ancestor_map_particles.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        print("saved", figure_path)

    plt.show()
else:
    print("No per-level MAP results were produced.")
