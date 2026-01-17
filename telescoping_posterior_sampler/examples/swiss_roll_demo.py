from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

from telescoping_posterior import (
    GaussianParams,
    GaussianLikelihoodSchedule,
    MLPScoreNet,
    SamplerConfig,
    ScoreTrainingConfig,
    SwissRollPrior,
    TelescopingPosteriorSampler,
    load_checkpoint,
    save_checkpoint,
    train_score_model,
)
from telescoping_posterior.diffusion import DiffusionSchedule
from telescoping_posterior.plotting import plot_final_samples, plot_level_particle_counts


def make_swiss_likelihood_schedule() -> GaussianLikelihoodSchedule:
    """Hard-coded level-wise likelihood parameters extracted from the original notebook."""
    means = [
        jnp.array([1.8621224716636886, 1.8621224716636886]),
        jnp.array([1.8870118346357823, 1.8870118346357823]),
        jnp.array([1.9373337518069131, 1.9373337518069131]),
        jnp.array([1.9754287539425668, 1.9754287539425668]),
        jnp.array([1.9940995726930146, 1.9940995726930146]),
    ]
    diag = [
        2.8296915828537745,
        2.9069127592439767,
        2.9301728830258558,
        2.9268392651819717,
        2.9760934081544934,
    ]
    levels = [
        GaussianParams(mean=m, cov=jnp.diag(jnp.array([v, v])))
        for m, v in zip(means, diag)
    ]
    return GaussianLikelihoodSchedule(levels=levels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_particles", type=int, default=3000)
    parser.add_argument("--n_levels", type=int, default=5)

    # score model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_checkpoint", type=str, default="")

    parser.add_argument("--out_dir", type=str, default="outputs/swiss_roll")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = SwissRollPrior(scale=0.2)
    likelihood_schedule = make_swiss_likelihood_schedule()
    diffusion_schedule = DiffusionSchedule()

    model = MLPScoreNet(hidden_dim=args.hidden_dim, out_dim=prior.dim)

    # ----- load or train score model -----
    if args.checkpoint:
        dummy_t = jnp.ones((1, 1), dtype=jnp.float32)
        dummy_x = jnp.ones((1, prior.dim), dtype=jnp.float32)
        template = model.init(random.PRNGKey(0), dummy_t, dummy_x)
        params = load_checkpoint(args.checkpoint, template)
    else:
        train_cfg = ScoreTrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_steps=args.train_steps,
            seed=args.seed,
            record_loss=False,
        )
        state, _losses = train_score_model(
            prior=prior,
            model=model,
            schedule=diffusion_schedule,
            config=train_cfg,
        )
        params = state.params
        if args.save_checkpoint:
            save_checkpoint(args.save_checkpoint, params)

    # ----- run telescoping sampler -----
    sampler = TelescopingPosteriorSampler(
        prior=prior,
        likelihood_schedule=likelihood_schedule,
        score_model=model,
        score_params=params,
        diffusion_schedule=diffusion_schedule,
        config=SamplerConfig(resampling="branching"),
    )

    key = random.PRNGKey(args.seed + 123)
    result = sampler.sample(key=key, n_particles=args.n_particles, n_levels=args.n_levels)

    prior_key = random.PRNGKey(args.seed + 999)
    prior_samples = prior.sample(prior_key, 3000)

    gen_final = result.trajectories_generated[-1][-1, :, :]
    sel_final = result.trajectories_selected[-1][-1, :, :]

    observation = (2.0, 2.0)

    plot_final_samples(
        prior_samples=prior_samples,
        generated_samples=gen_final,
        selected_samples=sel_final,
        observation=observation,
        title=f"Swiss-roll prior | telescoping posterior sampling (levels={result.meta['n_levels_run']})",
        savepath=out_dir / "final_samples.png",
        show=True,
    )

    plot_level_particle_counts(
        selected_trajectories=result.trajectories_selected,
        savepath=out_dir / "particle_counts.png",
        show=True,
    )

    print("Done.")
    print("Saved:")
    print(" -", out_dir / "final_samples.png")
    print(" -", out_dir / "particle_counts.png")
    print("Meta:", result.meta)


if __name__ == "__main__":
    main()
