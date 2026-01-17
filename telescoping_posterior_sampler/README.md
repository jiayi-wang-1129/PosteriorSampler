# Telescoping Posterior Sampling (toy 2D examples)

This repo is a cleaned-up refactor of two original Jupyter notebooks:

- `PosNegativemeanEvolution.ipynb`
- `SwissPosEvolution(2,3).ipynb`

The goal is to generate **posterior samples** using a **pretrained score network for the prior**, and a **telescoping / multilevel noise decomposition** with SMC-style reweighting + resampling at each refinement level.

## What’s in this repo

- `src/telescoping_posterior/`
  - `priors.py`: GMM prior and 2D Swiss-roll (spiral) prior
  - `score_model.py`: Flax MLP score network
  - `training.py`: small denoising-score-matching trainer (optional)
  - `noise_basis.py`: piecewise-constant Haar/Schauder-style noise basis
  - `likelihood.py`: Gaussian likelihood schedule (level-wise)
  - `sampler.py`: `TelescopingPosteriorSampler` (main class)
  - `plotting.py`: plotting helpers
- `examples/`
  - `gmm_demo.py`: run the GMM posterior-sampling example
  - `swiss_roll_demo.py`: run the Swiss-roll posterior-sampling example
- `notebooks/original/`: the original notebooks (kept as-is)

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

> Note: JAX installation differs depending on CPU/GPU. If you want GPU support, follow the official JAX install instructions.

## Running the examples

From the repo root:

```bash
# Make the package importable
export PYTHONPATH=src

# GMM case
python examples/gmm_demo.py --train_steps 3000 --n_particles 2000 --n_levels 5

# Swiss-roll case
python examples/swiss_roll_demo.py --train_steps 3000 --n_particles 3000 --n_levels 5
```

Each script will:
1. Train a score network (unless you pass `--checkpoint`)
2. Run the telescoping sampler
3. Save plots to `outputs/.../`

### Using checkpoints

Train once and save params:

```bash
export PYTHONPATH=src
python examples/gmm_demo.py --train_steps 20000 --save_checkpoint checkpoints/gmm.msgpack
```

Then load them:

```bash
export PYTHONPATH=src
python examples/gmm_demo.py --checkpoint checkpoints/gmm.msgpack
```

## Notes

- This code is focused on 2D toy problems and is meant to be readable and easy to modify.
- The example likelihood schedules are **hard-coded** (taken from the notebooks), since they were already computed there.
- The default resampling strategy is the notebook’s **branching** resampler (particle count can change across levels). You can switch to `multinomial` in `SamplerConfig`.

## Repo structure (suggested)

If you publish to GitHub, consider adding:
- a `LICENSE`
- CI (black/ruff, unit tests)
- a short project description / figure in the README
