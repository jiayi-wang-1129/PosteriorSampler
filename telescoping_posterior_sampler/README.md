# Telescoping Posterior Sampling (2D Toy Examples)

This repository is a cleaned and modular refactor of two original exploratory Jupyter notebooks:

- `PosNegativemeanEvolution.ipynb`
- `SwissPosEvolution(2,3).ipynb`

The goal is to generate **posterior samples** using a **pretrained score network for the prior** together with a
**telescoping (multilevel) noise decomposition** and **SMC-style reweighting and resampling** at each refinement level.

This code focuses on **2D toy problems** (GMM and Swiss-roll priors) and is intended to be **readable and easy to modify**
rather than optimized.

---

## Method overview

- We assume access to samples from a prior \(p(x)\).
- We train a score network \(s_\theta(t,x) \approx \nabla_x \log p_t(x)\) for the prior (once).
- We define a Gaussian likelihood (toy example), with an observation distribution
  \[
    y \sim \mathcal{N}(\mu,\, 3 I_2).
  \]
  The likelihood is injected across refinement levels via a precomputed decomposition \(L_n\) (incremental log-likelihoods \(\Delta L_n\)).
- Sampling proceeds from coarse to fine using a multilevel Haar/Schauder-style noise basis.
- At each level: propagate → reweight by \(\exp(\Delta L_n)\) → resample (default: branching).

---

## What’s in this repo

- `src/telescoping_posterior/`
  - `priors.py`: GMM prior and 2D Swiss-roll (spiral) prior
  - `score_model.py`: Flax MLP score network
  - `training.py`: denoising score-matching trainer (optional)
  - `noise_basis.py`: Haar/Schauder-style multilevel noise basis
  - `likelihood.py`: Gaussian likelihood schedule (level-wise)
  - `sampler.py`: `TelescopingPosteriorSampler` (main class)
  - `plotting.py`: plotting helpers
- `examples/`
  - `gmm_demo.py`: run the GMM posterior-sampling example
  - `swiss_roll_demo.py`: run the Swiss-roll posterior-sampling example
- `notebooks/original/`: the original notebooks (kept as-is)

---

## Installation

```bash
pip install -r requirements.txt
