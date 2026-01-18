# Telescoping Posterior Sampling — Figure Guide (2D Toy Examples)

This document explains the figures produced by the telescoping posterior sampler
and clarifies the **color semantics** and **experimental setups** used in the toy 2D examples.

The goal of these experiments is to visualize how posterior mass is constructed
**coarse-to-fine** when likelihood information is injected incrementally through
a multilevel (telescoping) noise decomposition.

---

## Color convention (important)

Across **all figures and levels**, the same color scheme is used:

- **Orange dots**: generated particles at the current level  
  (before resampling / selection)

- **Green dots**: selected particles at that level  
  (particles with nonzero offspring after branching resampling)

- **Blue dots**: reference posterior samples  
  (ground-truth posterior samples, used only for visualization)

This color convention is consistent across levels and examples.

---

## Example 1: Swiss-roll prior

**Prior**  
The prior distribution is a 2D Swiss-roll (spiral) distribution.

**Likelihood**  
A Gaussian likelihood centered near the spiral,
with covariance proportional to the identity.

**What the figures show**

- At **early levels**, generated particles (orange) follow the coarse geometry
  of the Swiss-roll prior.
- As refinement levels increase, likelihood information is injected
  incrementally.
- The **selected particles** (green) increasingly concentrate along regions
  of the spiral that are compatible with the likelihood.
- By the final level, the selected particles closely align with the
  **true posterior samples** (blue).

This example highlights that the sampler:
- preserves complex prior geometry,
- and incorporates likelihood information without collapsing particles prematurely.

---

## Example 2: GMM prior with disjoint likelihood support

**Prior**  
A Gaussian mixture model (GMM) with multiple well-separated modes.

**Likelihood**  
A Gaussian likelihood with mean approximately `(-12, -12)` and covariance `3 * I_2`.

Importantly, the likelihood mass lies **far from the dominant support of the prior**.

**Why this case is challenging**

- A naive sampler would struggle because:
  - prior samples have negligible probability under the likelihood,
  - importance weights would collapse immediately.
- This is a stress test for posterior sampling when
  **prior and likelihood supports are nearly disjoint**.

**What the figures show**

- Early levels show generated particles (orange) still concentrated near the prior modes.
- As refinement proceeds, particles that are compatible with the likelihood
  are gradually identified and amplified.
- The **branching resampler** keeps rare but relevant particles alive.
- By later levels, selected particles (green) migrate toward the likelihood region
  and match the posterior reference samples (blue).

This demonstrates that telescoping likelihood injection can bridge
large prior–likelihood mismatches.

---

## Interpretation of levels

Each level `n` corresponds to:

- activation of finer noise modes in the diffusion,
- injection of an incremental log-likelihood contribution `ΔL_n`,
- an SMC-style update:
  
