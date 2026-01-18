# Telescoping Posterior Sampling — Figure Guide (2D Toy Examples)

This document explains the figures produced by the telescoping posterior sampler
and clarifies the **color semantics** and **experimental setups** used in the toy 2D examples.

The goal of these experiments is to visualize how posterior mass is constructed
**coarse-to-fine** when likelihood information is injected incrementally through
a multilevel (telescoping) noise decomposition.


## Swiss-roll prior (2D): level-wise particle evolution

**Legend**  
- **Orange**: generated particles (proposal at this level)  
- **Green**: selected particles after reweighting/resampling at this level  
- **Blue**: reference posterior samples (ground-truth / target)

**What the figures show**
- At **early levels**, generated particles (orange) follow the coarse geometry of the Swiss-roll prior.
- As refinement increases, likelihood information is injected **incrementally**.
- The **selected particles** (green) increasingly concentrate along spiral regions compatible with the likelihood.
- By the final level, selected particles closely align with the **true posterior** (blue).

<table>
  <tr>
    <td align="center"><b>n = 1</b><br>
      <img src="https://github.com/user-attachments/assets/8f8bfed2-bdfd-4584-b844-294c17b5561f" width="460">
    </td>
    <td align="center"><b>n = 2</b><br>
      <img src="https://github.com/user-attachments/assets/014518f7-7292-49b3-b5f0-d3594a4cb2e1" width="460">
    </td>
  </tr>
  <tr>
    <td align="center"><b>n = 3</b><br>
      <img src="https://github.com/user-attachments/assets/c452fc07-2ffc-4656-999d-aa74b32c41c2" width="460">
    </td>
    <td align="center"><b>n = 4</b><br>
      <img src="https://github.com/user-attachments/assets/bb291758-aacb-44fa-bd57-22d4e366846f" width="460">
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>n = 5</b><br>
      <img src="https://github.com/user-attachments/assets/dbd1c05f-53f7-4522-b467-f8510539dd93" width="460">
    </td>
  </tr>
</table>

---

## Example 2: GMM prior with disjoint likelihood support

**Prior**  
A Gaussian mixture model (GMM) with multiple well-separated modes.

**Likelihood**  
A Gaussian likelihood with mean approximately `(-12, -12)` and covariance `3 * I_2`.

Importantly, the likelihood mass lies **far from the dominant support of the prior**.

<table>
  <tr>
    <td align="center">
      <b>n = 1</b><br>
      <img src="https://github.com/user-attachments/assets/8f8bfed2-bdfd-4584-b844-294c17b5561f" width="480">
    </td>
    <td align="center">
      <b>n = 2</b><br>
      <img src="https://github.com/user-attachments/assets/014518f7-7292-49b3-b5f0-d3594a4cb2e1" width="480">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>n = 3</b><br>
      <img src="https://github.com/user-attachments/assets/c452fc07-2ffc-4656-999d-aa74b32c41c2" width="480">
    </td>
    <td align="center">
      <b>n = 4</b><br>
      <img src="https://github.com/user-attachments/assets/bb291758-aacb-44fa-bd57-22d4e366846f" width="480">
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <b>n = 5</b><br>
      <img src="https://github.com/user-attachments/assets/dbd1c05f-53f7-4522-b467-f8510539dd93" width="480">
    </td>
  </tr>
</table>


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
```text
propagate (activate finer noise modes) -> reweight with ΔL_n -> resample (branching)

  
## Convergence with refinement level

A key theoretical feature of the telescoping / multiscale construction is that truncating the driving
noise at level `n` yields an error that decays geometrically. For Haar/Schauder truncations, the
strong error scales as `2^{-n/2}` (see the discussion in the research statement on convergence under
truncation).

(E sup_{t ∈ [0,1]} || X(t) − X^{(n)}(t) ||^2)^{1/2} = O(2^{-n/2})


**Experiment:** the measured `W2` decreases with `n` and closely follows the predicted
`2^{−n/2}` scaling.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea9df891-e8e1-431f-be47-55bf89e854ee"
       width="500"
       alt="Wasserstein distance vs level (empirical vs theoretical 2^{-n/2})">
</p>

## Relation to the research statement

This repository contains a runnable refactor of the **completed project**
(“telescoping diffusion SDE + multilevel SMC”):
a coarse-to-fine posterior sampler with intermediate targets `Q_n`,
incremental reweighting, and branching-style resampling
to mitigate particle degeneracy.

It also includes the 2D toy experiments (Swiss-roll prior and skewed GMM prior)
used to visualize coarse-to-fine refinement and validate the expected
`2^{−n/2}` convergence trend under multiscale truncation.

## Ongoing work: prior generation at the coarsest level (comparison)

This section connects the completed multiscale posterior sampler with the **ongoing project**
described in the research statement.

All figures below focus on **prior-only generation** (no likelihood, no selection) and visualize
snapshots along a coarse time grid  
`t ∈ {1.0, 0.8, 0.6, 0.4, 0.2, 0.0}`.

**Legend (all plots in this section):**
- **Blue**: reference marginal samples (`noise_data`)
- **Orange**: generated samples (`gen_data`)

---

### (A) Level-0 sampler from the multiscale SDE construction

<p align="center">
  <img src="https://github.com/user-attachments/assets/9d054bf1-74c7-4627-82b9-3873b13d28a8" width="1000">
</p>


This figure corresponds to the **0th (coarsest) level** of the telescoping / multiscale SDE
used in the posterior sampler above.

Only the coarsest noise modes are activated, producing a global diffusion trajectory.
This level serves as the **base trajectory** on which finer multiscale corrections are built.

---

### (B) Standard diffusion sampler (baseline)

<p align="center">
  <img src="https://github.com/user-attachments/assets/43c9ad6f-93ff-4c4c-b9b3-2531b0f91d5b" width="1000">
</p>

This figure shows a **standard diffusion sampler** applied to the same prior,
using the full diffusion dynamics without multiscale decomposition.

It is included as a baseline for comparison with the multiscale construction
and the flow-matching approach.



---

### (C) Flow matching on the level-0 SDE (ongoing work)

<p align="center">
  <img src="https://github.com/user-attachments/assets/d03d11a6-ae63-4d0c-950b-63fdddd7e9d8" width="1000">
</p>


This figure shows **flow matching applied to the same level-0 SDE**.
Here, the drift is learned directly via flow matching rather than simulated
through a stochastic diffusion.

The generated trajectories appear particularly clean: the learned flow transports
samples smoothly between marginals and recovers the prior geometry accurately at final time.

## Next step (ongoing): coupling multiscale spatial modes + posterior tilting

A nontrivial next step is to move from **multilevel prior generation** to **multilevel posterior sampling** under
a flow-matching parameterization.

So far, I have obtained **flow-matching formulations for the prior-generation SDE at all refinement levels**.
This means the multiscale telescoping construction can be reproduced cleanly for the *prior* across scales.

What remains is the core posterior challenge: **couple the spatial modes across levels and perform posterior
fine tuning / tilting**.

Concretely, the goal is to:
- preserve the telescoping structure (coarse modes capture global structure; finer modes refine local detail), and
- inject likelihood information in a **coarse-to-fine** manner so that the learned dynamics are progressively
  tilted from the prior toward the posterior.

This coupling/tilting step is the main technical bottleneck when replacing SMC-style reweighting/resampling with a
fully amortized (flow-based) multiscale sampler.

