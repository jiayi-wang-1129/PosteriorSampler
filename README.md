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


<table>
  <tr>
    <td align="center">
      <b>n = 1</b><br>
      <img src="https://github.com/user-attachments/assets/cda67c6e-521b-45f8-a925-febf96c3c6ce" width="480">
    </td>
    <td align="center">
      <b>n = 2</b><br>
      <img src="https://github.com/user-attachments/assets/a782175f-f815-468c-a2a4-c416c62edcce" width="480">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>n = 3</b><br>
      <img src="https://github.com/user-attachments/assets/4297ac34-86c9-40f6-b697-1a0bf8627106" width="480">
    </td>
    <td align="center">
      <b>n = 4</b><br>
      <img src="https://github.com/user-attachments/assets/cf31463d-2d75-4f57-9e56-eaf3220a9621" width="480"> 
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <b>n = 5</b><br>
      <img src="https://github.com/user-attachments/assets/041cf2e3-e85d-43a8-b135-94fd050cdd36" width="480">
    </td>
  </tr>
</table>





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

## Convergence in Wasserstein distance (W2)

We also track the empirical 2-Wasserstein distance `W2` between the **selected particle set** at refinement level `n`
and a reference set of **target posterior samples**.

**Theory:** the multilevel refinement predicts a convergence rate of the form

- `W2(n) = O(2^(-n/2))`  (equivalently: `W2(n) <= C * 2^(-n/2)` for some constant `C`)

**Experiment:** the measured `W2` decreases with `n` and closely follows the predicted `2^(-n/2)` scaling.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b1bd8a8e-d5a1-4b99-bca6-b53b39e2d060" width="500" alt="Wasserstein distance vs level (empirical vs theoretical 2^{-n/2})">
</p>
> In the plot, the blue curve shows the empirical `W2`, and the dashed curve shows the theoretical reference
> `C * 2^(-n/2)` for a fitted constant `C`.

  
