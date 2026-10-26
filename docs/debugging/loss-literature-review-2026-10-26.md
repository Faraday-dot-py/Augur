# Loss-function / error-characterisation literature review (2026-10-26)

Scope: losses for learned simulators (scatter-field gravity: particles -> grid U-Net -> per-particle acceleration, Verlet rollout, k-step unrolled squared position loss + 0.1*velocity). Failure: systematic energy-work bias in dense regions after tiling. Already failed: Euclid (non-squared) position loss, energy-similarity term, force-anchor aux, closed-loop relabel curriculum, Huber.

Method: every entry below was fetched from its arXiv abstract page this session. Tag VERIFIED primary = abstract page opened and read. Only abstract-level content is claimed; full-text details (loss formulas, ablations) were NOT read, so "relevance" lines are my inference and are marked as such. Fetch results are summaries from a small model over the abstract page, so wording is paraphrase. Two search/limit interruptions occurred; categories with no verified source are listed under "Gaps" and carry no claims.

## 1. Noise injection / rollout stability

- GNS, Sanchez-Gonzalez et al., arXiv:2002.09405. VERIFIED primary. Abstract names training-data noise corruption as a key factor for long-term rollout accuracy (reduces error accumulation), alongside number of message-passing steps. Relevance (inference): noise on inputs targets compounding position/velocity error, not a one-sided force bias; a systematic work bias in dense regions would not obviously be fixed by zero-mean noise.
- MeshGraphNets, Pfaff et al., arXiv:2010.03409. VERIFIED primary. Mesh-based GNN simulators with adaptive resolution, 10-100x faster than the simulator. Abstract page gave nothing on loss design; no relevance claim.
- Learned coarse turbulence models, arXiv:2112.15275 ("Learned Coarse Models for Efficient Turbulence Simulation"). VERIFIED primary. Training noise and temporal downsampling improve stability; generalisation beyond training data remains hard. Relevance: coarse-grid learned models resemble our grid field; supports noise as stabiliser only.
- Brandstetter et al., Message Passing Neural PDE Solvers, arXiv:2202.03376. VERIFIED primary (abstract). Treats autoregressive zero-stability as a domain-adaptation problem. The abstract page did not mention the pushforward trick or noise by name; pushforward details are SUMMARY-level/unverified here and excluded. Relevance (inference): our closed-loop relabel curriculum is of the same family (train on own rollout distribution) and already failed.
- PDE-Refiner, Lippe et al., arXiv:2308.05732. VERIFIED primary. Neural solvers lose accuracy because they ignore low-amplitude high-frequency components; multi-step diffusion-style refinement models all frequencies, giving stabler long rollouts and uncertainty estimates. Relevance (inference): an MSE loss is dominated by high-amplitude low-frequency content; dense-region short-wavelength force structure is low-amplitude in loss terms, which is a candidate cause of a bias confined to dense regions. Suggests a spectrally/amplitude-normalised force loss as a test.
- Solver-in-the-Loop, Um et al., arXiv:2007.00016. VERIFIED primary. Putting the numerical solver in the training loop (unrolled) lets the network learn corrections stable over hundreds of steps. Relevance: supports unrolled training with the Verlet integrator inside the loss, which we already do; nothing new.
- Gradients are Not All You Need, Metz et al., arXiv:2111.05803. VERIFIED primary. Chaos-driven failure of differentiating through long unrolls, tied to the Jacobian spectrum. Relevance (inference): dense N-body regions are locally chaotic; long-k unrolled gradients there may be noisy/exploding, which could bias which regions the optimiser effectively learns from. Check per-region gradient norms vs k before lengthening horizon.

## 2. Structure-preserving / energy models

- Hamiltonian Neural Networks, Greydanus et al., arXiv:1906.01563. VERIFIED primary. Learn a Hamiltonian so dynamics conserve energy by construction; better generalisation on two-body. Relevance: conservation by architecture, not by loss. Our model outputs per-particle acceleration from a field, so non-conservative force is possible; conflicts with the project rule against hard-coded conservative forces (soft priors OK) and with non-conserving systems. Flag as violation if used as hard constraint.
- SympNets, arXiv:2001.03750. VERIFIED primary. Symplectic network families with universal approximation for symplectic maps; pendulum and three-body tests. Relevance: structure-preserving map, not a force field; same constraint conflict as HNN.
- Lagrangian Neural Networks, Cranmer et al., arXiv:2003.04630. VERIFIED primary. Learn Lagrangians without canonical coordinates; energy-conserving. Same relevance note as HNN.
- Hamiltonian Graph Networks with ODE integrators, Sanchez-Gonzalez et al., arXiv:1909.12790. VERIFIED primary. Differentiable ODE integrator + Hamiltonian internal representation improves prediction, energy accuracy and generalisation to unseen step sizes. Relevance: closest structure-preserving analogue to a GNN particle simulator with integrator in the loop.
- sGDML, Chmiela et al., arXiv:1611.04678. VERIFIED primary. Learn the energy and obtain forces as its gradient, giving energy-conserving force fields. Relevance: a gradient-of-potential head on the grid (force = -grad phi) removes curl and hence path-dependent work by construction; this is a soft/architectural option to weigh against the "no hard-coded conservative forces" rule. Related to the recorded LJ finding (mesh injects non-conservative force).

## 3. Derivative / Sobolev / force-matching losses

- Sobolev Training, arXiv:1706.04859. VERIFIED primary. Matching target outputs and derivatives improves accuracy and data efficiency. Relevance (inference): applying it to the field means penalising error in grad(acceleration) (tidal tensor); a work bias along a path integrates force error, and field-gradient error controls how force error varies in dense regions. Differs from force-anchor aux (which matches values only).
- sGDML (above) also covers force matching in the gradient domain.

## 4. Distribution / statistics matching

- Training neural operators to preserve invariant measures, arXiv:2306.01187. VERIFIED primary. Squared-error trained operators forecast short-term well but miss long-term statistics; adds optimal-transport loss and a contrastive alternative to preserve the invariant measure. Relevance: directly matches our symptom class (short-horizon MSE fine, long-run statistic wrong). Candidate: OT/sliced-Wasserstein on pair-distance or density-field histograms, or on the per-particle work distribution, as an added term.
- DiffTRe, Thaler and Zavadlav, arXiv:2106.01138. VERIFIED primary. Trains potentials on time-independent observables (e.g. structure) via trajectory reweighting without backprop through MD (~100x cheaper gradients). Relevance (inference): a radial-distribution-function or energy-distribution target could be trained without long unrolls; unverified whether it handles N-body gravity.

## 5. Tail / robust risk

- Tilted ERM, Li et al., arXiv:2007.01162. VERIFIED primary. One tilt hyperparameter smoothly interpolates between emphasising and suppressing outliers; links to superquantile (CVaR-like) risk. Relevance: our bias is systematic not outlier-driven (measured), so tail weighting alone is unlikely to fix it; useful only if the biased region is a loss-weight minority (dense cells). Huber (outlier suppression) already failed, consistent with this.

## 6. Uncertainty

- Kendall and Gal, arXiv:1703.04977. VERIFIED primary. Input-dependent aleatoric uncertainty loss attenuates noisy samples. Relevance (inference): heteroscedastic weighting would down-weight dense regions with large intrinsic error, likely worsening a dense-region bias. Not recommended for this failure.

## 7. Learned cosmological / N-body surrogates

- Li et al., AI-assisted super-resolution cosmological simulations, arXiv:2010.06608. VERIFIED primary. Matches power spectra and halo mass function; scales to 1000x larger box. Relevance: statistic matching (spectra) is the evaluation standard; adversarial loss details not read on the abstract page, so no claim.
- He et al., D3M, arXiv:1811.06533. VERIFIED primary. Predicts nonlinear structure beyond 2LPT; generalises across cosmology. No loss details read.
- Simple lessons from complex learning, arXiv:2206.04573. VERIFIED primary. Network trained on Gaussian N-body data fails on transverse-mode initial conditions; beats COLA on nonlinear power spectrum. Relevance: evidence that learned field surrogates can miss a force-field component class (transverse = non-gradient), analogous to our non-conservative curl leakage.
- Ramanah et al., Painting halos, arXiv:1903.10524. VERIFIED primary. Local-patch network reproduces power spectrum and bispectrum, applied to larger boxes. Relevance: patch/tiled application as in our tiling; abstract reports no tiling-induced bias.
- CosmoGAN, arXiv:1706.02390. VERIFIED primary. GAN emulator matches summary statistics of lensing maps. Relevance: adversarial critics match marginal statistics, not conservation; no evidence for work/energy bias control.
- Gupta and Brandstetter, arXiv:2209.15616. VERIFIED primary. U-Net/FNO benchmark for multi-scale PDE surrogates. Relevance: U-Net baseline context only.

## 8. Synthesis (inference, untested)

1. Evidence closest to our symptom: invariant-measure (2306.01187) and low-amplitude-frequency neglect (PDE-Refiner 2308.05732). Both say plain squared error is blind to a systematic effect that is small in per-step amplitude.
2. Candidate tests, each separate vs frozen baseline: (a) loss on path-integrated work/energy increment per particle, summed over the rollout (signed, not squared per step, so bias cannot cancel into noise); (b) curl penalty or force = -grad phi head as a soft prior; (c) Sobolev term on field gradients; (d) OT or sliced-Wasserstein on per-particle dE distribution or pair-distance histogram. Note (a) resembles the failed energy-similarity term only if it was per-step/unsigned; check the earlier definition before running.
3. Likely not helpful: heteroscedastic weighting, tail weighting (bias is systematic), more noise injection, hard HNN/SympNet structure (rule conflict).

## Gaps (no verified source obtained; no claims made)

- Pushforward-trick specifics (Brandstetter), Stachenfeld abstract-level detail beyond 2112.15275, CVaR / top-k / L-p losses in simulators, hard-example mining, horizon curriculum for learned simulators, adversarial critics for dynamics (beyond CosmoGAN), PINN N-body, FMM-inspired learned gravity, GNN N-body papers, map2map original (Li et al. 2010.06608 is the super-resolution paper; map2map displacement paper not checked).
- Rejected: arXiv ids 2106.01638 and 2210.07180 resolved to unrelated papers and were discarded.
