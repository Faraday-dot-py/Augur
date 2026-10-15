# Error correction / stability for long autoregressive rollouts: literature survey (2026-10-12, uncommitted, research only)

Marks: [V] = number/claim read in the source text this session; [S] = from a search snippet or abstract only; [U] = unverified / my inference. Several fetches returned only abstracts or unreadable PDFs; those papers are marked and their numbers are not quoted.

## 0. What Augur already has (so it is not re-proposed)

Sources: memory session 2026-10-12, docs/debugging/scatter-correction-research.md (worktree agent-a98bdf87385c8eb37), scripts/train_scatter_field.py.

- Training: teacher-forced supervised unroll with full BPTT, curriculum k 4 -> 20 (`--k-start/--k-end`); optional `--long-windows` = soft dE/dL penalty on free-rollout windows with state detached between windows and no truth targets (ft_longpen: nr100 dE@1e3 .017 -> .0044, bh300 .48 -> .097, accuracy cost at lam 3). No input noise, no pushforward, no statistical/distribution loss in the scatter trainer (grep of train_scatter_field.py and scatter-*.md: no hits).
- Tried and negative/rejected: closed-loop relabel curriculum (control .0298 vs relabel .0461 err@20; fails 8 harness rows), thermostat/L-rescale (hard-coded), parareal, Kalman, far quadrupole. C40 (k_end 40) diverged in training. Stepwise unroll curricula on the token model gave no gain (v19), unroll 26-40 worsened.
- Dominant open failure (mechanism 8): runaway of the grid-net force/a0 recurrent channel past the training horizon (17 -> 1150 over ticks 21-24, x2-6 per tick). Positive feedback through recurrent field + a_prev. Inside the horizon the model is good. Mechanism 5 (grid-edge spike) is a boundary bug, not a learning problem.
- Constraint: any short fine-tune of shared weights corrupts 1000-tick dE (nr100 dE@1e3 .017 -> .29); same-schedule control always required. Chaos ceiling: position error beyond ~tick 50-100 is uninformative; judge on statistics.
- Rules: no hard-coded conservative forces (soft OK); one fix per run vs frozen baseline; architecture before hparams.

## 1. Methods

### 1.1 Input / recurrent-state noise injection (GNS, learned turbulence, GameNGen)
- Mechanism: corrupt the model's inputs during training so it learns to pull perturbed states back; the clean target means the net learns a contraction toward the data manifold.
- Evidence: GNS [V abstract; details V via ar5iv text]: random-walk noise on input velocities sigma_v = 3e-4 with positions adjusted accordingly; targets recomputed so the model corrects the velocity noise; random-walk beat other noise types; rollout error best at an intermediate scale while one-step error rises with scale (no numeric table in text) (Sanchez-Gonzalez 2020, arXiv 2002.09405). Learned coarse turbulence models [S]: Gaussian input noise sigma=0.01 gave worse one-step but more stable long rollouts; no-noise models unstable at small dt (Stachenfeld et al., arXiv 2112.15275). GameNGen [S]: noise-augmenting context frames with a learned noise-level embedding removed quality degradation seen after 20-30 frames; authors call it mitigation, not fix (Valevski, arXiv 2408.14837). Brandstetter [V]: Gaussian sigma=0.01 improved stability somewhat but lowered accuracy, pushforward better.
- Cost: negligible (one randn per step); no inference change.
- Fit: generalization yes (sim-agnostic), tileable yes, O(N) yes, interpretability neutral. Soft prior, no forces coded.
- Novelty vs tried: NOT tried in scatter. Contact-C notes list "noise recurrent state" as an open idea only. Directly aimed at mechanism 8 if noise is applied to the recurrent channels (a_prev, field/a0 state) and not only to positions. Caveats: (a) noise on a_prev must keep Verlet consistency; (b) any retrain/fine-tune must be checked on dE rows (mechanism 13); (c) GNS noise is per-particle, whereas Augur's runaway lives on the grid field, so noise scale for the field channel needs a probe of the observed field error distribution first [U].

### 1.2 Pushforward trick (Brandstetter et al.)
- Mechanism: unroll 2 steps, no gradient through the first; train second step from the model's own (perturbed) output against truth for that step; the first-step error acts as an adversarial perturbation.
- Evidence: [V] MP-PDE paper (arXiv 2202.03376): "important not to backpropagate through the first unroll step"; Table 1 pushforward+bundling vs FNO-RNN E1 (250,100) accumulated error 11.93 -> 0.54, E2 17.09 -> 2.53 (confounded with temporal bundling). Counter-evidence [V]: PDE-Refiner reports pushforward "does not improve performance in our setting" on KS (arXiv 2308.05732). DySLIM [V] finds Pfwd-trained models drift off the attractor.
- Cost: ~1 extra no-grad forward per sample; cheaper than BPTT.
- Fit: sim-agnostic, O(N). But it needs truth targets for a state reached from the model's own rollout. For chaotic N-body, truth at tick s from a perturbed state diverges: this is exactly why Augur's relabel used the exact sim from the model state. Relabel was negative, and the same-schedule control fine-tune was better, so pushforward-with-truth-target sits inside the already-negative class.
- Novelty: low. Skip unless the relabel failure is shown to be due to fine-tune drift rather than the idea (the relabel doc's open item: rerun at lr ~5e-5 with smoothed gate).

### 1.3 Unrolled multi-step training / solver-in-the-loop (Um et al.; GraphCast; curriculum)
- Mechanism: backprop through k own-rollout steps; model sees its own error distribution.
- Evidence: [V] Solver-in-the-Loop (arXiv 2007.00016): wake flow velocity error 0.146 uncorrected -> 0.049 supervised -> 0.013 SOL32; buoyancy flow ~40% gain at 1-4 unrolled steps, >54% at 64, ~60% at 128; stable several hundred steps; no gain where forcing is random/unpredictable; cost per iteration 0.21 s (SOL2) -> 1.25 s (SOL16) (memory grows with k); early training can diverge, so pretrain with ~2 steps. GraphCast trains to 12 steps (6 h -> 3 days) with a curriculum and full BPTT [S].
- Fit: sim-agnostic, O(N). Augur is already here (k_end 20); C40 diverged; token-model 26-40 step unrolls worsened. The literature gain is for deterministic, well-predicted flows; chaotic N-body targets beyond ~20 ticks are not predictable (chaos floors), consistent with Augur's result.
- Novelty: none. Do not re-propose. Useful only as the recipe for stabilizing longer unrolls: gradient truncation / stochastic-length unroll (Self Forcing uses stochastic gradient truncation [S]) so memory stops being the limit [U for N-body].

### 1.4 Statistical / invariant-measure losses (DySLIM, Jiang et al., adversarial OT)
- Mechanism: add a loss comparing distributions of model-rollout states with truth statistics (MMD on batches of states, optimal transport, or contrastive), so long-horizon behaviour is constrained without matching trajectories.
- Evidence: DySLIM [V] (Schiff et al., ICML 2024, arXiv 2402.04467): loss = base + lam1*MMD(mu*, S_theta# mu*) + lam2*MMD(S# mu*, S_theta# mu*); Kolmogorov flow, Table 1 (x1e-2), base -> DySLIM: 1-step Wass1 16.2 -> 5.55, TCM 5.39 -> 2.45; Curr Wass1 9.66 -> 4.76; Pfwd Wass1 4.64 -> 4.51, TCM 3.68 -> 0.72 (MELRw worse for 1-step 0.44 -> 0.85, i.e. not uniformly better); KS: regularized runs stay on attractor while Curr diverges and Pfwd drifts (qualitative). Cost: O(n_b^2 d) extra compute and memory in batch size, wall-clock +0-3% in Table 3 (40 -> 40 h, 74 -> 76 h, 145 -> 146 h). Jiang et al. [S] (arXiv 2306.01187): OT loss needs expert-chosen features, contrastive version near OT performance, no physics knowledge needed.
- Fit: soft penalty, allowed. Generalization: statistic must be chosen per system, but contrastive version is generic; interpretability neutral. O(N)-compatible if statistics are computed on local patches/subsamples. Strong conceptual fit to Augur's chaos-ceiling problem: beyond tick ~50-100 only distributional properties (energy per body, speed histogram, g(r), virial ratio, bound fraction) are meaningful.
- Caveat [U]: DySLIM assumes an ergodic invariant measure; gravity/BH scenes are non-stationary (clusters collapse, evolve), so the reference must be a conditional distribution at matched time (the lam2 "conditional" term, S# mu*) rather than a stationary one. Augur's ft_longpen already is a one-statistic (dE, dL) special case; this generalizes it.
- Novelty: medium. dE/dL soft penalties tried; distributional/MMD terms not.

### 1.5 Jacobian / spectral-radius / Lyapunov regularization
- Mechanism: penalize growth of the one-step state-to-state map (leading singular value / eigenvalue, via power iteration with JVP/VJP) so errors cannot amplify per step.
- Evidence: weak and mostly recent preprints. [S] Ainslie et al. (arXiv 2608.16084): Jacobian-based loss gave smaller leading eigenvalue and lower next-step error vs RMSE across integrators; full text not readable here, numbers not verified. [S] arXiv 2605.24868: pure damping yields a dissipative approximation of chaos; chaotic systems need perturbation growth near the true Lyapunov exponent. [S] arXiv 2506.12810: |lambda| penalty, full Jacobian + QR is O(d^3) (secondary review only).
- Cost: a few JVP/VJP per training step on the penalized channel; none at inference.
- Fit: soft prior, sim-agnostic, no inference cost. Interpretability neutral. Risk: over-damping (Augur already has KE collapse / restitution .5 in B2).
- Novelty: high and targeted. Augur's mechanism 8 is exactly an instability of the recurrent-channel map (field/a0/a_prev -> next field), growth x2-6 per tick. Penalize only the spectral norm of that sub-Jacobian (not of the physical dynamics), cap at ~1 rather than minimize [U]. Untested anywhere on this model.

### 1.6 Learned inference-time denoiser / thermalizer
- Mechanism: separate diffusion model trained on the data distribution; a noise-level classifier on the rollout state detects drift; when above threshold, re-noise and denoise the state back toward the data manifold.
- Evidence: [V] Thermalizer (Pedersen, Zanna, Bruna, ICML 2025, arXiv 2503.18731): 2D Kolmogorov (Re 1e4) and 2-layer QG; unthermalized U-Net emulators mostly unstable by ~2000 steps; thermalized stable for all 10,000 steps and stay stable at 1e5 (main-text ratio ~5x; abstract claims 10x in v1 and 100x in the ICML version, the sources disagree); QG kinetic energy lower than reference; temporal autocorrelation unchanged. Cost: classifier head every step ~60% of one U-Net step (QG); ~140 thermalization passes per 10k steps; thermalizer 53M params vs 48M emulator; s_init/s_stop need per-pair brute-force tuning; no head-to-head against noise/pushforward in the paper.
- Fit: local convolutional, tileable, O(N). Problems: (a) needs an invariant measure to define "on-manifold"; gravity states have none [U]; (b) heavy: extra 50M-param model plus +60% step cost on a 1.5 s/tick 1e9 run; (c) learned projector toward the data distribution may pull toward training-like clusters and damage OOD generalization, core requirement 1.
- Novelty: medium, cost high. A scoped variant [U]: denoise only the recurrent field channel with a small auto-encoder trained on fields produced by the exact force, rather than whole-state diffusion. Not a literature result.

### 1.7 Failure-driven rollout training (StABlE; related to DAgger)
- Mechanism: run many parallel rollouts of the current model; detect unstable replicas; rewind them to a recent stable state; take gradient steps on an observable (statistical) loss from those states plus the original supervised loss; repeat.
- Evidence: [V] StABlE (Raja et al., arXiv 2402.13984): SchNet aspirin 500 K median stable time 42 ps -> 602 ps (~14x, my arithmetic) using 1,000 reference structures, beating baselines trained on 10,000 and 50,000; water GemNet-T 5/5 runs stable to 1 ns vs 1/5 baseline (median 312 ps); diffusivity 2.4e-9 vs reference 2.3e-9 m^2/s (baseline 5.0e-9); cutting timestep 10x did not fix baseline instability; cost: no extra reference labels, replicas vectorized; small rise in held-out energy/force error.
- Fit: soft observable loss, sim-agnostic, no inference cost. Differs from Augur's failed relabel in one key respect: it does not need a truth label at the failed state; it uses a loss defined on observables (distance/RDF; here dE, dL, virial, speed distribution). Overlaps with Augur's ft_longpen (long-window soft penalty) except for the rewind-to-failure-state restart and the replica pool.
- Caveat: Augur's fine-tune drift (mechanism 13) is the same trade-off StABlE reports as a small accuracy loss; here the dE@1e3 rows degrade 10x, so the drift would need to be measured on the harness.
- Novelty: medium-low (variant of ft_longpen). Adds "train from where the model actually breaks" which Augur does not do (long-windows start from truth/training IC windows, I did not find otherwise in the trainer [U]).

### 1.8 Multi-step refinement / diffusion decoder (PDE-Refiner, GenCast)
- PDE-Refiner [V] (Lippe et al., arXiv 2308.05732): K denoising steps per forward step, exponential noise schedule sigma_k = sigma_min^(k/K); KS correct-rollout time: MSE U-Net ~75 s (94 steps), PDE-Refiner almost 100 s, 1 refinement step ~20% better than best baseline, gains flatten at 3-4 steps; 2D Kolmogorov cost 16.53 s vs 4.04 s (~4x); sigma_min^2 = 2e-7 (KS), 1e-3 (Kolmogorov). Also: pushforward did not help, noise-free Sobolev loss shorter, 5-ensemble stays <80 s. Mechanism claim: error comes from neglected low-amplitude (high-frequency) components.
- GenCast [S]: diffusion ensemble beats ENS on >96% of 1320 targets (CRPS, ensemble-mean RMSE), members sharp; deterministic models blur.
- Fit: sim-agnostic, O(N), but 3-4x inference cost for +33% horizon on a smooth PDE. Augur's problem is not missing high-frequency content but runaway; chaos ceiling means a sharper single trajectory is not more correct beyond ~tick 50. Reject for now; revisit only if the goal becomes distribution-level realism.

### 1.9 Conservation-enforcing projection
- Hard projection [V] (arXiv 2511.17258, training-free): LBFGS/linearized projection of output onto discretized PDE residual constraint; NS FNO residual 8.13e-2 -> 0.00901e-2 and MSE 13e-1 -> 2.63e-1; KS FNO MSE 4.01e-2 -> 3.91e-2; no long-rollout stability metric, not differentiable, needs the PDE. [S] Conservation-constrained FNO (ICML 2026 listing): hard projection beats penalty sweep, divergence down ~5 orders; [S] FluxNet learns conservative flux form. ACE2 [S] conserves dry air and moisture exactly by construction; ACE v1 [V] had no conservation term and trained on single-step loss only (no multi-step loss or noise mentioned), stable 100 years [S] via SFNO + normalization choices.
- Fit: the requirement bars hard-coded conservative forces and the thermostat/L-rescale class is already rejected; projections needing the governing equation violate generalization/interpretability (the point of Augur is to recover rules). Momentum is already exact via momfix. Soft versions = ft_longpen (already there).
- Novelty: none. Take only the observation that conserving structure (fluxes/pairwise antisymmetry) was the route to stability in ACE2/FluxNet, which is an architecture argument (already satisfied for momentum).

### 1.10 Learned corrector / nudging / data assimilation
- Mechanism: train a correction network to the increment that nudges a biased model toward reference (Watt-Meyer-style nudged training sets; solver-in-the-loop correction).
- Evidence: [V] SOL (see 1.3): up to 4-5x lower error than supervised in unsteady wake flow, but needs a differentiable solver and relies on solver-in-the-loop unrolling. [S] nudged-dataset correction (arXiv 2304.02117; Watt-Meyer 2021 via slides): nudging targets pulled toward reference make a learnable bias correction when free-run trajectories diverge by chaos. Brajard-style ML+DA: not found in this search [U].
- Fit: needs reference data at deployment (nudging) or an inner solver; tile/O(N) fine as local conv. In Augur the nudge-analog is exactly the exact-sim relabel (negative). Prior writeup already ranked frozen-base residual corrector below diagnostics and found it inherits the same cliff beyond its horizon.
- Novelty: none.

### 1.11 Ensembles / ranking / best-of-k
- Mechanism: run several perturbed or stochastic members; average (variance reduction) or select by a consistency score (e.g. energy drift).
- Evidence: PDE-Refiner [V]: ensemble of 5 MSE models stays <80 s vs 100 s (ensembling does not fix drift). GenCast [S]: CRPS-trained/diffusion ensembles give sharp calibrated members; averaging deterministic members blurs. NeuralGCM [S]: ensembles needed beyond chaotic divergence.
- Cost: k x inference (violates the tick budget at 1e9: 1.5 s/tick). Best-of-k by energy is a hard-coded selector but does not add forces; still k x cost.
- Novelty: low value. A cheap variant: ensembles only as an evaluation tool (spread = trust signal), not for correction.

### 1.12 Scale/diversity and architecture notes
- Pretraining/diversity [V] (arXiv 2506.14850): MD17 aspirin onset-of-instability 42.6 ps (SD 25.9) from scratch vs 274.2 ps (SD 44.6) after OC20 pretraining (~6.4x; abstract says "up to 3x", unexplained), force MAE about equal (~5 meV/A); model width also differs (confound), one molecule, 3 seeds. Same lesson as "Forces are not enough" [S]: force error does not predict rollout stability; Augur saw the same (pp8 halved force error, rollout 73% worse).
- Fit: mixes with Augur's finding that ft2 orbit-mix and cluster IC diversity decide OOD behaviour. Not a correction method.
- Scheduled sampling / DAgger (Bengio 2015, Ross 2011): classical sources, not re-fetched [U]. Search for DAgger-with-relabelled-solver in neural PDEs found nothing directly [S]; Augur's relabel experiment is thus itself a rare data point.

## 2. Ranking for Augur

Criteria: targets mechanism 8 (the dominant long-horizon failure), cost, fit to the four requirements and the no-hard-coded-forces rule, novelty versus tried.

| # | Method | Best evidence | Cost | Fit | Novelty vs tried |
|---|---|---|---|---|---|
| 1 | Noise on recurrent channels (a_prev, field/a0) during training | GNS sigma_v 3e-4 random walk; turbulence sigma 0.01 more stable; GameNGen noise aug. removes drift after 20-30 frames | ~0 | Full | Not tried; named open in contact-C doc |
| 2 | Spectral-norm penalty on the recurrent-channel Jacobian only | Weak: preprints only [S]; warning about over-damping | few JVP/VJP in training | Soft prior, no inference cost | Not tried; directly matches x2-6/tick runaway |
| 3 | Distributional soft loss (MMD/contrastive on dE per body, speeds, g(r), virial) | DySLIM Kolmogorov Wass1 16.2 -> 5.55 (1-step), +0-3% wall time; Jiang contrastive ~ OT | O(n_b^2) in batch | Soft, generic if contrastive; non-stationary issue [U] | Generalizes ft_longpen (dE/dL); new as a distribution term |
| 4 | Failure-state restarts with observable loss (StABlE-style) | Aspirin 42 -> 602 ps stable (~14x), water 5/5 vs 1/5 stable to 1 ns | modest (replica pool) | Soft, no inference cost; drift risk (mech. 13) | Partly ft_longpen; restart-from-failure is new |
| 5 | Local field AE denoiser on recurrent field only (Thermalizer-scoped) | Thermalizer: 2,000 -> 10,000 stable steps (QG/Kolmogorov), +60% step cost | training a second net + per-step cost | Local conv, tileable; no invariant measure for gravity [U] | New; high cost |
| 6 | Pushforward with truth target | E1 error 11.93 -> 0.54 (with bundling); no gain in PDE-Refiner KS | low | OK | Same class as negative relabel/control fine-tune |
| 7 | PDE-Refiner / diffusion step | KS 75 -> ~100 s, 4x cost | 3-4x/tick | O(N) but expensive | Not tried; poor cost/benefit |
| 8 | Hard conservation projection | NS residual 4 orders down; no rollout stability study | solver per step | Violates rules | Rejected class |
| 9 | Ensembles / best-of-k | 5-ensemble < 80 s vs 100 s | k x | Poor | Evaluation use only |

Not re-proposed (tried/rejected): longer unrolls, closed-loop relabel, thermostat/L-rescale, parareal, Kalman, quadrupole.

## 3. Test protocol implied by Augur's rules

- One method per run vs frozen baseline (E_ms_kp_pot_v_g128) plus a same-schedule control fine-tune; 4+ held-out seeds; headline from fresh seeds.
- Count nr100 dE@1e3 and the other harness dE rows as failing for any fine-tune that regresses them (mechanism 13); prefer training from scratch at the contact-C horizon (k_end 20) and testing at 100+ ticks.
- First diagnostic for #1/#2 (inference only, no training): measure the growth factor of a0/a_prev perturbations per tick in the existing checkpoint by injecting known-size noise at tick 15 and tracking amplification (this gives the noise scale for #1 and the target for #2) [U].
- Judge with channel breakdown (max|a0|, dE, tunnel, restitution), not aggregate MSE; beyond ~tick 50 use statistics only.

## Sources
- Sanchez-Gonzalez et al. 2020, GNS: https://arxiv.org/abs/2002.09405 (details read via ar5iv)
- Brandstetter et al. 2022, MP-PDE / pushforward: https://arxiv.org/abs/2202.03376
- Lippe et al. 2023, PDE-Refiner: https://arxiv.org/abs/2308.05732
- Stachenfeld et al. 2022, learned coarse turbulence: https://arxiv.org/abs/2112.15275 (snippet only)
- Um et al. 2020, Solver-in-the-Loop: https://arxiv.org/abs/2007.00016
- Schiff et al. 2024, DySLIM: https://arxiv.org/abs/2402.04467
- Jiang et al. 2023, invariant measures: https://arxiv.org/abs/2306.01187 (abstract only)
- Pedersen et al. 2025, Thermalizer: https://arxiv.org/abs/2503.18731
- Raja et al., StABlE: https://arxiv.org/abs/2402.13984
- Fu et al., Forces are not enough: https://arxiv.org/abs/2210.07237 (abstract only); Beyond force metrics: https://arxiv.org/abs/2506.14850
- Valevski et al., GameNGen: https://arxiv.org/abs/2408.14837 (snippet only)
- Huang et al., Self Forcing: https://arxiv.org/abs/2506.08009 (snippet only)
- Training-free projections: https://arxiv.org/abs/2511.17258
- Eigenanalysis of emulators: https://arxiv.org/abs/2608.16084 (PDF unreadable; abstract-level only); comparative rollout-stability study: https://arxiv.org/abs/2605.24868; Lyapunov penalty: https://arxiv.org/abs/2506.12810 (secondary summary only)
- ACE: https://arxiv.org/abs/2310.02074; GraphCast: https://arxiv.org/abs/2212.12794; GenCast: https://arxiv.org/abs/2312.15796; NeuralGCM: https://arxiv.org/abs/2311.07222 (stability ablations not found)
- Nudged-dataset correction: https://arxiv.org/abs/2304.02117; APEBench: https://arxiv.org/abs/2411.00180 (text unreadable; not used for numbers)
