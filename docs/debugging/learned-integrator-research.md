# Learned leapfrog / symplectic-style integrator for the scatter field (research, 2026-10-20)

Verdict: do not build. No GPU used. One small CPU diagnostic (jobs 3763, 3764; script `scripts/scatter_integrator_noise.py`; raw `results/integrator_noise.json`, `results/integrator_noise_coherent.json`).

Definitions: E = Exp E checkpoint (`E_ms_kp_pot_v_g128`, the frozen baseline); N = bodies per scene; dt = 0.1 (one tick); eps = 0.5 (softening); err@k = pooled mean per-body position error after k ticks vs the fp64 4-substep truth; dE@1000 = |E(t)-E(0)|/(K0+|U0|) after 1000 ticks (K kinetic, U potential energy); expB = 10-100-body scale-init scenes (24 scenes, seed 9300); cluster = 84-269-body cluster scenes (12 scenes, seed 9200); nr100 = the 100-body conservation scene set (seeds 9100/9200/9300/4738, 1 scene each); M = force evaluations (substeps) per tick; sigma = relative force noise; Verlet-1 = the model's integrator (M=1); KDK-M = kick-drift-kick with M substeps per tick. Seeds 9000/12000 were not used (CPU oracle, seeds as in the earlier root-cause tables so numbers compare).

## 1. Premise

Velocity Verlet (kick-drift-kick, KDK) and leapfrog (drift-kick-drift, DKD) are the same second-order symplectic map up to a half-step relabelling of the state, for a fixed dt and a force that depends on position only. "Learned leapfrog" can therefore only mean one of: (1) learned splitting coefficients; (2) a learned correction to a drift or kick (modified/shadow-Hamiltonian view); (3) a learned Hamiltonian that a symplectic integrator then preserves; (4) learned or adaptive step size and substeps; (5) symplectic-by-construction networks that replace the step.

What the code actually integrates (verified, `scripts/scatter_field.py` `_force` 445-478, `_step` 496-507): the acceleration is -grad(phi) of one net channel on the grid (central differences, `neg_grad` 312), gathered by CIC, plus an optional kernel term, far field, learned pp=2 pair term, then `momfix` removes the net force. The net input contains the deposited momentum channels, a recurrent field, and the predicted velocity `vel + dt*a_prev` (505). So the kick is velocity dependent, the force is not the gradient of one scalar of the particle configuration, and the step is an explicit predictor-corrector. It is not a Hamiltonian system and Verlet-1 is not symplectic for it. Already measured on E (`scatter-conservation.md`): curl/total .13, tangential force 3-40%, velocity dependence 3-100%. "The model already outputs a potential" therefore does not give a conserved energy: phi is a field functional of all positions and velocities, and F_i is not dH/dx_i of anything.

## 2. How much is integrator error? (the bounding diagnostic, exact forces, no model)

Existing data (all exact force, fp64, truth = KDK-4):
- `substep-theory.md` / `long-rollout-root-cause.md`: err@20 Verlet-1 .0023 (expB) / .035 (cluster) vs E .0254 / .067. Quadrature model err^2 = F^2 + I^2 gives the integrator share of E's squared err@20 as 0.8% (expB) and 27% (cluster); removing it entirely would cut err@20 by 0.4% (expB) and 15% (cluster). Exact Verlet-1 equals the cost of about 0.7% (expB) / 2% (cluster) iid force noise.
- `scatter-conservation.md` (job 3323): nr100 dE@1000 signed median: E .020, exact Verlet-1 (t1) .0055, truth KDK-4 3e-5 (E is 680x truth; t1 is 180x truth, 27% of E).
- Job 3629 (inference-only substeps of trained models): no gain >= 20% on cluster err@20, abundant-neighbour scenes worse (+9..+91%), E dE@1000 .018 -> .93 (u2).

New (this note), exact force + injected relative force noise (sigma) at M = 1/2/4, noise error smooth in time within a tick ("coherent": the per-tick noise is linearly interpolated across substeps so a better integrator does not average the noise away; sigma .08 reproduces E's expB err@20 .0247 vs .0254):

| err@20 | M=1 | M=2 | M=4 |
|---|---|---|---|
| expB sigma .02 | .00683 | .00611 (-10%) | .00605 (-11%) |
| expB sigma .08 | .02473 | .02429 (-1.8%) | .02421 (-2.1%) |
| cluster sigma .02 | .04496 | .02357 (-48%) | .02249 (-50%) |
| cluster sigma .08 | .10155 | .09041 (-11%) | .08966 (-12%) |

err@100, sigma .08: expB 1.503 / 1.484 / 1.482; cluster 4.865 / 4.616 / 4.552.

Confound found and discarded as evidence: with iid noise per force evaluation (job 3763) M=2 and M=4 cut sigma .08 err@20 by 30% / 48% (expB) and 36% / 54% (cluster), but that is noise averaging (variance per tick falls as 1/M), not integration accuracy. It does matter for interpretation: E's force error has a 34% (expB) sub-cell phase part that decorrelates between substeps, so some of the old ft_sub2 gain (-20..-25% err@5-20, 2x cost, confounded by fine-tune drift, `scatter-fixes-2026-10-06.md`) is probably averaging, like the 4-shift average (-39% err@20, 4x cost).

nr100 dE@1000 (|dE| median over 4 scenes, large scene spread: exact Verlet-1 per scene +.033 / +.010 / -.002 / -.017):

| | M=1 | M=2 | M=4 |
|---|---|---|---|
| exact | .0135 | .00043 | .00009 (= truth) |
| coherent sigma .02 | .0214 | .0019 | .0016 |
| coherent sigma .08 | .0450 | .0171 | .0221 |

Reading: err@20 integrator share is 2% (expB) and 11% (cluster) of E-sized force error, below the 20% gate in both and below 10% in expB. dE@1000 is the one metric where the integrator is large (27% of E by signed medians; a factor 2 under sigma .08 noise), but n = 4 scenes, the noise proxy gives Verlet-1 + noise .045 vs E's actual .020 (E is not iid noise), and the only way found to capture it is M >= 2 force evaluations.

Predictions written before the runs: expB err@20 M=2 within 3% of M=1 at sigma .08 (got -1.8%, confirmed); cluster .095-.097, i.e. <= 7% gain (got .0904, -11%: integrator share slightly above the quadrature estimate, still < 15% bound); nr100 exact dE M=1 ~.005 (got |.|.0135 abs median, .0055 signed median, same scenes as job 3323); the iid-noise variant was not predicted and is the confound above.

## 3. Candidate evaluation

Budget statement: one force evaluation per tick. "Extra evals" = additional full grid solves (global FFT + U-Net) per tick; density scaling is of the added cost.

(1) Learned/trainable splitting coefficients. A one-evaluation symmetric splitting has a unique second-order member (Verlet); any other drift/kick weight is first order and, with one evaluation, is a rescaling of the force the net already learns, so it adds nothing. Real gains (optimized higher-order, force-gradient) need >= 3 evaluations or a Hessian-vector product through the U-Net. Measured with exact force: KDK3 3 evals err@20 .00012 / .0013 (expB / cluster), Yoshida-4 .00062 / .045 (worse than Verlet-1 .035 in clusters), exact-jerk drift worse than Verlet-1 (`substep-theory.md`). With 6% force error the extra order is wasted. Fixes: integration error only (and <= 11% of err@20). Extra evals 2-3+; O(N), cost x2-3 at all densities. Constraints: O(N) yes; no pp>=8 yes; no hard-coded conservation yes; generalization yes; tileable yes; interpretable yes. Evidence: measured (own data). No paper found that learns the coefficients of a leapfrog (search 2026-10-20: the closest hits treat the integrator as a learned map, see (5)).

(2) Learned correction to drift/kick (hypersolver / inverse modified Hamiltonian). Two facts. First, E is already this: it is trained with Verlet-1 unrolled, so its force absorbs the dt = 0.1 modified-force bias (the Symplectic RNN idea, arXiv 1909.13334, abstract only; the project's own pair probe shows E -20% at r = .1 vs B -2%, consistent but untested; `substep-theory.md` sec. 2 predicts the absorbed bias is 7.4% / 4% / 1.3% of |a| for nn < .25 / .25-.5 / .5-1). A second corrector would double count, which is why E + sub2 got err@20 +17% and dE x50. Second, the needed position bias (b_x) and velocity bias (b_v) differ by 3.5-7x with poorly aligned directions (cos .23-.81), so one force-like correction cannot fix both; a separate (x, v) corrector is the hypersolver S4 in `substep-theory.md` (one-step e1 predicted -50..-80%, err@20 <= -20% only in clusters, fails nr100 dE; an earlier learned residual (D4) cut one-step MSE 44% with null rollout and failed dE). Fixes: integration error, partly already absorbed. Extra evals 0, cost ~1 pair pass (O(N k)). Constraints: non-conservative correction is fine; non-symplectic so dE is not protected. Evidence: measured negative analogue (D4) plus theory; hypersolver papers read at abstract level only. Offen and Ober-Blobaum (2108.02492, abstract read) learn an inverse modified Hamiltonian so a symplectic integrator's error is compensated, but for a given Hamiltonian system with a Gaussian process; it presupposes the Hamiltonian structure this model lacks.

(3) Learn the Hamiltonian so a symplectic integrator preserves a learned energy. Requires F_i = -dU/dx_i for one scalar U(x) (translation invariant, momentum conserved exactly). The scatter field has no such scalar (sec. 1). Making it so (potential energy functional on the grid, force by autograd through scatter/U-Net) is O(N) with a backward pass (about 2-3x tick cost, extra evals 0 but heavy) and removes velocity input and momfix. Honest assessment: this is the "make the force conservative and central" fix the user rejected on 2026-10-06 (no hard-coded conservative forces; model must learn uneven interactions) and it breaks non-conserving systems (energy term = similarity to truth E(t)). Symplectic integration of it would be a structure choice stronger than the integrator; I count it as a violation unless an additional learned non-conservative head is kept (then the guarantee is lost for that head). It also does not touch the force-error floor: B and E share the architecture and differ 6x in force error. Evidence: constraint analysis from the repo plus HNN literature below. Recommendation: do not run without an explicit user decision; if run, as an opt-in arm vs frozen E.

(4) Learned timestep / adaptive / substep schemes. Measured negative or too costly: uniform substeps no gain on trained models (job 3629); per-body adaptive and router not worthwhile (error spread widely: nn < 1 holds 98% of squared one-step error but 43% / 75% of bodies; promoted sets < nn .5 worse than Verlet-1 at @50/@100); RESPA (grid once, pair term on substeps) is the only scheme that survives with exact force (rc 8, M 4: err@20 .0001 / .0014) but on E (pp = 2, learned pair) nf2 gave cluster -3%, dE unchanged; its premise needs pp = 8-size exact pairs, which the user dropped. Fixes: integration error only. Extra evals: M-1 grid solves (uniform) or M pair passes (RESPA, cost grows with local density times rc^2, and ~sqrt(rho) for individual timesteps). Evidence: measured. A learned step-size controller (RL; Dellnitz et al., 2104.03562, from `substep-literature.md`, not re-read) is validated on low-dimensional ODEs only.

(5) Neural symplectic maps (SympNet, SympFlow, HNN, SRNN, generating-function nets). All presuppose a (possibly non-separable) Hamiltonian system and a state of (q, p) for the whole system; N-body evidence is 3 bodies (SympNet 2001.03750: pendulum, double pendulum, three-body; Cai et al. 2111.15631: 3-body for 1e5 steps) and Horn et al. (2310.20398, via `substep-literature.md`) report HNNs hard to train across scales. They guarantee volume/symplecticity of the learned map, not accuracy of the force, so the 6% force-error floor is untouched; they also conflict with interpretability of a mean-field particle-mesh and with non-conserving systems. A Wisdom-Holman split (analytic Kepler + learned interaction, Cai et al.) is orthogonal to a softened mean-field force. Fixes: neither floor. Extra evals 0 for a learned map but replaces the grid pipeline (not O(N) tileable as written). Evidence: abstract-level only; no N >= 100 demonstration found. Not worth building.

## 4. Why the force floor is the real limiter (connection to documented problems)

- Typical-body force error 6.1% rel_rms (95% of squared error from nn < 1 bodies); exact Verlet-1 + 8% iid noise reproduces E's expB err@20/50/100 (.0256 / .168 / 1.54 vs E .0254 / .169 / 1.47); the coherent version here gives .0247 for err@20. A better integrator removes <= 2% (expB) or 11% (cluster) of that at E's force error; chaos amplifies the force seed, not the integrator seed, once force error > ~1-2%.
- Any gradient step on the U-Net breaking nr100 dE@1000 is a weight-space problem (E1, `error-correction-plan.md`); an integrator change does not alter that, and E + sub2 shows the converse (integrator change without retraining moves dE x50 because the force absorbed the old dt).
- The recurrent fed-back field runaway (contact model C) is a feedback gain of the net, independent of integrator order.
- Learned integrators would only matter for models whose force error is <= 1-2% (Exp B ckpt, untrained P3M pp = 8 on cluster); those need the analytic pair radius the user dropped.

## 5. Literature (searched 2026-10-20)

Tags: ABS = arXiv abstract page fetched this session (fetch tool returns a summary; no full text read); SNIP = search snippet only; 2ND = taken from `substep-literature.md` / `long-rollout-root-cause.md`, not re-checked; MEM = recalled, unchecked.

| ref | claim relevant here | tag |
|---|---|---|
| SympNets, arXiv 2001.03750 | symplectic-by-construction layers; universal approximation theorems for symplectic maps; pendulum, double pendulum, 3-body | ABS |
| Greydanus, Dzamba, Yosinski, Hamiltonian Neural Networks, 1906.01563 | learn H; energy conserving and time reversible by construction; two-body, pendulum | ABS |
| Chen, Zhang, Arjovsky, Bottou, Symplectic Recurrent Neural Networks, 1909.13334 | NN Hamiltonian + symplectic integrator inside an RNN, multi-step training, noisy complex systems (the "learned modified Hamiltonian offsets discretization error" reading is from 2ND) | ABS (modified-Hamiltonian claim 2ND) |
| Sanchez-Gonzalez, Bapst, Cranmer, Battaglia, Hamiltonian Graph Networks with ODE Integrators, 1909.12790 | zero-shot generalization to unseen step sizes and integrator orders | ABS |
| L2HMC (learned generalization of HMC), 1711.09268 | neural-parameterized generalization of HMC; I could not confirm from the abstract what is learned about leapfrog | ABS (leapfrog details unverified) |
| Offen, Ober-Blobaum, Symplectic integration of learned Hamiltonian systems, 2108.02492 | learn the inverse modified Hamiltonian so integrator error is compensated (GPs) | ABS |
| Othmane, Flaskamp, NN-enhanced integrators, 2504.05493 | learned local-error correction to classical RK with an embedded-RK safeguard; wind-turbine model | ABS |
| Canizares, Murari, Schonlieb, Sherry, Shumaylov, Hamiltonian Matching for Symplectic Neural Integrators (SympFlow), 2410.18262 | integrator built from flows of parametrized time-dependent Hamiltonians; qualitative energy behaviour only | ABS |
| Cai, Portegies Zwart, Podareanu, 2111.15631 | Wisdom-Holman style neural N-body, 3 bodies, 1e5 steps | ABS |
| Poli et al., Hypersolvers, 2007.09601 | cheap learned solvers with accuracy guarantees; residual-correction mechanism not on the abstract page (2ND: local bound only) | ABS (mechanism 2ND) |
| Kochkov et al., ML accelerated CFD, 2102.01010 | learned corrections inside a numerical solver match 8-10x finer resolution, stable; learned interpolation, not an integrator | ABS |
| Schiff et al., DySLIM, 2402.04467 | chaotic systems: add an invariant-measure term because trajectory-only losses diverge over long rollouts | ABS |
| 2502.20212 pseudo-symplectic NN; 2606.30622 SympNet as splitting integrator in particle-in-cell | titles and one-line snippets only | SNIP |
| Horn et al. 2310.20398; Janik and Benner 2509.20212; Bigi and Ceriotti 2508.01068; Shankar 2607.20235; Cattin et al. 2510.06562; Quinn et al. astro-ph/9710043; Pelupessy et al. 1205.5668 | HNN training fragility, symplectic maps with adaptive steps, learned action maps, neural residual falsification at e = .99, DMTS tail failures, individual timestep criteria | 2ND |
| Yoshida 1990; Forest and Ruth 1990; Blanes and Casas optimized splitting; Chin / Omelyan force-gradient schemes | higher-order and force-gradient composition (no arXiv ids given on purpose) | MEM |

Not found: any paper that learns the coefficients of a leapfrog; any learned particle-mesh simulator that studies substeps per tick; any N >= 100 demonstration of a learned symplectic map.

## 6. Constraint check of the recommended action (do not build; stop)

Nothing is built, so no constraint is touched. For the closest viable item, retrained uniform KDK-2 (S1) or RESPA (S2) from `substep-theory.md`: O(N) yes but x2 cost or pp-radius dependence (dropped by user); fails the one-evaluation budget; pass/kill: err@20 cannot reach -20% (bound 2% expB, 11-15% cluster), dE@1000 unproven.

## Pass/kill applied

Stated before running: worth building only if >= 20% better err@20 or nr100 dE@1000 on held-out seeds without failing the regression rows, or KILL the whole direction if integrator error < 10% of total. Outcome: KILL for err@20 (upper bound 2% expB, 11% cluster by the injected-noise oracle, 15% by quadrature; below the 20% gate in both). dE@1000 not killed by data but no 1-evaluation mechanism exists and the 2-evaluation mechanism was measured to fail without retraining and to be confounded with fine-tune drift when retrained. No GPU job was run (another agent holds the GPU).
