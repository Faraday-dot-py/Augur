# Scatter-field vs CentralForceDynamics (CF) gap: root-cause investigation (uncommitted)

Scope: why scatter-field (`ms_kp_pot_v_g128`) is worse than CF at a fixed 900 s budget on Exp B (10-100 bodies, budgetB job 3257: CF 4332 it err@5/10/20/100 .0006/.0011/.0025/.278; scatter 8087 it .0006/.0014/.0039/.374) despite ~1.9x the iterations. Research only. Harness: `scripts/gap_probe.py` (uncommitted), results remote `~/bounce/results/gap`.

## Code facts (read, not measured)

- CF force (model/central_force.py:25-33): f(log d)/(d^2+1) along the pair axis, f = 1->64->64->1 tanh MLP, zero-init output; all-pairs, exactly antisymmetric, no self-force, no grid. Truth is softened a = d/(d^2+eps^2)^1.5, so f(log d) = F(d)(d^2+1) is O(1)-bounded (0 at d=0, ~1 at large d): a 1-D, well-scaled regression. Verlet is 2 force evals/step (same step as the truth integrator family).
- Scatter force (scripts/scatter_field.py:force, ~l.150-175): CIC scatter -> FFT conv with learned isotropic potential kernel K(r) (kmlp on (r, log(r+h)), output is a POTENTIAL ~ -1/r, window-split at pp=2) -> finite-difference gradient (neg_grad, 2h stencil) + U-Net potential residual (acts on the 128^2 grid, recurrent field input) -> CIC gather -> + kNN(16) pairwise MLP g(r)*window(r) for r<2 -> momentum fix. Verlet carries a_prev (1 eval/step).
- Loss/horizon/batch/lr/clip/k-curriculum are the same in both trainers (pos MSE + 0.1 vel MSE, steps 30, k 4->20, batch 16, Adam 1e-3 cosine to 5%, clip 1.0). Differences: scatter loss is body-weighted over a batch, CF is scene-mean; CF trainer uses absolute coords (~500, fp32) and a per-scene python loop; scatter is vectorised and padded.
- Thus the only learnable-structure differences are the force representation (mesh + U-Net + pp vs one 1-D MLP) and integrator eval count.

## Hypotheses and predictions (written BEFORE running)

H1 mesh representational floor. The mesh path (CIC + 2h-stencil gradient, h=0.5=eps at G128/extent 64) is not translation-exact; its error depends on sub-cell phase and cannot be cancelled by r-only learned parts (kmlp, g(r)). Prediction: the UNTRAINED construct with exact analytic K and exact analytic pp complement (`sf_floor`, no learning at all) has teacher-forced force rel error of order 1% (0.5-5%), concentrated at separations 1-4, and err@20 >= 5e-4; i.e. a floor comparable to the budget-run force error (~1%).
H2 optimisation/conditioning of the learned force (potential-valued kernel needing ~1e-4 absolute accuracy; U-Net adds noise on top). Prediction: under direct force supervision (E2, same data/iterations) CF reaches <0.5% rel force error within ~1000 it while sf_full is >=3x worse at equal iterations and flattens higher; ablating the U-Net (`sf_nonet`) lowers far-field error (nn dist > 2 bin).
H3 rollout-gradient path. If E2 (direct force supervision) shows scatter ~= CF but E1 (rollout loss, identical loop) shows a gap, the cause is the BPTT path (recurrent field, long chain), not the representation. Prediction: this is NOT what we see (gap appears in E2 too).
H4 U-Net/recurrence hurts gravity (force is a function of r only; the net can only add noise/self-force). Prediction: `sf_nonet` within +-30% of sf_full at err@20 in E1; far-field force error lower, near-field higher.
H5 physics-prior residual parametrisation (K = -(1+delta)/sqrt(r^2+eps^2) split, g = exact pp complement + learned residual; zero-init so training starts at the mesh floor) removes the learning burden: `sf_prior` should beat sf_full at every iteration count and sit near max(mesh floor, CF).
Gap-shape prediction (E1): if H2 dominates, scatter/CF err@20 ratio shrinks with more iterations (scatter catches up); if H1 dominates the ratio is ~constant (floor) from ~2000 it on.

## Results

Jobs (Polaris, 1 H200 each, seed 4738, Exp B data = 10-100 bodies scale-init, eval seeds 9000/12000, 48 scenes x 100 steps): 3260 failed (device bug), 3261 floors + force-supervision E2, 3262 self-force-subtraction E2, 3266 larger-pp E2, 3267 E1 baseline curves (rollout loss, identical loop for both models), 3268 E1 fix candidates; 3263 cancelled/relaunched as 3267. Raw: remote `~/bounce/results/gap{,2,3,_e1,_e1b}`, local copies `~/polaris-mcp-files/gap/`. Summaries: `scripts/gap_summarize.py`. All CF numbers use `CFDense` (all-pairs masked, same MLP/step as CentralForceDynamics; max rel accel diff vs the original class 1.3e-4) in the scatter trainer's loop (same data, loss, batch 16, k 4->20, cosine lr, 1e-3, clip 1).

### 1. CF is at the integrator floor from ~300 iterations; the "gap" is not a CF efficiency win

Exact analytic force + the CF integrator (2 force evals, dt .1) vs the 4-substep truth, Exp B seed 9000 (job 3261, `exact_v2`): err@5/10/20/50/100 .0006/.00112/.00222/.0135/.246. Budget-run CF (job 3257): .0006/.0011/.0025/.015/.278. Control in my loop (CF, 4000 it): .0006/.00114/.00253/.0154/.279 (reproduces the budget run, so the loop/loss/data differences are not a confounder). CF loss reaches its floor (~5e-6) by iteration ~300; err@20 is .0026 at 1000 it and .0025-.0028 at 2000-8000 (flat, slightly worse). A force that is ONLY a function of pair distance cannot beat the integrator error; CF is saturated at that floor. Also the original CF trainer ran 4.8 it/s (per-scene python loop); the vectorised CFDense runs ~47 it/s, so the 900 s budget under-used CF about 10x in iterations, which did not matter.

### 2. Scatter does not converge slowly, it converges to a worse, mesh-limited force (E1, E2)

E1 rollout curves, Exp B, err@20 (seed 9000 / 12000), iterations of a full cosine schedule:

| model | 1000 | 2000 | 4000 | 8000 | err@100 at 8000 |
|---|---|---|---|---|---|
| CF (CFDense) | .0026 | .0026 | .0025 | .0028 | .293 |
| sf_full (`ms_kp_pot_v_g128`, budget-run model) | .0072 | .0065 | .0055 | .0041 (12000: .0041) | .389 |
| sf_prior_pp8 (fix, see 4) | .0031 | .0030 | .0019 | **.0018** (12000: .0019) | **.201** |
| sf_prior_pp8_nonet (no U-Net) | .0031 | .0031 | .0031 | .0032 | .326 |

sf_full improves ~log-linearly (.0072 -> .0041 over 8x iterations) and at 8000 it (= the budget-run iteration count, .0039) is still 1.6x above CF; extrapolation is untested beyond 8000 (earlier Exp A history: 4k .0049, 12k .0030, 30k .0015, 60k .0009).

Direct force supervision (E2: teacher-forced loss on analytic acceleration, same data, batch 16, 4000 it, cosine; relative RMS acceleration error on held-out states, by the body's nearest-neighbour distance). This removes the rollout/BPTT path entirely:

| model | 250 it | 1000 it | 4000 it | bins at 4000: <.25 / .5-1 / 1-2 / >2 |
|---|---|---|---|---|
| CF | .0115 | .0018 | **.0009** | .0025 / .0002 / .0002 / .0004 |
| sf_full | .0221 | .0090 | .0071 (plateau) | .0060 / .0059 / .0116 / .0119 |
| sf_nonet (kernel + pp only) | .0248 | .0114 | .0098 | .0084 / .0080 / .0166 / .0153 |
| sf_norec | .0236 | .0239 | .0068 | .0058 / .0056 / .0111 / .0115 |

CF's force error falls 12x between 250 and 4000 it; sf_full falls 3x and flattens at ~0.7% (8x CF). The plateau is in the 1-6 unit separation range (bins 1-2 and >2: 1.2%), not at contact.

### 3. Mechanism: the mesh/pair handoff at r ~ pp is inaccurate by (h/r)^2, and an r-only learned kernel cannot cancel the phase-dependent part

- Untrained, exact-analytic mesh (analytic split kernel, analytic PP complement, no learned parts; `sf_floor`, job 3261): force rel error 11.5% in the > 2 bin; rollout err@5/10/20 .0087/.030/.093. This is pure CIC + 2h-stencil-gradient + FFT discretisation, with h = 0.5 = eps and the handoff at pp = 2 = 4h. Self-force subtraction (`selfsub`, exact P3M-style) changes nothing to 8 digits (job 3262): CIC-in/CIC-out with a symmetric stencil already has exactly zero self-force (the 16 stencil terms are antisymmetric), so self-force is NOT the cause (my initial suspicion; untrue).
- Error scales as (h/r)^2: with the handoff moved out (exact K + exact pp complement, untrained): pp=2 11.5%, pp=4 3.3% (> 2 bin), pp=6 1.2%, pp=8 0.58% (jobs 3262, 3266). Training only the pp residual cannot fix it (frozen exact K, learn g: > 2 bin stays 11.5% over 4000 it): the error lives in the mesh far-field region r in [pp, ~3 pp].
- A learned kernel K(r) reduces the isotropic average error (sf_full: 1.2% in the > 2 bin), but the remainder depends on sub-cell phase/orientation, which a function of r cannot express; the U-Net must cancel it from grid data and does so slowly, leaving the observed plateau (~0.7%).
- The U-Net's contribution is slow and partly harmful: E1 sf_prior_pp8 at 1000/2000 it equals the no-net version (.0031), then drops to .0019 by 4000 and .0018 by 8000, below the exact-force integrator floor .0022; so with the mesh error removed at the source the U-Net is free to learn a small integrator/dynamics correction. (Interpretation untested: I did not isolate what the U-Net learned.)
- Scatter excess error in rollouts is broad, not close-encounter dominated: share of err^2@20 in bodies whose min NN distance (steps 0-20) is < .25: CF 72-83%, sf_full 41-54%; share in 1-2: CF 3%, sf_full 16-19%. Per-scene err@20 percentiles (p10/p50/p90): CF .0021/.0034/.0067, sf_full@8000 .0041/.0052/.0064 (the whole distribution is shifted, the tail is not worse).
- Teacher-forced 1-step accel error with exact forces (integrator only) is 7.7% in the < .25 bin, 0.1% in > 2 (job 3261 `exact_v2`): close encounters are integrator-limited for every model; the mesh limitation is at moderate separations.

### 4. Fix test: physics-prior residual parametrisation + larger exact handoff radius

Variant `sf_prior_pp8` (`scripts/gap_probe.py`, class `SFx`): kernel K = -(1+kmlp)/sqrt(r^2+eps^2) x split window (zero-init so it starts exact), PP term = analytic complement of the mesh + learned residual, pp = 8, kNN k = 48, U-Net kept. E2 force error 0.11% at 4000 it (vs sf_full 0.71%, CF 0.09%); untrained start already 0.69%. E1 result in the table above: err@20 .0019 at 4000 it (CF .0025), .0018 at 8000 it, err@100 .20 vs CF .29 and sf_full .39. At 900 s (~6800-7000 it at 7.7 it/s) expect ~.0018-.0019 (inferred, not run at the time budget). No-net version: .0031 flat (mesh residual only), ~CF-class but not better than CF.

## Hypothesis scorecard (against predictions above)

| H | prediction | outcome |
|---|---|---|
| H1 mesh floor | untrained exact mesh ~1%, >= 5e-4 | CONFIRMED in direction, wrong in size: 11.5% at pp=2 (err@20 .093); mechanism is (h/r)^2 at the handoff, not self-force |
| H2 optimisation of learned force | CF <0.5% in 1000 it, sf >=3x worse, flattening; U-Net removal lowers far-field error | PARTLY: CF .18% at 1000 it, sf 5x worse and flat (confirmed). U-Net removal did NOT lower error (nonet worse, .0098) |
| H3 rollout/BPTT path | not the cause | CONFIRMED: gap present under direct force supervision (E2) |
| H4 U-Net/recurrence hurts | nonet within +-30% | REFUTED for sf_full-class (U-Net is needed to cancel mesh error: .0098 vs .0071); with the mesh fix it adds little for 4000 it then helps below the integrator floor. Cold-field force error of recurrent models is large (sf_full@8000 10.6%, prior_pp8@8000 12.5% at a cold recurrent state) while rollouts are fine: the force depends on the recurrent state (untested whether this matters for long rollouts or delete-the-star causality) |
| H5 physics-prior residual | beats sf_full at every iteration count | CONFIRMED, only together with a larger pp: `sf_prior_ss_pp4` E2 .47%, pp8 .11%; prior with pp=2 is worse (.0117, starts at 14%) |

## Root causes, ranked

1. (high confidence, measured) Mesh handoff error: far-field force from CIC + FD gradient at h=eps=0.5 with handoff pp=2 is wrong by ~11% untrained and ~1% after learning (E2 plateau), because the error is phase/orientation dependent and the learned kernel and PP MLP are functions of r only. Evidence: floors in jobs 3261/3262/3266, E2 plateaus, E1 curve slope.
2. (high) CF is not "more sample efficient"; it is at the integrator floor from ~300 it (exact-force err@20 .0022 vs CF .0025), so any model with force error > ~0.5% shows as a gap and a model with force error 0.1% matches it. The earlier "sample-efficiency" label conflates iteration count with a representational plateau (E2 plateau, E1 8000-it still 1.6x).
3. (medium) The remaining scatter learning burden is slow because the U-Net must discover the physics-prior (K ~ -1/r, exact PP complement) from scratch: with the prior parametrisation the same 1000 iterations give .0031 instead of .0072.
4. (low/untested) Recurrent field makes the force state-dependent; not shown to hurt the rollouts.

## Fix candidates (ranked)

| # | change | expected effect | cost | status |
|---|---|---|---|---|
| 1 | Analytic-prior residual kernel + analytic mesh-complement PP term (zero-init, starts at the exact split force) | converged force error x6 lower from iteration 0; err@20 .0072 -> .0031 at 1000 it | ~40 lines in `ScatterField` (the `SFx` class is a working draft) | measured |
| 2 | Raise the exact-pair radius pp from 2 to ~8 (kNN k 32-48), keep mesh as true far field | mesh handoff error 11.5% -> 0.6% (untrained); with 1 the E2 force error 0.11% | kNN cost grows ~linearly with k; needs a cell list for large N (current knn is a dense N^2 prototype; N=100 only here) | measured at N<=100 only; scalability untested |
| 3 | Keep the U-Net, train >= 4000 it | err@20 .0019 at 4000 it, err@100 .20 (beats CF .0025/.29 via apparent integrator compensation) | none | measured, U-Net compensation mechanism untested |
| 4 | Finer mesh (G=256 or h=0.25) instead of 2 | mesh error ~/4 at fixed pp; 4x grid cost | untested | untested |
| 5 | Phase-aware mesh correction (e.g. dipole moments or sub-cell features into the U-Net) | would address the phase-dependent residual directly | design work | untested |

Not fixes (shown): self-force subtraction (no effect), removing the recurrent field (same plateau), removing the U-Net with the old pp=2 (worse).

## Untested / caveats
- One training seed (4738) per cell; differences under ~20% are noise (sf_full err@20 is stable across iteration counts within that, the CF vs prior_pp8 gap at 8000 it, .0028 vs .0018, is ~35%, with seed-12000 agreeing).
- Exp B only (N 10-100, unit mass). Exp A and Exp C (mixed masses; prior fix assumes unit-mass-scaled K, mass is multiplied through the scatter so it should carry but was not run) not run.
- Fix 2 was tested with the dense N^2 kNN prototype; behaviour at 1000+ bodies, and dense clusters where >48 bodies lie within pp, is untested (kNN truncation would break the handoff).
- 900 s wall-clock re-run of sf_prior_pp8 not done (inferred from iteration curves).
- No rollout videos produced (no unbiased visual review needed); recurrent cold-field behaviour, delete-the-star causality of sf_prior_pp8 not re-checked.
- Throwaway: `scripts/gap_probe.py`, `scripts/gap_summarize.py` (uncommitted).
