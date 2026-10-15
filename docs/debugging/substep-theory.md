# Substep theory for the scatter-field simulator (2026-10-14, uncommitted)

Theorist pass. No training, no GPU. New numbers come from one exact-force fp64 CPU script, `scripts/scatter_substep_theory.py` (Polaris debug partition, jobs 3614 + 3615, 158 s + ~20 min; raw `results/substep/substep_theory.json`, `substep_theory_b2.json`; Polaris copy `~/augur-rc/results/substep/`). Scenes: expB 24 scenes seed 9300 (N 15-87), cluster 12 scenes seed 9200 (N 84-269), as in long-rollout-root-cause.md. Truth = fp64 KDK, 4 substeps/tick. The Verlet-1 rows reproduce the root-cause numbers exactly.

Definitions: N = bodies per scene; B = batch; dt = 0.1 (one tick); eps = 0.5 softening; nn = distance to nearest neighbour; v_rel = speed relative to the nearest neighbour; M = substeps per tick; f = promoted fraction (bodies given substeps); rc = radius of the "fast" pair force; err@k = pooled mean per-body position error at tick k; e1 = one-step position error of 1 Verlet step from a truth state; "Verlet-1" = the model's integrator (scatter_field.py:504-506) with the exact force; RESPA = multi-rate KDK (slow force half-kicked at tick start and end, fast pair force kicked on M substeps).

## Summary

1. The 1-step-per-tick integrator is not under-resolved anywhere in these scenes. It is in the asymptotic O(dt^3) regime in every nn bin. Two substeps cut e1 by 0.18-0.20 in every bin (theory: 0.25). dt^3|jerk|/6 predicts e1 within 6-19% (median), Spearman 0.99 / 0.96. Softening caps the free-fall time at sqrt(eps^3/2) = 0.25 = 2.5 dt. The floor is a magnitude problem. In iid-force-noise units the Verlet-1 floor is worth about 0.7% relative force error in expB and about 2% in clusters.
2. Trajectory errors add in quadrature: err@20^2 ≈ F^2 + I^2, where F is the force-error part and I is the integrator part (the root-cause noise rows fit this within about 5%). For E, I is 0.8% of squared err@20 in expB and 27% in clusters. Fixing the integrator therefore cannot help E on expB (≤1%) and can help at most about 15% on clusters. It matters for force errors ≤1-2% (B ckpt, untrained pp=8 P3M "pairfull0"). Pairfull0 in clusters sits at the floor (.037 vs .035).
3. Per-body adaptive substeps are a poor deal. Error is spread widely: nn<1 holds 98% of squared e1 but covers 43% of expB bodies and 75% of cluster bodies. A full force needs a global grid solve, so promoting bodies does not avoid the cost. Small promoted sets (nn<0.5) were worse than Verlet-1 at @50/@100. Naive variants without symmetric pair kicks or with a frozen slow force were also worse than Verlet-1.
4. The winning design is uniform symmetric RESPA. Mesh plus U-Net (slow) is evaluated once per tick and half-kicked at both ends. The pair term within rc (fast) is re-evaluated on M substeps for all bodies. Cost is 1 grid eval plus M pair passes. Exact-force err@20 for expB / cluster:
   - Verlet-1: .0023 / .035
   - uniform KDK2 (2 grid evals): .00045 / .0052
   - RESPA rc2 M4: .00065 / .0063
   - RESPA rc4 M4: .00027 / .0028
   - RESPA rc8 M4: .0001 / .0014
5. Predictions (untested):
   - RESPA rc8 M4 on pairfull0: cluster err@20 .037 → .010-.015, expB .0050 → ~.0044.
   - Any substep scheme on E without retraining: no gain (E sub2 already measured +17% err@20, nr100 dE x40).
   - Higher-order schemes (Yoshida4, jerk-extrapolated drift) are worse than Verlet-1 in clusters. Reject them.
6. First test: inference-only `sub2`/`sub4` on pairfull0. It has no net, so field cadence and absorbed modified force are excluded. Kill criterion: cluster err@20 improves <20%.

## 1. Error budget per tick

### 1.1 Timescales (theory)
- Softened pair force F(r) = r/(r^2+eps^2)^1.5. Its maximum is 1.54 at r = eps/√2. Inside the core a bound unit-mass pair is harmonic with ω^2 = 2m/eps^3 = 16, so ω = 4, ω·dt = 0.4, and the period is 15.7 ticks. Verlet is stable for ω·dt < 2. At ω·dt = 0.4 the phase error is about (ω dt)^3/24 = 0.0027 rad/tick and the energy oscillation about (ω dt)^2/8 ≈ 2%. Resolved, but not accurate.
- Free-fall time tau_ff = sqrt(s^3/(2m)) with s = sqrt(nn^2+eps^2). Its floor is 0.25 = 2.5 dt.
- Encounter time tau_enc = s/v_rel. It can drop below dt (fast flybys), but the force still varies smoothly on scale eps.
- Verlet local error per tick: position dt^3 j/6 (j = jerk = da/dt), velocity O(dt^3 · snap/12). The global error is O(dt^2), but in chaotic scenes the per-tick seed is what is amplified (error-growth-diagnostic.md).

### 1.2 Measured scene statistics (truth states, ticks 0-99; quantiles .05/.25/.5/.75/.95/.99)

| | expB | cluster |
|---|---|---|
| nn | .24 / .64 / 1.15 / 2.0 / 4.2 / 7.9 | .08 / .22 / .45 / 1.0 / 3.4 / 7.1 |
| v_rel | .56 / 1.3 / 2.05 / 3.0 / 4.9 / 6.6 | .66 / 1.8 / 3.6 / 6.7 / 12.2 / 16.5 |
| \|a\| | .19 / .45 / .89 / 1.6 / 3.0 / 4.3 | .55 / 2.0 / 5.2 / 12 / 22 / 29 |
| tau_enc | .15 / .33 / .66 / 1.4 / 4.6 / 11 | .046 / .097 / .22 / .58 / 3.0 / 8.0 |
| e1 (Verlet-1, one tick) | 6e-6 / 5e-5 / 2.3e-4 / 7.1e-4 / 2.5e-3 / 4.7e-3 | 2.6e-5 / 4.4e-4 / 2.2e-3 / 7.1e-3 / .024 / .047 |
| frac nn<.25 / <.5 / <1 / <2 | .054 / .18 / .43 / .74 | .29 / .54 / .75 / .89 |
| frac tau_enc < dt / 2dt / 4dt | .01 / .10 / .32 | .26 / .47 / .67 |
| frac tau_ff < 2dt / 4dt | 0 / .16 | 0 / .51 |

### 1.3 One-step error by nn bin (exact force)

| nn bin | expB frac | expB share of Σe1^2 | expB share of Σe1 | cluster frac | cluster share of Σe1^2 | e1(M=2)/e1(M=1) | position bias needed, % of \|a\| (expB / cluster) | velocity bias needed, % of \|a\| | median cos(bias_x, bias_v) |
|---|---|---|---|---|---|---|---|---|---|
| <.25 | .054 | .45 | .24 | .29 | .84 | .19 | 28 / 16 | 7.4 / 7.4 | .81 / .68 |
| .25-.5 | .12 | .36 | .32 | .24 | .14 | .19 | 13 / 10 | 4.0 / 4.2 | .23 / .42 |
| .5-1 | .26 | .17 | .31 | .21 | .02 | .19 | 8.1 / 6.4 | 1.3 / 1.5 | .26 / .34 |
| 1-2 | .31 | .022 | .11 | .14 | .0014 | .20 | 4.1 / 3.1 | .36 / .33 | -.11 / -.10 |
| 2-4 | .20 | 5e-4 | .014 | .07 | 4e-5 | .20 | 1.4 / 1.3 | .06 / .06 | -.35 / 0 |
| >4 | .058 | 5e-6 | 7e-4 | .04 | 6e-7 | .20 | .4 / .6 | .007 / .01 | .11 / .89 |

The ratio e1(M=2)/e1(M=1) is constant at 0.18-0.20 in every bin, including clusters, where 26% of bodies have tau_enc < dt. No regime breaks down. The per-tick error is the leading Taylor term everywhere, and it is concentrated by magnitude (jerk ∝ v_rel/eps^3 near the core) rather than by a resolution threshold.

Capture curves: the share of Σe1^2 (and of Σe1) held by the top-f bodies under each ranking.

| ranking | f=.05 | .1 | .2 | .3 | .5 |
|---|---|---|---|---|---|
| expB nn ascending | .42/.23 | .65/.39 | .84/.60 | .92/.75 | .99/.91 |
| expB jerk estimate dt^3\|j\|/6 | .61/.30 | .79/.47 | .92/.68 | .97/.81 | 1.0/.94 |
| expB oracle (true e1) | .66/.32 | .82/.49 | .93/.70 | .97/.82 | 1.0/.94 |
| cluster nn ascending | .23/.15 | .42/.28 | .69/.51 | .85/.68 | .97/.88 |
| cluster jerk estimate | .57/.29 | .73/.45 | .88/.65 | .95/.79 | .99/.93 |

The jerk estimate is nearly oracle-quality as a selector. 90% of squared e1 sits in f ≈ .19 (expB) / .22 (cluster) by jerk, and .27 / .35 by nn. Section 3 shows why that does not translate into a cheap adaptive scheme.

### 1.4 Calibration of the floor in force-noise units
The root-cause noise rows (exact + iid relative force noise σ, Verlet-1) fit err@20^2 = (c σ)^2 + I^2:
- expB: c ≈ .0032 per 1% (1/2/4/8% → .0033/.0066/.0128/.0255 after removing I = .0023).
- cluster: about .012-.017 per 1%.

So the Verlet-1 floor equals about 0.7% iid force noise in expB and about 2% in clusters. E has 6.1% / 6.5%. Integrator share of squared err@20 (Verlet-1 floor vs measured err@20 of each model):

| model | expB | cluster |
|---|---|---|
| E | .008 | .27 |
| B ckpt | .33 | .05 |
| pairfull0 (untrained P3M pp8) | .21 | .89 |
| pairfull (trained) | .02 | .53 |

At @50/@100 cluster errors are chaos-saturated (cluster Verlet-1 .97 @50), so quadrature is unreliable there. @20 is the decision horizon. Reference ambiguity: the truth itself moves by .00011 (expB) / .0006 (cluster) @20 when its substeps are doubled (scatter-chaos-floors.md). Any oracle below that is "at truth".

## 2. Interaction with the learned net (code reading)

How substeps run today:
- `apply_opt("subN")` (scatter_field.py:1085-1087) sets `model.substeps`.
- `step` (487-494) loops `_step` N times with dt/N.
- `_step` (496-507) is velocity Verlet. It takes `a_prev` from the state tuple, drifts with `pos + dt*vel + .5 dt^2 a_prev` (504), calls `force` once at the new position with predicted velocity `vel + dt*a_prev` (505), and kicks with `.5 dt (a_prev + a_new)` (506).

What happens when substeps are added at inference without retraining:

| concern | what the code does | consequence |
|---|---|---|
| Grid deposit staleness | None. Every `force` call re-scatters at current positions (`_force` 446) and re-runs kernel FFT, U-Net and gather (450-474), pair term (480-481) and momfix (482-484). | No stale field, but each substep is a full global grid solve: cost N× per tick. No local or partial path exists. |
| a_prev semantics | `a_prev` = force from the previous call at the current positions (507). With substeps it is the previous substep's end force. At tick 0, `a_prev is None` triggers one extra eval (502-503). | Consistent: identical to KDK with a carried force. Not a problem. |
| Velocity input | `force` receives `vel + dt*a_prev` with the substep dt (505), deposited as momentum channels (161). | Consistent. |
| Recurrent field cadence | The U-Net output is fed back as input on every `force` call (453-454; E: `potential=True`, cf = 1 channel, the potential itself, 471). | With N substeps the net sees a field that is dt/N old rather than one tick old, and iterates N× per tick. Training always iterated once per tick (train_scatter_field.py:170-171; `make_step_fns` 520-543 graphs `m.step`, so it would follow `substeps` if set at training). The recurrent ablation shows the field sets <5% of accuracy, but feedback gain is weight-specific (C weights ran away). Effect size untested. |
| Absorbed integrator error ("modified force") | Training loss is position + 0.1·velocity squared error at tick ends of the free Verlet-1 unroll (train_scatter_field.py:173-175), from a cold field per window (166). The optimum force is the one that makes Verlet-1 at dt = .1 match truth, not the physical force. | A dt-specific bias is over-applied when dt shrinks. Measured: E + sub2 without retraining gave err@20 +17% and nr100 dE@1000 .02 → .86 (scatter-fixes-2026-10-06.md:22). Fine-tuned with sub2 (ft_sub2, job 3370) it gave err@5-20 -20 to -25% vs baseline and nr100 dE .030 vs same-schedule ctrl .29 (line 26), which is confounded by fine-tune drift. |

### Can a force absorb the Verlet-1 error?
A tick-constant bias b changes position by .5 dt^2 b and velocity by dt·b. Fixing the position error needs b_x = 2δx/dt^2; fixing the velocity error needs b_v = δv/dt.
- Measured (table 1.3): |b_x|/|b_v| has a median of 3.5-7 for nn<1 and 10-50 beyond. The directions are poorly aligned (median cos .23-.81 for nn<1, ≤0 beyond).
- A single force bias therefore cannot cancel both. With the trainer's weights (one-step position term ~(.5 dt^2)^2 = 2.5e-5 vs 0.1·dt^2 = 1e-3), the optimum leans to b ≈ b_v.
- b_v is 7.4% / 4% / 1.3% of |a| for nn < .25 / .25-.5 / .5-1. That matches the size of E's near-field excess over B (E 9.9 / 6.5 / 5.2% vs B 1.0 / 0.9 / 0.9%) and E's two-body bias (-20% at r = .1).
- Hypothesis (untested): part of E's near-field "error" is absorbed velocity-error compensation. It cannot be all of it: B reaches 1% on expB without absorbing.
- Direct test: substep E's force on truth states and measure whether the near-field bias tracks b_v (eval only).

### Design implications for training
- With a fixed numerical substep scheme, tick-end truth targets are enough. Substeps are internal and the loss is unchanged (solver-in-the-loop); gravity_sim.rollout_torch already computes the 4-substep states (83-87) if dense targets are wanted.
- Dense substep targets only pay for a learned corrector (section 3, S4).
- Make the fast force analytic or pair-local and exact (prior P3M pair term). The net then cannot absorb near-field integrator error, because the near force is integrated accurately, and an absorbed bias would be wrong at every M.
- Train from scratch or from the analytic base. Do not fine-tune E: every U-Net fine-tune fails nr100 dE (E1).

## 3. Substep architectures

Cost unit: G = one grid pass (scatter + kernel FFT + U-Net + gather); P = one pair pass over all bodies at the model's pair radius. Exact-force oracle numbers are measured (jobs 3614/3615). Predictions for learned models are untested and use the quadrature model with F from the root-cause tables.

### Naive variants (measured, exact force): what not to do

| variant (exact force) | expB err@20 / @50 / @100 | cluster @20 / @50 | dE@100 expB / cluster | dP |
|---|---|---|---|---|
| Verlet-1 | .0023 / .013 / .25 | .035 / .97 | .0035 / .036 | 0 |
| promoted bodies on M=4 full-force substeps, other bodies on predicted drift, one-sided kicks: nn<.5 (f .19 / .50) | .0033 / .031 / .56 | .015 / .41 | .040 / .074 | 1e-3 |
| same, nn<1 (f .45 / .72) | .0011 / .0073 / .13 | .0020 / .058 | .0016 / .0013 | 3e-4 |
| frozen far + near pair (rc2) recomputed on substeps, far applied as start value (not half-kicks), nn<1 | .0048 / .047 / .77 | .10 / 2.4 | .061 / .47 | 2e-3 |
| same, all bodies | .011 / .085 / 1.0 | .11 / 2.4 | .075 / .45 | 0 |
| jerk-extrapolated drift (+dt^3 j/6), backward-difference / exact jerk, 0 extra evals | .0038 / .0029 @20 | .066 / .061 | .029 / .026 (expB) | 0 |
| Yoshida-4 (3 evals) | .00062 / .0078 / .20 | .045 / 1.3 | .0057 / .09 | 0 |
| uniform KDK3 (3 evals) | .00012 / .00059 / .013 | .0013 / .063 | .0001 / .0004 | 0 |

Lessons:
1. A frozen slow force must be applied as symmetric half-kicks at tick start and end. A start value held through the tick is first order and is worse than Verlet-1 by 2-5x. D4's null "RESPA on E's pair term" is consistent with both this and E's force-error dominance; its kick placement was not re-checked.
2. Pair kicks must hit both bodies. One-sided promotion violates Newton 3 (dP 1e-3) and raises dE about 10x.
3. Higher-order and extrapolated schemes lose to plain KDK at equal cost and fail in clusters.
4. The "extrap" variant in job 3614 had a bug (it extrapolated by zero) and is excluded.

### Proposals

**S1. Uniform KDK-M, full evals (existing `subN` opt), trained with substeps unrolled.**
- Mechanism: as now.
- Cost: M·(G+P).
- Consistency: momentum exact via momfix; symplectic for exact force.
- Failure modes: N× grid cost; recurrent field iterated M× per tick; requires retraining.
- Oracle: M2 .00045 / .0052, M3 .00012 / .0013 (expB / cluster @20).
- Prediction (pairfull base retrained with M2 vs ctl): expB @20 ≈ ctl (F-dominated, ≤ -5%); cluster @20 -25 to -35% (.035 → ~.025); cost 2.0x.
- Use it as the control for S2.

**S2. Uniform symmetric RESPA, slow = grid, fast = model pair term (recommended).**
- Mechanism, per tick:
  1. v += dt/2 · a_slow(t0), where a_slow = full - fast is kept from the previous tick end.
  2. M substeps: half-kick by fast, drift all bodies, recompute the fast pair force (`pp_acc` over pairs within rc = pp), half-kick by fast.
  3. One full `_force` at the tick end gives a_slow(t1) = a_full - fast. Then v += dt/2 · a_slow(t1).
- Code needed: `_force` returns grid and pair parts separately (now summed at 472-481); momfix (482-484) applied to the slow part; state carries a_slow instead of a_prev.
- Cost: G + (M+1)·P (M·P if the start-of-tick fast force is reused).
- Fast pairs per body, measured (expB / cluster): rc1 .75 / 5.8, rc2 2.4 / 18, rc4 7.1 / 47, rc8 18 / 94. The cost grows with ρ·rc^2; cap it with knn (with pp8 knn=48 truncation, the old `pair` arm was negative) or tie rc to local density (untested).
- Consistency:
  - Momentum: exact when the pair term is untruncated (pair forces antisymmetric for equal masses, `_pair_w` 208-217); with knn truncation, momfix only.
  - Splitting: fixed within the tick, so time-reversible for a fixed split.
  - Resonance: impulse-MTS risk when a pair period approaches 2 dt. The core period is 15.7 ticks, so it is not reached here; untested in denser scenes.
- Oracle err@20 (expB / cluster; dE@100):

  | rc | M2 | M4 | dE@100 (M4) |
  |---|---|---|---|
  | 1 | .0015 / .017 | n/a | n/a |
  | 2 | .00085 / .0089 | .00065 / .0063 | .00046 / .002 |
  | 4 | .00057 / .0062 | .00027 / .0028 | .00017 / .0006 |
  | 8 | .00047 / .0053 | .0001 / .0014 | .0001 / .00026 |

  rc8 M4 is better than uniform KDK2 at one grid eval.
- Predictions (untested):
  - pairfull0 + S2 rc8 M4: expB @20 .0050 → .0043-.0046; cluster .037 → .010-.015; harness nr100 dE@1000 improves vs pairfull0's own Verlet-1 (exact Verlet-1 dE@100 .0035 → .0001).
  - Trained pairfull + S2 (from scratch, same schedule as ctl, loss through the RESPA tick): cluster @20 .048 → .033-.040, expB ≤ -5%.
  - E architecture (pp=2, learned pair) with S2 rc2 M4 retrained: expB unchanged within noise, cluster ≤ -15%.
  - Cost: pairfull pair pass at knn400 is roughly 0.4 G by training-time ratio (761 vs 537 s; untested as a per-pass number), so M4 ≈ 3x tick time. For pp2 with knn16, P ≪ G, so ≈ 1.1-1.3x (untested).

**S3. S2 with per-body promotion (pair-symmetric; fast set = pairs touching a promoted body).**
- Oracle err@20 (expB / cluster):

  | promotion rule | f (expB / cluster) | err@20 | cost vs all-bodies S2 |
  |---|---|---|---|
  | nn<1, rc4, M4 | .45 / .72 | .00092 / .0032 | about 80% of the pair evals |
  | nn<.5, rc4, M4 | .19 / .50 | .0031 / .0085 | worse than Verlet-1 at expB @50/@100 (.027 / .49), dE .024 |
  | jerk > 1e-4 | .67 / .86 | .00041 / .0029 | |

- Promotion switching between ticks breaks reversibility, and the pair cost saved is small because the fast-pair count is set by the dense bodies anyway.
- Prediction: never better than S2 at equal cost on expB or cluster. The only use is a hard density cap (top-k per tile, MoD-style). Rank 4.

**S4. Learned corrector on Verlet-1 ("hypersolver"; dense one-step labels from truth substeps).**
- Mechanism: after the step, Δx, Δv = dt^3·g(pair features: r, v_rel, relative acceleration). Labels are truth minus Verlet-1 on truth states.
- Cost: about 1 P, no extra G.
- Theory: the corrector must output position and velocity separately (b_x ≠ b_v, section 2). The position part is already analytic (dt^3 j/6, Spearman .99), and applying it alone is worse than Verlet-1 over a rollout (jerk-drift rows: non-symplectic, dE x8). A learned version inherits that unless it carries the velocity correction and an energy structure. D4's learned residual cut one-step MSE by 44% with a null rollout and failed dE.
- Prediction: one-step e1 -50 to -80%, err@20 ≤ -20% in clusters, fails nr100 dE. Rank 5.

**S5. Higher-order uniform schemes (Yoshida-4, Hermite with jerk from the field or backward differences).**
- Measured: worse than KDK3 at 3 evals; worse than Verlet-1 in clusters (.045 vs .035); jerk drift worse everywhere. With learned force noise the extra order is wasted.
- Reject. Rank 6.

**S6. Global KDK2 with frozen grid (= S2 with rc = ∞): the reference for S2's split error.**
- Oracle equals KDK2 (.00045 / .0052). Not deployable at rc = ∞ (O(N^2)); it only bounds S2.

Ranking: S2 > S1 (control) > S3 > S4 > S5. S6 is a reference only.

## 4. Experiment plan (ranked; each a single change vs a same-schedule control; held-out seeds expB 9300 / cluster 9200 plus harness; seed 4738)

1. **Inference-only `sub2`/`sub4` on pairfull0** (existing opt, zero code). pairfull0 has `nonet=True`, so there is no recurrent field and no absorbed bias.
   - Prediction: cluster @20 .037 → .012-.016 (sub2), .010-.014 (sub4); expB .0050 → .0044-.0046.
   - Kill: cluster @20 improves <20% → the integrator floor does not matter even at 0.5% force error; drop all substep work.
2. **Same on the B ckpt (expB scenes) and on E (sub4 added to the known sub2).**
   - Prediction: B expB @20 .0040 → .0033-.0036; E no gain or worse (as sub2's +17%).
   - Kill for "substeps help a trained net without retraining": any E gain >10%.
3. **Eval-only test of absorption:** on truth states, compare E's and pairfull's near-field force bias (by nn bin, projected on b_v from table 1.3) at the bins of section 1.3.
   - Prediction: E's bias correlates with b_v (cos > .3 for nn<.5); pairfull0's does not.
4. **S2 inference path on pairfull0 (needs the grid/pair split of `_force`)**, rc = 8, M = 2/4 vs pairfull0 Verlet-1 and vs sub2.
   - Prediction: M4 cluster @20 ≤ sub2's; tick time ≤ 3x.
   - Kill: S2 M4 worse than sub2 at lower cost, or harness dE rows worse than pairfull0.
5. **Training: pairfull base, loss unrolled through S2 (M4, rc8) from scratch, vs the identical pairfull ctl** (4000 it, k 4→20, lr 1e-3 cosine).
   - Prediction: cluster @20 -15 to -30%, expB within ±10%, harness PASS, nr100 dE@1000 ≤ ctl .057.
   - Kill: cluster @20 > 0.85x ctl, or any harness dE row worse than ctl.
   - Run after the robust-loss experiment, so the force term is not the moving part.
6. **Only if S2 passes and density cost bites:** S3 with a top-k per-tile budget, vs S2 at an equal pair budget.
   - Kill: err@20 worse than S2 by >10% at equal cost.

### What would refute "the integrator floor matters for E"
- The numbers already refute it for expB: the floor is 0.8% of E's squared err@20, so the predicted gain is ≤0.4%.
- For clusters (27% share; ≤ -15%), it is refuted if a retrained E-architecture with S1 M2 or S2 shows cluster err@20 within 5% of its same-schedule Verlet-1 control.
- The cheaper refutation is to run the existing exact+8% noise oracle with KDK2 instead of Verlet-1. If cluster err@20 stays at .10 ± .005, the floor is irrelevant at E's force error. Prediction: .095-.097.
- Conversely, the floor matters for the next model only if experiment 1 passes.

## Caveats
- Exact-force oracles only. Learned-model numbers are predictions from the quadrature model, which was fitted at @20 and not checked at @50/@100.
- Single scene draws (24 expB, 12 cluster). Unit masses only; star/BH/relativistic scenes not run.
- The oracle's smooth split window (1-(r/rc)^2)^2 matches the shape of the model's split (scatter_field.py:192, 210-211), but the model's split is between learned kernel, U-Net and pair head; E's U-Net may carry short-range force that S2 would freeze.
- The pair-pass cost ratio P/G is inferred from training wallclock, not measured per pass.
- Literature reconciled with substep-literature.md:
  - Its design 1 = S2, confirmed numerically here, with the caveat that half-kick placement and symmetric pair kicks are required.
  - Its design 2 = S3, not supported (selection buys little).
  - Its design 3 = S4, ranked low by the b_x/b_v mismatch.
  - Its design 4 (high-order for close components) is related to S5, which failed in its uniform form; not tested per component.
  - Its "not recommended" list (Yoshida, Hermite) is confirmed.
