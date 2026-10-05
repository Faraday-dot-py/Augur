# Scatter-field scene generalization (plan agent 3), 2026-10-04

Branch `scatter-scenes` (from `scatter-opt`). Code: `scripts/scatter_scenes.py` (scene generators, truth, metrics, eval/train), `scripts/scatter_bounce.py` (opt-in bounce extension), `scripts/render_scatter_scenes.py`, `scripts/polaris_scatter_scenes.sh`. Results `results/scenes/`, videos `videos/scenes_*`.

Protocol (user): tier 0 ZERO-SHOT (Exp E ckpt `E_ms_kp_pot_v_g128.pt`, unchanged) -> tier 1 FINE-TUNE (warm start from E, short, scene-mix data + replay of E data) -> tier 2 FULL-TRAINED (from scratch on the scene, only if tier 1 insufficient). Seed 4738 for all training/data; eval seeds 9100/9200/9300 (held out; 48 scenes/seed for bounce/orbit, 4-8 scenes/seed for globular/BH). Truth = fp64 exact sim (same integrator as gravity_sim.rollout_torch; bounce = bounce.py penalty-force physics, 8 substeps).

Common units: sim units of gravity_sim (G=1, eps .5, dt .1; bounce dt .15, box 15, r .75, k 400, gravity 9 toward +x as bounce.py). err@k = mean per-body position error (sim units / cells for bounce). Chaos floor from `chaos-floor-investigation.md` (branch worktree-agent-a3afdebef86797bbc): BH err@100 is saturated (floor 1e-5 relative perturbation already .54), err@50 floor .006-.015; Exp B err@100 floor .013-.018. Every scene gets its own perturbed-truth floor (positions +1e-5 L, same 4 seeds) computed in-script so each pass criterion is stated against a floor.

## Scenes, pass criteria, predictions (written before any run)

### 1. BOUNCE (2-20 unit balls, 15x15 box, walls + penalty contact + uniform gravity 9, 100 ticks, dt .15)
Observables: err@5/10/20/50/100 vs truth, vs ballistic baselines (const-vel; gravity-only free flight without walls/contact); fraction of balls outside the box at tick 100; wall restitution e = |v_n out|/|v_n in| from a single-ball wall probe (speed 2-30 along y, gravity acts along x only; truth e = 1.0 +-.05 for the conservative penalty wall); total energy drift dE/E (KE - g x + wall/contact potential not tracked: use KE - 9 x, valid away from contact) and mean-x (height along gravity) and mean KE vs time; min pair distance / fraction of overlaps deeper than .3 r (contact working).
Pass (any tier): err@20 <= 0.5 x ballistic-gravity-only err@20 AND outside-box fraction < 1% AND e within .85-1.15 for speeds up to 15 AND |KE(t)/KE_truth(t) - 1| < 25% to tick 100. "Strong" pass: err@10 within 3x of the conservative_contact model on the same scene type (its published err@5/10/20 .12/.31/2.39 is a different box/density, so only order of magnitude; the same-scene run of cons_pure_100 is attempted if cheap).
Predictions: ZERO-SHOT fails completely: no gravity-9 term (the Exp E net only sees mass density and was never given an external field), no walls (tokens outside grid get zero force, inside the box nothing differs from empty space), contact replaced by attractive softened gravity. Expect err@k ~ const-vel err (slightly worse; balls fly out through the walls, outside-box fraction ~100% by tick 50, e = n/a). err@5 ~ 0.5*9*(.75)^2 ~ 2.5 cells (free-fall), >10 by tick 20. FINE-TUNE with the minimal addition below: err@10 1-3x of the cons-contact order of magnitude; restitution e ~ .8-1.0 below speed 10 but unreliable for the fastest hits (dt .15 gives 2.4 cells/tick at 16 units/time; contact lasts ~1-2 ticks, so a single-Verlet-per-tick model needs `sub` substeps). Predict: fine-tune with sub=1 FAILS the restitution/outside criterion (stiff contact unresolved), fine-tune with sub=4 passes it. FULL-TRAIN (same architecture, scratch) ~ fine-tune (the net body is irrelevant, the new terms do the work).
Minimal architectural addition (opt-in subclass `BounceScatterField`, all new params zero-init, zero-shot == Exp E when untrained): (a) learnable uniform acceleration vector `gvec`; (b) wall term: for each of the 4 walls, an inward acceleration MLP(d, log(d+.05)) windowed to range 2 of the distance d from the token to that wall (box bounds are an input, so box size generalizes); (c) separate contact MLP (own pair kNN term, dense, window 2) so the pair-gravity pp term of the base model can be switched off by (d) a learnable scalar `gscale` on the base (mesh+kernel+pp) output; (e) optional `sub` substeps of velocity-Verlet for the local terms (wall, contact, gvec) within a tick while the mesh field is evaluated once per tick. Without (b) the grid net cannot see walls at all (no wall channel, no padding signal); this is the structural reason zero-shot cannot work. Not implemented: wall as a mesh source term (would let the grid carry it but loses sub-cell resolution at h=.5 vs penetration ~.2).

### 2. BLACK HOLE (existing runner scatter_bh.py logic, 300 bodies c=10 sigma 8 vfac .3, 100 ticks)
Observables: err@5/10/20/50, core radius vs truth, vmax/c < 1, fraction outside grid, relativistic energy drift. Also zero-shot size/parameter transfer: N=100, 1000; sigma 5 and 12; c=5, 20.
Pass: err@50 <= 1.0 and err@20 <= .08 at N=300 (baseline .91/.067, 4 scenes, chaos floor .006-.015 at 50), vmax/c < 1, core radius within 25% of truth at ticks 20/50. Exp E was TRAINED on the cold-cluster mix (collapse mix + c=10 relativistic is the training scene), so tier 0 here == the trained model; there is no fine-tune/full tier needed (reported as "in-distribution (trained)"). Predictions: N=300 reproduces baseline (~.008/.021/.062/.91 +-30% across seeds); N=100 better (fewer encounters), N=1000 worse (beyond 80-300 training range, density 3x): err@20 .15-.5; sigma 12 (looser) easier; c=5 (stronger relativistic cap) err@50 2-3x worse; c=20 close to nonrelativistic collapse, similar.

### 3. ORBIT (1000 ticks = 100 time units)
Two families: (a) equal-mass binary, a (semi-major) 3-12, e 0-.7; (b) planetary: star M 10-30 + 3-6 planets mass .01-.5 (mutual gravity included), a 3-25, e 0-.4. Zero-shot tier uses Exp E as is; Exp C-style masses are OOD for the net (trained unit masses).
Observables: err@5/10/20/50/100/300/1000, per-body mean motion (unwrapped angle rate about the star / COM) vs truth -> period error %, per-planet specific orbital energy E_orb(t) and eccentricity drift, total energy drift dE/E, angular momentum drift dL/L, planets lost (r > 3 a0 or outside grid) fraction. Floors: perturbed-truth err@k at 1e-5 L.
Pass: period error < 5% (median over bodies), planets lost < 5%, |dE/E| < 5% and |dL/L| < 5% at tick 1000, err@100 < 0.5 a0 (i.e. orbit phase kept within about a half radian for the median planet). Reference: CentralForce baseline achieved dE/E .010 and dL/L .001 at tick 100 on 2 bodies; scatter ms_rec .29/.20; Exp E value known only for 100 ticks (.131 err@100, two_body regress).
Predictions: binary zero-shot: ok at 100 ticks (err@100 ~ .1-.3, as two_body regress .131), period error 3-10% over 1000 ticks, dE/E 5-30% (non-conservative learned field, curl/total .2-.5), dL/L similar; fails the 5% pass bar on drift but keeps bound orbit for >300 ticks. Planetary zero-shot: worse: star mass 10-30 is 10-30x the input range the mesh net saw, light-planet self-force (probe .05-2.6 vs planet accel ~.5-1) dominates, momfix pushes star self-force onto planets as a uniform acceleration (Exp C finding 3): expect planets to drift/escape within 100-300 ticks (loss 30-70%), period errors > 10%. Fine-tune (scene-mix of both families + E replay, ~15 min) should reach err@100 < .3 and dE/E < 10% for binaries, planets lost < 20%; may not reach the 5% bar by tick 1000 (non-conservative learned field; only an architecture change toward a potential-gradient-only force with exact antisymmetry would). Full-train: ~ fine-tune.

### 4. GLOBULAR (projected-Plummer clusters, virial q=.5, unit-mass bodies, N=300 primary, N=100 and 1000 zero-shot extras, 1000 ticks)
Not per-body positions beyond the chaos horizon (a few crossing times). Observables, model vs the exact sim AND the spread across K=8 truth replicas (same N, new random ICs, plus 1e-5-perturbed twin of the same IC): Lagrangian radii r10/r50/r90 vs time (about the median centre), KE and PE and E=KE+PE vs time, virial ratio 2KE/|PE|, velocity dispersion, escaper count (E_i>0 and r>3 r50), fraction outside the grid, per-body err only at k=5/10/20.
Pass: r50(t) and E(t) of the model inside the truth-replica band (mean +- 2 std, floor at 5% of value) for >= 90% of ticks to tick 1000; total energy drift |dE/E| < 10% at tick 1000; escaper fraction within +-5 percentage points of truth; err@20 < .1.
Predictions: Exp E was trained on 10-200 uniform-box bodies and cold Gaussian collapse (80-300 bodies, sigma 5-10), not equilibrium bound clusters; the long-range kernel is right and the pair term's 16-nearest neighbours cap close encounters. Zero-shot N=300: err@5/10/20 ~ .005/.015/.05 (like BH early), r50 tracks truth for ~100-300 ticks then drifts (non-conserving learned field heats or cools it: predict slow expansion, dE/E 10-40% by 1000), fails the 90%-of-ticks band criterion at 1000 but passes over the first ~200 ticks. N=1000 zero-shot worse (outside training N), and the escaper count too high. Fine-tune on Plummer mix (N 100-300) ~15-25 min: predict r50 band pass for the full 1000 ticks at N=300 and N=100, N=1000 marginal; dE/E halved. Full-train unnecessary unless fine-tune fails both energy and r50 criteria.

## Compute plan / time estimates (before submission)
- Zero-shot eval job (all four scenes, truth + floors + replicas): truth all-pairs fp64 on H200; globular N=1000 x 1000 ticks x 4 substeps ~ 4000 force evals of 1e6 pairs ~ 0.3 s each at fp64 worst case -> ~5 min per replica worst case; use N=300 replicas (K=8, ~5 s each) and one N=1000 replica. Model rollouts ~1 ms/tick -> negligible. Estimate whole job < 20 min.
- Fine-tune jobs: batch 16, k 4->20, graph/cl opts, ~0.2-0.3 s/it measured by agent 1 (218 ms/it at B32/N100) -> 900 s budget ~ 4000 iterations each (scene data gen 3-6 min on GPU). Each job < 40 min including data and eval; checkpoints (model+optimizer) written every 100 it.
- Full-train (only if needed): 3600 s budget, scratch, same data size, < 80 min.

## Results
(filled in below as jobs finish)

### Jobs
Zero-shot eval 3319 (all scenes). Fine-tune 3325 (bounce ft4/ft1, orbit ft [DIVERGED: one finite-but-huge step at it ~5780 wrecked the weights, 3968 skipped steps after; discarded], globular ft). 3328: orbit ft2 (lr 2e-4, `--loss-cap`), bounce full4 (scratch), orbit full (scratch), globular full (scratch; trained but diverged at eval, see below). Work copy on Polaris `~/bounce-scenes`; checkpoints in `~/bounce-scenes/checkpoints_scenes/`. `scripts/scatter_scenes_summary.py` prints the table below. Added `loss_cap` to `train_scatter_field.train` (opt-in, default off) and made the final save conditional on finite weights.

### Results (mean of held-out seeds 9100/9200/9300; err in sim units, bounce in cells)

BOUNCE (48 scenes/seed, 2-20 balls, 100 ticks). Truth-noise floor (1e-5 perturbation) .0035/.12/2.5/6.3 at k=5/10/20/100: the ceiling at k=100 is chaos (box-size saturation), err@20 within 2x of floor is near the best achievable.
| tier | err@5 | @10 | @20 | @100 | outside box @100 | restitution v=10 / 15 |
|---|---|---|---|---|---|---|
| zero-shot | 3.16 | 6.63 | 7.64 | 23.1 | 91% | no wall contact (balls leave) |
| ft, sub=1 | 1.10 | 2.67 | 5.41 | 5.71 | 0% | 1.05 / 1.03 |
| ft, sub=4 | 0.65 | 2.04 | 5.60 | 6.43 | 1% | 1.05 / 1.02 |
| full-train, sub=4 (scratch, 20 min) | 0.33 | 1.66 | 5.28 | 6.41 | 0% | 1.01 / 1.02 |
| ballistic (gravity only) | 1.95 | 6.60 | 38.9 | 1009 | | |
Prediction check: zero-shot failure CONFIRMED (err ~ ballistic early, 91% leave the box; restitution undefined). WRONG: sub=1 fine-tune passes (I predicted it would fail restitution); the 9190 vs 5156 iterations of sub=1 vs sub=4 and lack of a clear sub=4 win at err@20 mean contact is resolved well enough at dt .15 by a learned 1-step Verlet; sub=4 only helps err@5/10. Pass criteria: err@20 <= .5 x ballistic (5.4 vs 38.9) PASS; outside < 1% PASS (ft4 .96%, ft1/full 0%); restitution .85-1.15 to v=15 PASS; KE tracking criterion not tabulated here (ke curves in json, not checked: untested). err@5/10 are not close to the cons-contact model's .12/.31 order (different box/density; not a same-scene comparison, attempted comparison NOT run).

ORBIT (24 scenes/seed per family, 1000 ticks; truth dE/E ~6e-6, floor err@1000 1e-3 binary / .56 planetary).
| family | tier | err@20 | @100 | @1000 | period err (median) | planets lost | dE/E @1000 | dL/L @1000 |
|---|---|---|---|---|---|---|---|---|
| binary | zero-shot | .0081 | .094 | 3.22 | 10% | 0 | .18 | .17 |
| binary | ft (E replay + mix) | .0056 | .081 | 3.60 | 28% | 0 | .40 | .19 |
| binary | full-train scratch | .0026 | .038 | 2.39 | 7.9% | 0 | .13 | .072 |
| planetary | zero-shot | 1.16 | 33.8 | 3124 | 93% | 100% | 1300 | 2600 |
| planetary | ft | .049 | 1.27 | 18.5 | 37% | 26% | .53 | 5.1 |
| planetary | full-train scratch | .063 | 1.97 | 27 | 60% | 31% | .67 | 4.9 |
Prediction check: binary zero-shot "5-30% drift, period 3-10%" roughly right (period 10%, dE/E 18%). Planetary zero-shot worse than predicted (all lost, not 30-70%): the star-mass OOD failure. No tier meets the 5% period/drift pass bar; binary is closest (scratch 7.9%/13%). Fine-tune fixes escape (100% -> 26% lost) but not energy/L drift (dL/L 5: planets exchange angular momentum non-physically). The first ft attempt (lr 5e-4, no loss cap) blew up from one close-encounter step; ft2 with lr 2e-4 + loss cap OK. Binary fine-tune was WORSE than zero-shot at long horizon (period 28%) because half the data is planetary+replay and it trades off; scratch binary is better, an effect of 1800 s vs 1200 s and of 14.5k vs 9.7k iterations is not separated (untested).

GLOBULAR (Plummer, unit masses, 1000 ticks; 6 N=300 runs, 3 N=100, 3 N=1000; band = fraction of ticks where |model - truth| <= max(2x |1e-4-perturbed twin - truth|, 15% of truth) for r50 / E).
| tier | N | err@20 | r50 band | E band | dE/E @1000 | escapers model/truth |
|---|---|---|---|---|---|---|
| zero-shot | 100 | .065 | .82 | 1.00 | .033 | .10/.04 |
| zero-shot | 300 | .231 | .85 | .97 | .131 | .12/.04 |
| zero-shot | 1000 | .868 | .97 | .91 | .179 | .12/.03 |
| ft | 100 | .074 | .83 | 1.00 | .103 | .08/.04 |
| ft | 300 | .210 | .91 | .96 | .112 | .10/.04 |
| ft | 1000 | .897 | .82 | .64 | .220 | .15/.03 |
| full-train scratch | any | err@5 ~6, NaN by tick 50-100 | - | - | - | - |
Truth dE/E < 0.4%. The 90%-of-ticks band criterion is marginal at N=300 (zero-shot .85, ft .91); dE/E is 11-13% (fails the 10% bar, truth <.1%), escapers 3x the truth's. err@20 .21-.23 vs predicted <.1 (FAIL; the 1e-4-twin floor err@20 is much smaller, so this is model error). Fine-tune changes little: r50 band +6 points at N=300, dE/E same, worse at N=1000. The model keeps r50 right and loses energy: r90 grows to ~180 vs truth ~50 by tick 1000 in the visual review (halo evaporates), E drifts down. Full-train from scratch FAILED: 12.7k of 14.7k training steps skipped as non-finite/over loss-cap, weights never became usable (a long-N 300 scene start from scratch with all-pairs-free mesh + zero-init heads is unstable; Exp E itself needed a warm-start chain from B). Not retried (would need the same B->E curriculum, hours).

BLACK HOLE (in-distribution for Exp E; tier 0 = trained model). err@5/10/20/50/100, floor@50 in brackets:
| variant | err@20 | err@50 | err@100 | vmax/c max |
|---|---|---|---|---|
| N=300, c=10 (headline; 4 scenes) | .062 | .914 [.007] | 4.79 | .94 |
| N=100 | .031 | .247 [.0015] | 2.97 | .75 |
| N=1000 (above train N) | .191 | 3.16 [.032] | 9.24 | .99 |
| sigma 5 | .165 | 2.00 [.015] | 4.76 | .96 |
| sigma 12 | .042 | .319 [.003] | 3.86 | .91 |
| c=5 | .054 | .564 [.003] | 3.73 | .997 |
| c=20 | .066 | 1.09 [.009] | 5.22 | .71 |
Reproduces baseline (.0080/.021/.062/.91). Predictions: N=100 better CONFIRMED, N=1000 worse CONFIRMED (err@20 .19, in my predicted .15-.5), sigma12 easier CONFIRMED, c=5 err@50 predicted 2-3x worse: WRONG (better, .56); c=20 similar CONFIRMED. All variants are far above the chaos floor at k=50 (60-200x). |v|<c holds everywhere (max .997 c at c=5).

### Minimal architecture additions identified (bounce)
Needed: (1) uniform acceleration, (2) wall term as a function of distance to each box wall (windowed 2 units), (3) a separate contact kernel (the base pair term is attractive softened gravity; here repulsive, short range), optional (4) substeps for local terms. Implemented opt-in as `BounceScatterField` (scripts/scatter_bounce.py), zero-init, untrained == Exp E. Param count +8.6k (two 64x64 MLPs + 3 scalars), all-pairs contact (N<=20 here; the cell-list kNN path from agent 1 would be needed for large N, not done). The mesh net does the rest: it was never needed for the bounce dynamics here; ablation (turning the base off via gscale) NOT run. Not tested: balls > 20, other box sizes (box bounds are an input but trained only at 15), non-unit mass, the cons-contact model on the same scenes.

Unbiased visual reviews (docs/debugging/scatter-scenes-review-prompt.md, fresh agents): bounce zero-shot: red bodies leave the box, no wall bounce, divergence sudden by tick 10. Bounce ft: bodies stay in box, configurations unrelated to truth after ~tick 10 (chaos). Binary orbit (zero-shot and ft): follows ~100-150 ticks then gradual phase/amplitude drift, bound loops kept. Planetary zero-shot: all planets drift away on straight lines from ~tick 100-150 (distances thousands). Planetary ft: inner planets roughly right, outer planets become eccentric with close passes, one truth escape missed. Globular: core matches, halo differs from tick 20; r90 grows to ~180 vs ~50 and E drifts down after tick ~500. BH: tracks to tick 50-60, halo differs at 100.
