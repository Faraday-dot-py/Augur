# Scatter-field OOD stress (plan agent 7), 2026-10-04

Zero-shot, one axis at a time, Exp E checkpoint (`~/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`), inference flags `kcache,fastio,cl,graph` (scatter-opt). Tool: `scripts/scatter_ood.py`, Polaris wrapper `scripts/polaris_scatter_ood.sh`, results `results/scatter_ood.json`. Research only, no model code changed.

## Training distribution (what "in" means), from code
Exp E: unit masses only; N 10-200 uniform box (spread 5 sqrt(N/8), v ~ N(0, .5 (N/8)^.25)) plus 50% cold Gaussian clusters (N 80-300, sigma 5-10, per-component speed vfac sqrt(N/(4 sigma)), vfac .1-.6), Newtonian, eps .5, dt .1, extent 64 (grid 128, h .5), open boundary. Fine-tuned on 50-100-step windows. BH eval (c=10, N=300, vfac .3) is relativistic but never trained on it (model is fed v(p)).
Architecture facts relevant to OOD: mass enters only as the scattered density m and m*v (U-Net input, not normalised), the learned isotropic kernel K(r) (acceleration linear in source mass), and the pair term (linear in m_j); `momfix` removes mass-weighted net force. No eps, dt, c or charge input. Tokens outside +-32 get zero mesh force (pair term still acts). No wall/periodic support.

## Method
Per point 4 scenes (seeds 9100/9200/9300/4738), 10 time units. Truth: float64 KDK, h=.025 (4 substeps at dt .1), relativistic momentum state when c given, masses supported (checked against `gs.rollout_torch`). err@t = mean per-body position error at t = .5/1/2/5 (ticks 5/10/20/50 at dt .1; err@10 saved but unreliable: chaos ceiling). Floor = truth with 1e-5 x rms-radius position perturbation. Ballistic = constant-velocity coasting. dE/E_scale = max_t |E-E0| / (|T0|+|U0|), also truth's own. dP, dL normalised by sum m|p|, sum m|r||p|. "t>1" = first time err exceeds 1.0 (2 eps). Reference in-distribution scenes: Gaussian N=100 sigma 8 vfac .3 (Newtonian) and BH N=300 c=10.

## Predictions (written before any run)
Ranked expected severity, most to least:
1. softening eps in truth (model has eps=.5 baked into K and pair net): err grows with |log(eps/.5)|, but only for scenes with close encounters; cluster collapse breaks at eps .05-.1 (err@2 > .3), eps 1-2 mild. Architectural (no eps input) / data coverage.
2. mass ratio / dominant mass: 1e3:1 breaks. Reason: U-Net input scale (m up to 1e3 per cell vs ~1 in training) and heavy-body near field (pair net saw only m_j=1). Predict equal-mass scaling m=.1-.3 and 3 mildly worse (input scale 10x), m>=30 and heavy >=100 break (err@2 >> floor, ballistic-level by t~2-5). Heavy body error itself will be small (it barely moves); light-body errors dominate. Mostly data coverage (force is linear in mass by construction), fixable by fine-tune with mixed masses; U-Net input normalisation would be the architectural fix.
3. close passes / tight binaries d <= .25 (inside softening core, orbital period ~ dt scale) : orbital phase error grows linearly, err@5 ~ 1 at d .125-.25, fine for d >= 1. Predict tight binary d<=.25 breaks, hierarchical triple with inner a=.2 breaks. Data coverage + integrator (dt .1 vs period).
4. dt: model learned dt .1 Verlet compensation. dt .05 and .2 roughly 2-5x worse err@2 than reference; dt .4 breaks (err@5 ~ ballistic), dt .025 also degrades (recurrent field input out of regime). Architectural-ish (dt baked in via integrator compensation and the recurrent field).
5. boundary/offset: cluster offset by >= 24 puts mass off-grid: mesh force zero for off-grid tokens, err explodes once > ~30% of mass is outside; shift 8-16 fine. Hot clusters (vfac >= 2.5) expand out of the grid by t~3-5 and lose the mesh force: err grows once escapers cross +-32. Architectural (fixed domain, open boundary = zero force, not a monopole tail).
6. velocity scale: zero-velocity cold collapse (vfac 0) is a harder collapse than any training (trained vfac >= .1): err@5 several x reference; hot (vfac 4-8) is easy dynamics (nearly ballistic): small err because ballistic coasting is accurate until escapers leave the grid. Relativistic c: c >= 10 as reference; c=3-5 with N=300 (R_s > cluster) hard because truth is far from Newtonian; c>=30 behaves as Newtonian = fine.
7. density: sigma 1-2 (denser than training min 5, N=100): breaks (collapse timescale ~ 1 time unit, pair near-field and grid h=.5 resolution); sigma 16-32 sparse: easy (nearly ballistic). Plummer a=.25-.5 (core below grid resolution) degraded; a>=2 near equilibrium, long-lived: fine, probably better than cluster collapse.
8. IC shapes: shell/disk/two-cluster moderate: err within 2-3x of reference (they are still gravity with N 100, unit mass); shell collapse at rest (focusing) is the worst (violent crossing), filament next.
9. float precision: fp32 vs fp64 model: identical to within the floor at t<=2 (no sensitivity); tf32/bf16: bf16 worse (~2x err@2), half16 ~ +1.5% as in scatter-optimization.md. Not a failure axis.
Non-finite: predicted none anywhere except possibly mass 1e4:1 and eps .05 (large forces); the model output is a bounded U-Net + MLP so blow-up risk is low.

## Results
Jobs 3317 (main) and 3324 (mass dt-matched); `results/scatter_ood.json`, `results/scatter_ood_dtmatch.json`, logs `results/scatter_ood_3317.log`, `_dtmatch_3324.log`. Harness check: truth == `gs.rollout_torch` (max diff 0.0, Newtonian and relativistic). In-distribution refs (4 seeds): cluster N100 err@.5/1/2/5 (time units = ticks 5/10/20/50) .0037/.0103/.031/.261; BH300 c10 .0080/.0209/.062/.91 (matches baseline). No non-finite tick at ANY of ~140 points: failure is always silent wrong dynamics, never NaN. Ratios are err@2 vs the reference .031.

| axis | point | err@.5/1/2/5 or err@2 | verdict |
|---|---|---|---|
| mass ratio, binary d=4 | q=1/3/10/100/1e3/1e4 | .0005/.0033/.011/.040; .0015/.0075/.036/.25; .023/.097/.35/2.2; .42/1.6/7.0/58; 10/40/216/1724; 130/505/2537/18471 | breaks q>=10 (err>1 at t=3.7), q=100 at t=.85; q>=100 worse than coasting; dE/E 1 at q=10, 3e2 at q=100 |
| dominant mass in 300-body cluster | heavy 1/3/10/30/100/300 | err@2 .030/.033/.063/1.47/10.6/42.6 | ok to 3, 2x at 10, breaks 30 (47x), >=100 coasting-level |
| mass spectrum log-uniform | .5-2/.1-10/.01-100/.001-1000 | err@2 .043/.21/20.6/3987 | breaks beyond ~1 decade |
| equal mass scale m | .01/.1/.3/1/3/10/30/100 | err@2 .0017/.0095/.016/.031/.119/1.9/11.3/737 | smooth: ok .1-1, 4x at 3, breaks >=10 |
| velocity vfac | 0/.1/.3/.6/1/1.5/2.5/4/8 | err@2 .030/.031/.031/.027/.026/.031/.056/.14/.54 | zero-velocity cold collapse NOT a failure; hot grows (escapers leave grid; vfac 8 err@5 2.1 vs coasting 1.8) |
| relativistic c (BH300) | 1.5/3/5/10/30/100/1e3 | err@5 .086/.21/.56/.91/1.17/1.18/1.19 | not a failure axis; Newtonian limit = reference |
| density sigma (N=100) | 1/2/3/5/8/16/32 | err@2 1.15/.40/.16/.051/.031/.031/.088 | breaks sigma<=2 (trained min 5): 37x at 1; sigma 32 3x (mass clipped at +-29) |
| Plummer core a | .25/.5/1/2/4/8 | err@2 3.3/.60/.26/.11/.047/.027 | core below grid cell h=.5 breaks (~1/a) |
| impact parameter (2-body pass, head-on fall) | b 0..8; fall d 1/4/12 | err@5 .041-.065 all | NOT a failure axis, no b dependence |
| tight binary d | .125/.25/.5/1/2/4/10 | err@5 .115/.174/.107/.042/.056/.040/.060 | mild, 3-4x at d<=.25 |
| hierarchical triple (inner a) | .2/.5/1/2; a=.5 m3=10 | err@5 .18/.074/.080/.064; 1.97 | hierarchy fine; the m3=10 case breaks (mass) |
| softening eps in truth | cluster .05/.1/.25/.5/1/2 | err@2 1.42/.45/.29/.031/.39/.73 | sharp optimum at trained eps .5; binary d=1 eps .1/.25/.5/1 err@5 1.14/.86/.04/.82 |
| dt (model dt = truth tick) | cluster .0125/.025/.05/.1/.2/.4 | err@5 .32/.30/.28/.26/.43/1.32 | tolerant 0.25x-2x (<=1.7x); 4x breaks (5x); BH dt .05/.2 err@5 1.01/1.94 vs .91; dE/E rises off-dt (.15 at .0125 vs .02) |
| IC shape | uniform L10/25, shell R8/R16 rest, shell rot, disk R12/24, two-cluster v0/1/4, filament | err@2 .047/.019/.11/.061/.067/.036/.016/.046/.043/.046/.27 | all 1-4x except filament 8x |
| boundary offset (open) | shift 0/8/16/24/32/40 | err@2 .031/.031/.069/.43/1.28/1.45 | breaks when cluster leaves +-32 grid (14-45x); hot vfac 1.5 shift 20: .21 |
| precision | fp32/fp64/tf32/half16/bf16 | cluster err@5 .2614/.2618/.2615/.2612/.2663; BH .9129/.9135/.9122/.9129/.9221 | not an axis; bf16 +2% |

dt-matched mass (job 3324; dt and horizon scaled by M^-1/2, same tick count, so errors are at scaled times): equal-mass m=10 err@5 10.2 -> 1.21, m=100 2415 -> 4.3; heavy 100 33 -> 11.5, heavy 1000 9144 -> 39.5; binary q=100 58 -> .64, q=1000 1724 -> 2.3. So much of the mass-axis failure is dt vs dynamical-time mismatch (mass is a time scale); a remainder (heavy>=30, q>=1000: 10-100x reference) is real mass-range coverage.

## Review of failing rollouts (unbiased subagents; grids `videos/scatter_ood_fail_grid_a.png` and `_b.png`, mp4 `videos/scatter_ood_fail_heavy30_q100.mp4`)
All failures start immediately (error jumps orders of magnitude in the first tenths of a time unit) then grow gradually; no late sudden jump. heavy30 and shift32 sit on the coasting curve (model effectively gives no force); q=100 error is above coasting (a body flung to ~800 units: spurious acceleration; path plot hides it, error panel shows it); eps .1 ends near the 1e-5-perturbed-truth level (chaos-limited); dt .4 shows one step at t~.5 then smooth growth; Plummer a=.25 error ~.2 after step one (core unresolved); equal m=10 and sigma 1 are chaotic scenes where error is 2-3x the perturbed-truth curve.

## Ranking by severity (zero-shot)
1. Mass range (ratio >=10:1, equal-mass scale >=3x, dominant >=30x): 5 to 1e5x worse, dE/E to 1e5.
2. Open boundary (mass beyond +-32): 14-45x, coasting-level.
3. Softening mismatch (eps 2x or more off): 10-45x, sharp at trained eps.
4. Core density below ~2 units: sigma 1 37x, Plummer a .25 107x.
5. dt 4x off: 5x; 0.25-2x tolerated.
6. Mild: filament 8x, tight binary 3-4x, shells 2-4x.
7. Not failures: velocity (incl. zero), c, impact parameter, precision.

## Architectural vs data coverage (inferences from code + results; fine-tune/retrain effects UNTESTED)
- Mass: data coverage plus input scaling. K and pair term are linear in source mass; the U-Net input is un-normalised density; dt-matching recovers 10-1000x. Fine-tune with log-uniform masses should extend range (guess ~1e2:1); normalising density/passing dt*sqrt(M) would be the architectural fix for 1e3+.
- Boundary: architectural (fixed grid, zero force outside, no monopole tail, no wall mode). Fine-tuning cannot fix; needs far-field monopole outside grid or recentring/rescaling (overlaps scaling agent).
- eps: architectural-ish (baked into K and pair net, no input). Add eps conditioning + mixed-eps training; fine-tune likely enough for 2x, input needed for 10x.
- Core density: grid h=.5 and pair radius 2 give a hard limit below ~h; data coverage between h and 5 (fine-tune on dense clusters).
- dt: tolerant (Verlet), no fix needed. Velocity, c, impact parameter, precision: robust.

## Caveats
4 scenes/point, 10 time units; one truth discretization (substeps = round(dt/.025), so other dt use different substeps than training); err@10 saved but at chaos ceiling. dE/E for relativistic c<=3 includes the truth's own non-conservation (compare model vs true column in JSON). Open boundary only; wall/contact OOD belongs to the scenes agent; N and world-size sweeps to agents 2/3.
