# Scatter-field obstacle + ball-ball contact (decision #7 fix 1), 2026-10-06

Branch `scatter-contact`. Spec: docs/debugging/scatter-review-decisions-2026-10-06.md #7, options B (main) and A (ablation) in obstacle-contact-force-options.md. C/D held, E not built. Code: `scripts/scatter_contact.py` (geometry, truth, model), `scripts/scatter_contact_scenes.py` (scenes, train/eval), `scripts/scatter_contact_test.py` (unit tests), `scripts/render_scatter_contact.py`, `scripts/polaris_scatter_contact.sh`. Results `results/contact/`.

## What was built (all opt-in; `ContactScatterField` subclasses `ScatterField`, plain checkpoints still load, zero-init heads, gscale=1 => untrained == Exp E, unit-tested bitwise)
- Static SOLID obstacles as primitives (hbox/hdisc = free space inside a rectangle/disc, circle, polygon), rasterised once per scene to `[sdf, d sdf/dx, d sdf/dy, material]` on the 128^2 grid (h=.5). Material channel is a reserved slot (input to the B MLP, zero everywhere now). Moving obstacles out of scope.
- Option B (`obstacle="sdf"`): per ball gather (bilinear, border clamp) SDF d, normal n; 5-point stencil (p, p+-s n, p+-s t, s=1) of SDF differences; features gap = d - R_i (per-body radius), log(gap+.05), v.n, v.t, 4 stencil diffs, material -> MLP(9->64->64->2) -> force in the (n,t) frame, windowed to gap<2, scaled x10. Added AFTER momfix. Force is learned; geometry is input; no hard-coded reflection/projection.
- Option A (`obstacle="kernel"`): phi = K * occupancy (FFT path, K = MLP(r, log r) windowed to r<4), force = -grad phi gathered at the ball, added after momfix. Conservative, no velocity, radius-blind (inherent to the option).
- Ball-ball contact (both models): per-body radius state, overlap gap = r - (Ri+Rj) and log(gap+.05) -> pair MLP, k=8 nearest neighbours (dense top-k for N<=512, uniform cell list above, unit-tested equal to dense). Learned gravity vector `gvec` and `gscale` on the base output as in AugurScatterField.
- Truth: fp64 penalty physics (bounce.py convention, k=400, gravity 9 toward +x, dt .15, 8 substeps) with SDF walls and per-body radii; reproduces `bounce_truth` to <1e-8 on the box scene (unit test).

## Protocol
Train mix (800 scenes, 200 each, 80 ticks, seed 4738): box15 (the old 14x14 box, R .75), box_var (random rectangle size/offset, R in U(.5,1)), post_circle (box + 1-2 solid discs), post_rect (random box + rotated solid rectangle). Held-out shape classes never seen in training: disc (circular container, offset) and post_poly (box + solid triangle/pentagon). Eval seeds 9100/9200/9300, 24 scenes each, 100 ticks, chaos floor by 1e-5 perturbed truth, plus ballistic baseline, centre-beyond-surface ("tunnel") fraction, KE ratio vs truth, deep pair overlap, and a restitution probe against a flat wall (to v=15) and a circular post (to v=10).
Models, each tested separately on identical eval sets: E zero-shot (frozen baseline), augur1 = `AugurScatterField` ft sub=1 (`bounce_ft1.pt`, box15 only, hard-wired box), B, A. B and A warm start from E, 8000 iterations, batch 16, k 4->30, lr 5e-4, identical data.

## Predictions (written before any training/eval run)
- Unit tests pass (parity, SDF, gather, zero-init identity, cell-list == dense, gradients).
- E zero-shot: err ~ ballistic gravity, tunnel ~90% by tick 100 on every family (no walls).
- augur1: on box15 near its published (err@20 ~5.4, tunnel ~0); on box_var/post_*/disc/post_poly it cannot see the obstacle: tunnel through posts/disc wall > 30% by tick 100, err@20 several times worse than on box15.
- B: box15 err@5/10/20 ~ .7-1.1 / 2-2.7 / 5-6 (same order as ft1; floor at k=20 is ~2.5); in-distribution non-box families within ~1.3x of box15 err@20, tunnel < 1%; held-out disc/post_poly within ~1.5-2x of box15 err@20, tunnel < 3% (polygon vertices the weak spot); restitution .85-1.15 to v=10-15 on both wall and circular post.
- A: worse than B on tunnelling/restitution at high speed (conservative kernel on a blurred solid, h=.5), err@20 +20-50% vs B, clearly worse on box_var (radius-blind); held-out shapes worse than B.

## Results
(filled in below)

### Jobs
3359: unit tests (2 gradient-test failures from a test-geometry bug: pair head had no neighbour in window; fixed in test) + truth cache (387 s). 3372: ONE serial wrapper (tests, E zero-shot, augur1, train_eval B 8000 it, train_eval A 8000 it; checkpoints every 100 it via tsf.train, resumable). Pending on QOSMaxGRESPerUser at time of writing.

3372 findings: tests ALL PASS; E zero-shot and augur1 evals done (results/contact/eval_E_zs.json via download pending); first B run (lr 5e-4, no loss cap) DIVERGED at it ~4800 (finite huge loss, 2.5k skipped steps; err@20 ~2000), discarded (results/contact/eval_B_diverged.json). Fixed defaults lr 2e-4 + loss-cap 100 (same fix scenes agent used for orbit ft2). A in 3372 started after the patch (stable settings). 3380: B rerun, tag B2, same settings as A, queued behind 3372.

A (first run, lr 2e-4, cap 100) ALSO diverged at k=21 (loss 10 -> 2.7e3 -> 3e4, then every step skipped by the cap; err@20 ~30). Both B and A blow up at k~20-21, so it is the architecture/training of the added heads, not option-specific. Mitigation (bounded learned force, no conservation hard-coding): forces from all new heads pass through FMAX*tanh(x*OUT_SCALE/FMAX), FMAX=40 (penalty truth peaks ~36 at 0.3 penetration), slope at init unchanged; lr 1e-4; k_end 20. Reruns: 3380 = B2, 3382 = A2 (both serial, one wrapper each). E zero-shot/augur1 results from 3372 stand (frozen baselines).

### Results (2026-10-07; B2 job 3380, A2 job 3382, E/augur1 from 3372; eval seeds 9100/9200/9300 x 8 scenes per family, mean over seeds; err = mean position error per body, sim units; err@k at tick k)
B2 and A2 trained 8000 it (1212 s / 1285 s) without divergence at k=20-21 (bounded force heads + lr 1e-4 fixed it). Chaos floor (1e-5 perturbed truth) err@20 0.4-0.9, err@100 4.5-5.9; model err ~5 at tick 100 is at the floor (error saturates at box scale), so tunnel fraction, KE ratio and restitution are the discriminating metrics.

| family | model | err@5 | err@20 | err@100 | tunnel@100 % | KE ratio@100 |
|---|---|---|---|---|---|---|
| box15 | E zero-shot | 2.37 | 7.03 | 22.97 | 92.9 | 0.049 |
| box15 | bounce_ft1 | 0.49 | 4.96 | 5.396 | 0.2 | 0.0587 |
| box15 | B (SDF+MLP) | 1.70 | 5.11 | 5.177 | 0.0 | 0.00267 |
| box15 | A (kernel) | 1.79 | 5.07 | 3.731e+04 | 100.0 | 4.08e+06 |
| box15 | floor/ballistic/truth-tunnel | 7.2e-05 | 0.90 (ball. 38.7) | 5.63 | 0.6 | |
| box_var | E zero-shot | 2.31 | 6.53 | 23.47 | 96.1 | 0.0701 |
| box_var | bounce_ft1 | 3.43 | 6.00 | 6.568 | 49.5 | 0.0236 |
| box_var | B (SDF+MLP) | 1.62 | 4.53 | 4.781 | 0.0 | 0.00361 |
| box_var | A (kernel) | 1.79 | 4.50 | 7.376e+04 | 100.0 | 4.16e+07 |
| box_var | floor/ballistic/truth-tunnel | 8.5e-05 | 0.68 (ball. 39.2) | 5.31 | 1.5 | |
| post_circle | E zero-shot | 2.35 | 7.37 | 24.85 | 93.8 | 0.059 |
| post_circle | bounce_ft1 | 0.56 | 4.54 | 5.511 | 1.4 | 0.0329 |
| post_circle | B (SDF+MLP) | 1.65 | 4.92 | 5.516 | 0.0 | 0.00385 |
| post_circle | A (kernel) | 1.81 | 5.11 | 3.531e+04 | 100.0 | 8.89e+05 |
| post_circle | floor/ballistic/truth-tunnel | 7.2e-05 | 0.44 (ball. 38.0) | 5.88 | 0.6 | |
| post_rect | E zero-shot | 2.31 | 6.83 | 24.07 | 96.3 | 0.0597 |
| post_rect | bounce_ft1 | 1.91 | 5.17 | 5.404 | 35.8 | 0.016 |
| post_rect | B (SDF+MLP) | 1.65 | 4.11 | 4.75 | 0.0 | 0.0038 |
| post_rect | A (kernel) | 1.80 | 4.25 | 5.606e+04 | 100.0 | 6.32e+06 |
| post_rect | floor/ballistic/truth-tunnel | 8e-05 | 0.58 (ball. 38.3) | 5.51 | 1.1 | |
| disc | E zero-shot | 2.36 | 7.00 | 22.51 | 95.3 | 0.0613 |
| disc | bounce_ft1 | 2.11 | 5.76 | 5.585 | 53.2 | 0.0139 |
| disc | B (SDF+MLP) | 1.68 | 4.49 | 3.841 | 0.0 | 0.00859 |
| disc | A (kernel) | 1.80 | 4.44 | 3.814e+04 | 100.0 | 3.7e+06 |
| disc | floor/ballistic/truth-tunnel | 7.1e-05 | 0.49 (ball. 39.0) | 4.52 | 0.2 | |
| post_poly | E zero-shot | 2.36 | 7.34 | 24.79 | 95.9 | 0.0528 |
| post_poly | bounce_ft1 | 0.52 | 4.52 | 5.357 | 0.6 | 0.0193 |
| post_poly | B (SDF+MLP) | 1.65 | 4.83 | 5.296 | 0.0 | 0.00365 |
| post_poly | A (kernel) | 1.74 | 4.89 | 4.715e+04 | 100.0 | 2.65e+06 |
| post_poly | floor/ballistic/truth-tunnel | 7.6e-05 | 0.42 (ball. 37.9) | 5.89 | 0.0 | |

Reading against predictions:
- B: tunnel 0.0% at tick 100 on ALL six families, including held-out disc and post_poly (predicted <1% in-dist, <3% held-out: met). err@20 4.1-5.1 on all families, within 1.1x of box15 (predicted 5-6 on box15: met; early err@5 1.65 is worse than ft1's 0.49 on box15, ft1 is better where its hard-wired box applies). bounce_ft1 fails box_var (49.5% tunnel), post_rect (35.8%), disc (53.2%), as predicted; it does NOT fail post_circle/post_poly (1.4%/0.6%), contrary to the prediction (box walls hold, posts rarely hit at these densities).
- B negative: KE ratio vs truth at tick 100 is 0.003-0.009 (ft1 0.014-0.059, E 0.05-0.07): B is heavily dissipative. Restitution probe: wall model 0.49 at v=12, 0.57 at v=15 (truth 1.0); at v<=10 the probe found no rebound (None), circular post None at all speeds (to be verified, probe may be failing to detect rebound rather than the model failing). Restitution prediction (.85-1.15) NOT met for B.
- A: in-distribution horizon fine (err@20 4.3-5.1, same as B) but unstable beyond the training horizon (k_end 20): 100% "tunnel" and err 3.5e4-7.4e4, KE ratio 1e6-4e7 on every family by tick 50. Restitution probe: wall 0.99-1.01 at v 10-15 (the kernel force conserves energy on a wall when it works), circular post 2.19 at v=2 (spurious energy gain). Prediction "A worse than B" met for rollout stability, not for err@20 (equal) and not for restitution (A better on wall).
- A2 failure localisation (disc, 3 scenes): max per-tick position step 0.5-2.4 until tick ~25, then 60-76 at tick 30, then a constant ~420-565 per tick from tick 35. Constant step size looks like a clamp/saturation artefact (e.g. ball escaping the occupancy FFT domain) rather than a smoothly growing force; NOT verified in code.

Unbiased review (fresh agent, hypothesis-free, grids only; videos/contact_{disc,post_poly,post_circle}_{B2,A2}_grid.png): B2 red stays inside the container and never overlaps posts; by tick 30-100 red is a compact clump/column against the gravity-side wall while truth (blue) stays a looser, more spread cloud (consistent with the KE ratio 0.003-0.009: B settles, truth keeps bouncing). Error flat 3-7 after tick 20. A2 matches B2 to tick ~25, then red balls fly out of the frame by tick 30-35 (error jumps ~3 orders of magnitude, then grows to 1e4-1e5); by tick 60 no red is visible.

Videos (truth | model, scene 0): videos/contact_{disc,post_poly,post_circle}_{B2,A2}.mp4 in the worktree.

Conclusion: B (SDF gather + MLP) fixes tunnelling on all shapes including held-out ones with no per-shape engineering; the remaining B defect is over-damping (restitution ~0.5 at wall, KE collapse), not geometry. A as trained is unusable beyond its training horizon. Next (not done): add energy/restitution term or train B with longer unrolls and wall-bounce supervision; check A's blow-up in the occupancy FFT domain handling.
