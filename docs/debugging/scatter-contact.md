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
