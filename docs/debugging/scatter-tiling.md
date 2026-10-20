# Tiling copies of the scatter-field model (Exp E, `E_ms_kp_pot_v_g128`)

Branch `worktree-agent-abc314d273210207b`. Code: `scripts/scatter_tiling.py` (`TiledScatter`), `scripts/scatter_tiling_eval.py`, tests `tests/test_scatter_tiling.py` (4 pass on Polaris job 3766). Weights untouched, `scatter_field.py` untouched. Remote copy `~/augur_tiling`.

## Goal and reference

nt x nt copies, one per tile; each tile owns a square (side `tile`) and runs one copy of the model on its square plus a halo margin (`halo`, same cell size h=0.5, so tile grid G = (tile + 2 halo)/h, multiple of 32). Reference = one single grid over the whole domain (2x2 tiles of 64 -> extent 128, G=256). Emulation: all tiles run as one batch on one GPU; communication is exact array copies (assemble/slice), so it matches what neighbour exchange would give. Run time per job is minutes, so no resume/SIGTERM machinery (rule applies only to >1 h jobs).

Mechanisms (flags, tested one at a time against the frozen baseline = naive):
- naive: tile sees only its own bodies, no communication (H=0, G=128 for a 64 tile).
- ghosts (`g`): bodies of neighbours inside the tile's extended box are deposited and enter the pair term (pp=2, knn=16); only own bodies keep their force; bodies migrate by position every force evaluation.
- field_x (`fx`): the recurrent potential field halo cells are overwritten from the neighbours' interiors after each force evaluation.
- far: coarse global grid (cell 4h=2 units) carrying the learned-kernel convolution of all mass; each tile adds (coarse global potential - coarse potential of its own extended box) = kernel contribution of mass outside its box, to the kernel acceleration that feeds the U-Net and the output. Same `kmlp`, so it reproduces the single grid's (possibly wrong) far field, which is the target.
- Global momentum fix: two scalars all-reduced (sum m a, sum m). Verlet `a_prev` travels with each body.
Variants: `g16` etc. = halo 16 units; `t32_*` = 4x4 tiles of 32 with halo 16 so each copy is exactly G=128.

## Constraints satisfied (state)
O(N): tile cost depends on own+ghost bodies and halo cells only; pp unchanged (pp=2, knn=16); the far term is O(G_coarse^2 log) over the domain (cells ~ area ~ N at constant density). One trained model, no retraining; no hard-coded forces (kernel/U-Net/pp unchanged; coarse term uses the same learned kmlp); momentum fixed globally as in the single grid; interpretability unchanged. Two flags: (1) the far term needs a global coarse FFT = an all-to-all across tiles (not neighbour-local); (2) a halo larger than 0 makes a "64-wide tile" copy G>128 (192 at halo 16); strict G=128 copies need 32-wide tiles (`t32_*`).

## Pass/kill criteria (recorded before any run)
Scenes: Gaussian cluster at the domain centre (all four tiles meet at its centre), N=300 sigma 8 cold (vfac 0.3) and warm (vfac 1.0), N=1200 sigma 16 warm; scene seeds 4738 (dev), 9000, 12000 (held out), 3 repetitions each.
- Seam force error: acceleration-weighted relative error sum|a_tiled - a_single| / sum|a_single| over bodies within 8 units of any internal seam (and >=8 from the outer domain edge), one step from shared single-grid states at ticks 0/5/10/20 (field and a_prev taken from the single run). Interior = >=8 from every seam, >=8 from the outer edge.
- PASS: seam error <= 1.1 x interior error (interior error floored at 1e-3, the fp32 summation-noise level, since a perfect scheme has interior error ~0); err@5/10/20 (mean position error per body vs exact fp64 all-pairs) <= 1.1 x the single-grid run on held-out seeds; |total momentum| drift not worse than the single grid.
- KILL: seam error > 2 x interior after halo 16, or cost per tick per body rising with tile count.
- Context numbers reported: single-grid force error vs exact (~0.1 expected), tiled vs exact, bodies within 8 of the outer edge reported separately (single grid has its own edge artifact there that tiles do not reproduce).

## Predictions (before running)
1. naive: seam<2 error >> 1 (edge artifact of 1e3-7e3 vs ~120), 2-8 units ~0.3-1, interior also degraded (~0.2-0.5, other tiles' mass missing); err@20 >= 3x single.
2. ghosts g16 (no fx): seam error drops to ~0.1-0.2; interior ~0.05-0.15 (far mass beyond 16 units and U-Net context missing); err@20 within ~1.3x single.
3. + fx: small rollout gain at ticks >=10 (stale local halo field carries edge artifacts inward); no one-step gain at tick 0.
4. + far: interior error down 2-3x; seam/interior ratio 1-1.5. Pass uncertain at 1.1.
5. Halo 8 worse than 16 without far; with far the halo matters less. Halo 32 marginal over 16.
6. Cost: per-tile ms/tick and memory independent of tile count (launch-bound; halo overhead (G_ext/G_tile)^2 = 2.25 in cells at halo 16); far adds one coarse FFT (+ ~0.5 ms).

## Results (Polaris jobs 3767 force/roll/scale, 3768 big/snaps; raw `results/scatter_tiling_*.json`, summary `results/scatter_tiling_summary.txt` from `scripts/summarize_scatter_tiling.py`)

Metrics: force error = sum|a_tiled - a_single| / sum|a_single| (acceleration-weighted relative), one step from shared single-grid states at ticks 0/5/10/20, held-out seeds 9000/12000 x 3 reps. err@k = mean per-body position error vs exact fp64 all-pairs (sim units). "seam" = within 8 units of an internal tile boundary, "interior" = >=8 from every seam, both >=8 from the outer domain edge.

### Force error vs single grid, seam / interior (held out, tick-averaged)
| scene | naive | halo16 (g16) | halo16 + field exchange | halo16 + far | halo8 + far | halo32 |
|---|---|---|---|---|---|---|
| cold N300 sigma8 | 1.08 / 0.55 | .0046 / .0063 | .0046 / .0063 | .0007 / .0013 | .0029 / .0026 | .0007 / .0012 |
| warm N300 sigma8 | 1.26 / 0.43 | .0078 / .0092 | .0078 / .0092 | .0009 / .0014 | .0040 / .0028 | .0009 / .0013 |
| warm N1200 sigma16 | 0.93 / 0.41 | .104 / .075 | .104 / .075 | .0094 / .0115 | .0154 / .0126 | .0142 / .0146 |
Naive seam<2 units: 1.7 / 2.2 / 1.4 (error larger than the force). Single grid vs exact fp64 for the same bodies is 0.06-0.075 (seam) / 0.04-0.065 (interior), i.e. tiling-with-far error is 5-50x below the model's own error.
Seam/interior ratio (interior floored 1e-3): naive 2.0-2.9 (KILL region); halo16 0.73 / 0.85 / 1.39; halo16+far 0.54 / 0.66 / 0.82; halo8+far 1.12 / 1.41 / 1.23; halo32 0.54 / 0.65 / 0.97; G=128 copies (4x4 tiles of 32, halo 16) +far 0.57 / 0.70 / 1.06. Bodies within 8 units of the OUTER domain edge (warm N1200 only): error 0.75 even with far (single grid's own edge artifact is not reproduced by tiles whose margin is empty); not a seam issue.

### Rollouts (held out, err@20 vs truth; ratio to single grid)
| scene | single | naive | halo16 | halo16 + fx | halo16 + far | halo8 + far | halo32 | 4x4x32 + far |
|---|---|---|---|---|---|---|---|---|
| cold N300 | .0646 | 3.88 (60x) | .0725 (1.12) | .0725 (1.12) | .0645 (1.00) | .0674 (1.04) | .0646 (1.00) | .0645 (1.00) |
| warm N300 | .0705 | 3.51 (50x) | .0781 (1.11) | .0781 (1.11) | .0705 (1.00) | .0730 (1.04) | .0705 (1.00) | .0705 (1.00) |
| warm N1200 | .1129 | 2.60 (23x) | .4246 (3.76) | .4246 (3.76) | .1077 (0.95) | .1124 (1.00) | .1111 (0.98) | .1074 (0.95) |
err@5 and err@10 show the same ratios (cold: .0079/.0205 single vs .0079/.0205 halo16+far). Dev seed 4738 agrees. Mean position difference tiled vs single at tick 20: 0.0014 (halo16+far, cold/warm300), 0.014 (N1200). Momentum drift ||sum v| change| after 20 ticks: 5-9e-6 (N300) and 2-3e-5 (N1200) for every variant including single (fp32 round-off; global momentum fix); halo16+far / single = 1.1 / 0.8 / 1.2-1.3 (not distinguishable from single across 9 scenes, but not strictly "not worse" on N1200).
Long horizon (seed 4738): err@100 single 6.8 / 5.7 (N1200 / cold), halo16+far 5.2 / 5.4, naive 37 / 21; tiled-vs-single difference at 100 ticks 2-3.5 (chaos amplifies fp32 summation differences; single-vs-single reruns do this too, see scatter-scaling.md self-noise).

### 16 tiles (4x4 tiles of 64, extent 256, N=4800 sigma 32, warm; vs one G=512 grid; job 3768)
Single grid is itself poor here (err@20 3.4-3.7, far-field kernel out of training range, scatter-scaling.md). Force error vs single, seam / interior: halo16 .66 / .71 (fails), halo16+far .0129 / .0178. Tiled-vs-single position difference at 20: 3.0 vs 0.07. err@20 vs truth: halo16 1.2 (better than single because it drops the wrong far field), halo16+far 3.4-3.6 (= single, as designed). Momentum drift 2.7e-5 (far) vs 4.2e-5 (single).

### Cost and communication (job 3767, scale phase, constant density N=300 per 64x64 tile, sigma = L/8, H200)
| tiles | N | single-grid ms/tick (G) | tile copy ms per force (G, halo16) | tile peak MB | single peak MB | ghost bodies / own bodies per tile | halo cells / tile | batched-emulation ms/tick |
|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 2.4 (128) | 2.85 (128, no halo) | 33 | 33 | 0 / 300 | 0 | 2.9 |
| 4 | 1200 | 2.5 (256) | 2.98 (192) | 74 | 132 | 554 / 300 | 20480 | 4.0 (4.7 with far) |
| 16 | 4800 | 5.1 (512) | 3.10 (192; busy centre tile 1088 own + 1143 ghost) | 116 | 539 | 1143 / 1088 | 20480 | 13.6 (14.3 with far) |
Per-tile cost and memory are flat in tile count (tile copy time is launch-bound ~3 ms; per owned body 9.5 / 9.9 / 2.9 us). The batched emulation rises at 16 tiles only because it pads every tile to the busiest tile (Nmax 2258) in the dense pair term: load imbalance of a Gaussian scene (central tiles hold 3.6x the mean), not a per-tile cost rise. Far term adds ~0.7 ms (one coarse FFT over the whole domain: 1024 coarse cells per tile, 16 tiles = 128x128); it is global (all-gather of coarse density, T x 1024 floats), so its per-tile cost grows with T (redundant FFT) or needs a distributed FFT; fine to hundreds of tiles, untested beyond 16. Halo overhead: (192/128)^2 = 2.25x cells at halo 16 for 64-wide tiles; ghost bodies = 1.9x (4 tiles) own bodies for N300 sigma 16 scene (halo 16 is 25% of tile side).

### Frame review (fresh unbiased subagent, `videos/scatter_tiling_{warm1200,cold300}.png`, prompt `rollout-frame-review-prompt.md`)
Exact, single, halo16, halo16+far and 4x4x32+far rows look nearly identical at every step (diffuse cloud -> central clump by step 50-100; small differences in clump tightness, slightly looser clumps for the +far rows at step 50 in N1200). Naive: matches to step 20 in N1200 then stays diffuse and ragged ring-like at step 100; in cold300 a gap along the vertical and horizontal midlines appears at step 10 and the cloud sits as four separate clumps (one per tile) with an empty cross to step 100. No grid or stripe pattern in any other row.

## Prediction vs measured
1. naive: seam<2 error >>1 and interior degraded: confirmed (1.4-2.2 and 0.41-0.55); err@20 >=3x: confirmed (23-60x), far worse than predicted; visible four-clump split.
2. halo16 without far: predicted seam .1-.2, err@20 ~1.3x: compact scenes far better than predicted (seam .005-.008, err@20 1.11-1.12x, just over the 1.1 bar); wide scene as bad as predicted or worse (seam .10, err@20 3.8x).
3. field exchange: predicted a small rollout gain: WRONG, no measurable effect (identical to 3-4 digits; the recurrent halo field is not what limits the match). Drop it unless a different scene shows otherwise (not tested: much longer horizons, scenes with fast bulk flow across seams).
4. + far: predicted interior 2-3x better, ratio 1-1.5: interior 5-7x better and ratio .54-.82 at halo 16. Better than predicted.
5. halo 8 worse than 16, 32 marginal: confirmed without far (halo8 err@20 5-7x single on compact scenes); with far halo 8 is already within 1.04-1.05x; halo 32 without far matches single on compact scenes (1.00) but not on N1200 (0.98 err, force .014).
6. cost: confirmed flat per tile; halo overhead as computed.

## Verdict
Criteria (at halo 16, held out): halo16 + far PASS on all three scenes at 2x2 and at 4x4x32 (G=128 copies): seam/interior force ratio .54-1.06 (<=1.1), err@5/10/20 ratio .95-1.00, momentum drift at fp32 round-off of the single grid (1.1 / 0.8 / 1.2-1.3x). Halo-only (token ghosts, with or without field exchange) FAILS: err@20 1.11-1.12x on compact scenes, 3.8x on N1200, force seam error .10 and 16 tiles .66. Naive FAILS (seam/interior 2.0-2.9, 23-60x err). KILL not triggered (seam <= 2x interior at halo 16 with far; per-tile cost flat).
Minimal mechanism set: ghost bodies (halo) + coarse global far-field coupling. Field-state exchange not needed. No learned seam module needed; no retraining. Caveat: far reproduces the single grid's far field, which is wrong beyond ~16-32 units (16-tile scene: both far-matched and single err@20 ~3.5; halo-only is closer to truth 1.2 because it drops the bad far kernel). Matching the single grid and matching physics diverge there; fixing `kmlp` at large r is a separate item.

## Not tested
Real multi-device run (emulated on one GPU; exchange is array copies); distributed coarse FFT; far_c other than 4 and tile counts above 16; bodies crossing the outer domain edge; cell-list pair term with tiles (dense N^2 pp used; fine at <=2.3k bodies per tile); non-Gaussian / uniformly dense scenes; speeds large enough that bodies cross more than the halo in one tick (halo 16 vs speed*dt ~ 1); load balancing (centre tile 3.6x mean in Gaussian scenes); 100-tick tiled-vs-single agreement (chaos-limited, diff 2-3.5 at tick 100).

# Round 2 (2026-10-20): far-kernel check, cell-list pair term in tiles, load balancing

Branch `worktree-agent-a6dc0f9444b821098` (merged round 1). Remote `~/augur_tile2` (copy of `~/augur_tiling`). User decisions: strict G=128 copies (32-wide core, 16 halo, h .5), global coarse grid (cell 2) accepted for non-neighbour coupling, far kernel = cheap check only (no fix).

## 1. Far-kernel check (`scripts/scatter_far_kernel_check.py`) -- DIAGNOSTIC ONLY
Pass/kill: none (diagnostic). Question: where/how does learned `kmlp(r)` (times the pp split factor) depart from the true softened force a = r/(r^2+eps^2)^1.5 (eps .5, G=1, the exact-sim law; potential kernel K = -1/sqrt(r^2+eps^2), acc = -grad(phi), phi = sum m K) for r 40..500, does the sign flip, and how much of the wide-scene (16 tiles of 64, extent 256, N 4800 sigma 32 warm, seeds 4738/9000/12000) far-term error does it explain. Analytic kernel substituted at r > 64 and r > 32 in BOTH the local kernel and the coarse far term (hard-coded force: labelled diagnostic, not a model component).
Prediction: learned K tracks truth to ~r 30-60 (training grid 128 -> max r ~90), departs beyond (force ratio off by tens of percent, sign flip plausible at r > 100 where never evaluated); analytic r>64 brings tiled+far err@20 from ~3.5 to well under 2 (near halo-only 1.2 or better) and single-grid err@20 down likewise.

Job 3774 (first run) had a flawed splice (K stepped from learned to analytic at the threshold -> spurious ring force; err@20 10.5): discarded, kernel table sign convention also fixed (inward force = +dK/dr). Job 3775 (continuous splice K_l(thr)+K_t(r)-K_t(thr); truncation = K constant beyond thr) is the result; raw `results/scatter_far_kernel.json`.

### Result (job 3775, seeds 4738/9000/12000, means)
Learned inward force / true softened force (`F_ratio`): r 2: .74 (pp-split region), 5: .94, 10: .986, 20: 1.000, 32: .989, 40: 1.049, 48: 1.077, **64: .206, 80: -3.27, 100: -14.4, 150: -88, 200: -217, 300: -547, 400: -993, 500: -1593**. Departure starts at r ~40 (5-8% over), collapses at r = 64 (the trained extent), the force changes sign between 64 and ~70 (repulsive beyond), and grows to >1000x the true force at r 400 (kmlp extrapolates its last-layer growth; the true force falls as 1/r^2). K itself is offset by a constant (+0.14 at r 10-50), irrelevant to the gradient.
Wide 16-tile scene (N 4800, sigma 32, warm, extent 256), err@20 vs exact / force error vs exact fp64 at tick 0:
| kernel | single G=512 err@20 | tiled+far err@20 | tiled+far force err | tiled halo-only err@20 |
|---|---|---|---|---|
| learned | 3.55 | 3.56 | .875 | 1.23 |
| splice analytic r>64 (DIAGNOSTIC) | .182 | .180 | .245 | 1.18 |
| splice analytic r>32 (DIAGNOSTIC) | .170 | .168 | .244 | 1.21* |
| truncated (zero force) r>64 (DIAGNOSTIC) | .514 | .512 | .281 | 1.19 |
| truncated r>32 (DIAGNOSTIC) | 1.51 | 1.51 | .440 | 1.64 |
(*tiled halo-only is insensitive: it never uses the far kernel beyond its box.) Tiled+far still tracks the single grid (position diff at 20: .004 vs .007 learned), so the tiling scheme is not the issue.
Verdict: the learned kernel beyond r ~64 explains essentially all (95%: 3.56 -> .18) of the far-term error on the wide scene. Replacing it by the true force at r>64 (no other change) takes err@20 from 3.5 to .18, below even the halo-only drop of the far field (1.2); truncating gives .51. Prediction check: departure beyond ~64 confirmed, sign flip confirmed (predicted "plausible"); err@20 effect much larger than predicted (predicted "under 2"; got .18). The analytic splice is a hard-coded force: DIAGNOSTIC ONLY, no fix built. Constraint check: diagnostic; no retraining; model weights untouched; not part of the model path. Stopped here on the kernel per the decision. Open decision for the user: how to fix `kmlp` at r>~40 (retrain with larger extent / feed r in training, regularise the tail to decay, or clamp the kernel to its trained range) -- not started.

## 2. Density-adaptive cell-list pair term (`adapt` opt, `ScatterField.pp_acc_adapt`; bench `scripts/scatter_celllist_bench.py`)
Scheme: multilevel sorted cell list; level l has cells of side pp/2^l (pp = 2). Level 0 searches 3x3 cells (covers the whole pp window, so exact); finer levels search 5x5 and accept a query only if its 16th neighbour lies inside the guaranteed radius 2*cell, else go finer; per-query candidate list is capped (<=64 at l0, <=32 deeper, bucketed by power of 2); the few queries unresolved after 8 levels (cell 0.0078) fall back to chunked all-pairs. Same pp=2, knn=16, same top-16 semantics and force formula, no larger exact radius. Cost per query = (window cells) x (capped occupancy) independent of local density => O(N) in N and flat in density down to the fallback depth; sorting 9 levels is O(N log N) per level. Changes this round: (i) cell index from the bounding box of active bodies instead of clamping to the grid extent (clamped edge cells made bodies outside the grid pile up and be mis-searched); (ii) `qmask`: only flagged bodies are queried (tiles query own bodies only, ghosts are candidates), the stock/`cell` terms compute and discard ghost forces; (iii) `adapt_stats` (per-level resolved counts, fallback count).
Pass/kill (recorded before running): PASS if forces identical to stock within fp32 noise (rel max diff <=1e-5 of max|a|) AND ms/tick at N=1e5 cold collapse (extent 64, G 128, sigma 8, vfac .3, the scene behind the 452 ms of `cell`) at least 3x faster than `cell` (452 ms) with log-log slope <=1.3 in N; KILL if no gain over `cell` or forces differ.
Prediction: forces identical (rel diff <=1e-6) incl. dense core / masked batch / outside-grid bodies (where `cell` differs); N=1e5 cold ms/tick 60-120 (4-7x), slope 1.1-1.3 over 1e3..1e5; fallback queries < 0.1% of N; density sweep (N=1e5, sigma 32 -> .03): adapt flat within 2x while `cell` grows ~ sigma^-2 and is skipped past ~3e11 candidates. Constraints: O(N) satisfied if slope holds; pp=2/knn=16 unchanged; no retraining; no hard-coded force.

## 3. Load balancing across tiles (cold collapse; `scripts/scatter_tiling_cores.py` `CoreTiled`/`split_cores`/`lpt_assign`, `scripts/scatter_balance_eval.py`)
Setup: domain 128 (extent 128, h .5), N 20000 and 60000 (60000 seed 4738 only), Gaussian sigma 16, vfac .3 (cold: free-fall ~5 ticks), seeds 4738 / 9000, states at ticks 0,3,6,10,20,40 from a single-grid G=256 rollout (shared state, tile fields sliced from the single field), `adapt` pair term with own-only queries, far term on. Uniform = 4x4 cores of 32 (16 tiles, G=128 copies, halo 16). Per-tile time = solo force time on one tile (what one device would do; includes pack), tiles packed onto 16 workers by LPT using the measured times (profile-guided balancer), worker time = sum of its tiles.
Structural finding (no experiment needed): with strict G=128 copies and a 128 domain, 16 tiles tile the domain exactly (16 x 32^2 = 128^2), so adaptive boundaries at 16 tiles are impossible without shrinking the domain. Balancing therefore means over-decomposition: a core holding more than N/(k*16) bodies is split 2x2 (quadtree, down to 8 wide; every core is multiple-of-2 aligned, <=32 wide, own G=128 window centred on the core, halo >=16, far term unchanged), tiles are packed onto 16 workers. k = 1,2,4,8 tested (threshold N/(16k)).
Pass/kill (recorded before running): PASS if max/mean per-worker time <= 1.25 on cold collapse (ticks 6-40) with 16 workers, total work (sum of tile times) not more than 10% above uniform, and seam/interior force-error ratio (vs single grid, interior floored at 1e-3) <= 1.1; KILL if max/mean still > 2 after balancing. Also reported: makespan (max worker) vs uniform, batched ms.
Prediction: uniform own-count max/mean 4-8x after the collapse (ticks >=6) but time max/mean only ~1.5-2.5 (mesh floor ~3-4 ms per tile launch-bound; pair term with `adapt` is cheap); split reaches <=1.25 for k>=2 but each extra tile costs the mesh floor, so total work rises 30-100% (FAIL on the 10% clause) while makespan improves modestly. Seam/interior ratio <=1.1 (windows are exact copies; more internal seams in the dense region but ghosts cover them). Constraints: O(N) (tile cost = own + ghost bodies + 128^2 cells), tileable (cores partition the domain), global momentum fix kept, no retraining, no hard-coded force; load balancing is scheduling only.

### 2 Results (jobs 3776 parity+density, 3777 scale+density with lmax; raw `results/scatter_celllist_{parity,scale,density,density2}.json`)
Forces vs stock dense pair term (fp32; `cell` is bit-identical to stock): `adapt` max rel diff 1e-8 .. 4e-7 of max|a| on cold t0/t40 N 1000/3000, dense core (0.5) + halo, ultra-dense core (sigma .02, 1786 of 2000 queries via the all-pairs fallback), masked batch B=4, bodies outside the grid; adapt vs `cell` at N=1e5 (extent 64 and 256, tick 0 and 30): 1.2e-7 .. 1.8e-7. `qmask` queries match stock on the queried bodies, non-queried output exactly 0. PASS (<=1e-5).
Cold collapse, N sweep, ms/tick mean over ticks 1-10 (extent 64, G 128, the scene behind the 452 ms): 
| N | 1e3 | 3.2e3 | 1e4 | 3.2e4 | 1e5 | 3.2e5 | 1e6 |
|---|---|---|---|---|---|---|---|
| cell | 3.2 | 3.7 | 6.5 | 20.2 | **454** | - | - |
| adapt | 7.1 | 14.0 | 15.3 | 15.2 | **19.8** | 26.4 | 86.1 |
Ticks 31-40: cell 128 (3.2e4), 1718 (1e5); adapt 11.7, 11.3. Extent 256: cell 70 -> adapt 21.2 (ticks 1-10 at 1e5), 1437 -> 8.3 (ticks 31-40). Slopes of adapt: 0.11 (1e4 -> 1e5), 0.64 (1e5 -> 1e6), 0.36 over 1e3 -> 1e6 (cell: 1.84 over 1e4 -> 1e5, 2.7 over 3e4 -> 1e5). At 1e6 ticks 11-20 32.6 ms (peak work is the mesh, 86 ms at tick 1 is the pair term over the uncollapsed cloud, pair-only 126 ms at t0). Caveat: at extent 64 and N>=1e5 the 'cold' scene is flung out (88% of bodies outside the grid by tick 40) so late ticks are not dense; density is tested separately:
Density sweep, N=1e5, pair term only (ms), peak surface density 15 .. 1.8e7 per unit^2: adapt 13.0 (sigma 32), 17.1, 20.3, 26.9, 27.3, 27.0, 27.2, 27.0 (sigma .25), then 494 / 606 at sigma .1 / .03 with lmax 8 (80-100% of queries hit the all-pairs fallback: cell side pp/2^8 = .0078 is too coarse for ~25000 bodies in a 0.1 sigma clump); with lmax 14 (`model.adapt_lmax`) 32.7 / 31.9 ms. `cell`: 257, 26, 83, 308, 943, 1796, 1965 ... 1965 (saturates when everything is in one 2x2 cell). Adapt cost is flat in density over 6 orders of magnitude (13-32 ms) as long as lmax resolves the clump; cell scales as N * 9 * max occupancy.
Cost scaling: O(N) in N (slope <=1 asymptotically, mesh floor below 1e5), flat in density (bounded candidates per query: <=25 x 32 deeper levels, 9 x 64 at level 0, levels = log2 of the density contrast). Weakness: fixed ~4-14 ms host-sync/launch overhead per call (3.5 ms at N=1e3 -> 12.7 ms at t40 with 3 levels); `adapt` is slower than stock/`cell` below ~2e4 bodies (3.2 vs 7.1 ms at 1e3, 6.5 vs 15.3 at 1e4) and wins above (20.2 vs 15.2 at 3.2e4, 454 vs 19.8 at 1e5); `model.adapt_min` (default 512) sets the switch size per call (tile balance eval uses 6000). Prediction check: forces identical (confirmed, 4e-7 not 1e-6 wording), 4-7x predicted at N=1e5 -> 22.9x measured (454 -> 19.8), slope 1.1-1.3 predicted -> 0.11-0.64 measured (better, mesh-floor dominated), fallback <0.1%: confirmed for the cold scene (0 at tick 40), but 1-100% in ultra-dense clumps beyond lmax. 
Verdict: PASS (forces identical to fp32 noise; 22.9x faster than `cell` at N=1e5 cold; slope <=1.3). Constraints: O(N) (yes, flat in density to sigma .03 with lmax 14); pp=2/knn=16 unchanged; no larger exact radius; no retraining/hard-coded forces; tileable (used per tile with own-only queries).
