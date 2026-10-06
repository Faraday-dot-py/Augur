# Scatter-field scaling and world-scale generalization (zero-shot)

Agent 2 of docs/superpowers/plans/2026-10-04-scatter-field-hardening.md. Branch `scatter-scaling` (base `scatter-regress-harness`). Checkpoint: Polaris `~/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (Exp E), weights untouched. Scripts: `scripts/scatter_scaling.py`, `scripts/scatter_scaling_ops.py` (opt-in helpers, no edit to `scatter_field.py`), `scripts/polaris_scatter_scaling.sh`. Runs use a private copy `~/bounce_scaling` on Polaris so uploads from other agents cannot change the code under test.

## How the model is parameterised (verified in code)

`ScatterField(grid, extent)`: cell size `h = extent/grid`; trained at grid 128, extent 64 (h 0.5). `grid`, `extent`, `h` are read dynamically in `scatter`, `gather`, `kernel_acc`, `neg_grad`, `init_field`. The weights are grid-size independent (3x3 convs/avg-pool U-Net with 5 levels, and MLPs on absolute r), so a trained model can be re-targeted to any (grid, extent) by setting the three attributes (`ops.retarget`); G must be a multiple of 32 for the U-Net. Absolute-length constants that do NOT scale with the world: softening eps 0.5 (truth), pair-term window `pp` 2.0 and `knn` 16, `in_scale` 1 (density input = mass / h^2, so input amplitude depends on h and on N). Tokens outside `[-extent/2, extent/2)` get zero force and no deposit. `pp_acc` builds dense N x N tensors (position differences, r, mask, eye, topk) -> O(N^2) memory and time; this is the only N^2 term. The mesh part (scatter, FFT kernel convolution at 2G, U-Net) is O(N + G^2 log G).

## Setup

Scene: N unit-mass bodies, Gaussian cluster std sigma clipped to +-0.45 extent, per-component speed vfac*sqrt(N/(4 sigma)) (the scatter_bh scene, non-relativistic), zero net momentum, dt 0.1, eps 0.5, 100 ticks (all N; biggest N timed on fewer ticks, stated per point). Families: cold vfac 0.3 (collapse; as BH runs), warm vfac 1.0 (near virial). Reference: exact softened all-pairs fp64, 4 substeps/tick, chunked on GPU (`ops.exact_rollout`), all N. err@k = mean per-body position error vs reference (sim units); const-vel err is reported next to it as the trivial baseline. Scenes per point: 16 (N<=1000), 8 (<=1e4), 1-4 seeds (seed 4738 first; N>=31623: one scene). Extras per point: dE/E (model and truth, chunked exact energy), |dP|, dL/L, first non-finite tick, fraction of bodies outside the grid, ms/tick (CUDA synced, median), peak GPU memory above inputs, force-error probe at ticks 1/10/50 (model acceleration vs exact at the model's own positions, by radius tercile and with the pair term ablated). N sweep at the world the model was trained for (extent 64, G 128); world sweeps at N=300 plus a joint constant-density sweep:
- R: resolution only (extent 64, G 32..512)
- D: domain only (h 0.5, extent 32..512, fixed scene, G = 2 extent)
- S: scene scaled with the box at fixed G=128 (h grows)
- S2: scene scaled with the box at fixed h 0.5 (G = 2 extent), N fixed (density falls)
- J: joint constant density: h 0.5, extent L, N = 300 (L/64)^2, sigma L/8.
Chunked pair term = same selection/numerics as stock (checked, `equiv` phase), used for all accuracy runs; the stock vs chunked timing/memory is measured separately (`stock` phase).

## Predictions (written before any run)

1. Memory: stock `pp_acc` holds ~10 N^2 floats (B=1) -> ~4 GB at N=1e4, ~36 GB at 3e4, OOM near 5-6e4 on a 141 GB H200 and certainly at 1e5 (~400 GB). Chunked version: memory flat in N (a few hundred MB above inputs).
2. Time: mesh part ~2.5 ms/tick independent of N (launch-latency bound, as at N 100-300) up to N ~ 3e3; chunked exact kNN is O(N^2) compute, so the log-log slope tends to 2 beyond N ~ 1e4 (~0.3-1 s/tick at 1e5). The model is therefore NOT O(N) as written; the O(N) part is scatter/gather/FFT/U-Net, the pair term needs a cell list. Stock-vs-chunked: chunked equal or slightly slower below 3e3.
3. Accuracy, fixed world (extent 64): fine up to N ~ 300-1000 (trained to 200-300 in clusters; err/const-vel ratio similar to harness bh300). Degrades beyond N ~ 3e3: (a) density 10-100x the training maximum (OOD amplitude into the U-Net); (b) knn=16 truncates the near field once more than 16 neighbours lie within pp=2.0 (density > ~1.3 per unit^2; training peak ~2), and the kernel taper removes r < pp from the mesh force, so the near-field force is missing. Expect force-error to concentrate in the dense core and shrink when pp is ablated only at low density. At N=1e5 the model will have no skill (err ~ const-vel or worse) and may lose bodies out of the grid. Warm family less bad than cold (less collapse). err@50/100 saturate (chaos) for both.
4. Joint constant-density sweep (J): per-body density constant so (b) does not bite; expect error per body to stay within ~2x of N=300 for L up to 128-256, degrade where U-Net/in_scale/eps-vs-h are OOD (L=16 with G=32 at the small end; L=1024 at the big end), runtime ~ N^2 in the pair term.
5. World scale: D (domain only, same cell size): robust; zero-padding changes nothing except larger FFT, error within ~20% of L=64 for L>=64; L=32 worse because the cloud (sigma 8) is clipped/leaves the grid. R (cell size at fixed scene): h 0.25 and 1.0 within 2x of h 0.5; h=2 clearly worse (pair window 2.0 = one cell; mesh cannot resolve), h=0.125 worse (kernel evaluated at unseen r, in_scale density amplitude x16). S (h grows with box): worse than D, since the scene geometry in cells is constant but eps and pp (absolute) are not scaled; error in absolute units grows with L (dynamics speed ~ sqrt(N/sigma) falls with L so tick motion in cells is similar), expect usable (err/const-vel < 0.2 at tick 20) only for L in [32, 128]. No retraining is needed technically (shapes unchanged, no crash); accuracy is the question.
6. Conservation: |dP| ~ float noise (momfix) at all N; dE/E grows with N in the cold family due to collapse (truth dE/E also nonzero from softening/integrator).
7. First non-finite tick: none expected at N<=3e3; possible blow-up in the dense cold N>=1e4 runs.

## Results

(filled in below after the runs)

Raw: `results/scatter_scaling_{nsweep,worlds,stock,equiv,fprobe}.json` (chunked pair term; Polaris jobs 3304 smoke/comp, 3305 stock phase, 3308 nsweep+worlds, 3311 dumps, 3313 force probes), `*_cell.json` (scatter-opt `cell` pair term via `sf.apply_opt(m,"cell")`; job 3314; `nsweepL_cell` job 3320: extent 256, G 512). Tables: `results/scatter_scaling_tables.md` and `_cell.md` (`scripts/summarize_scatter_scaling.py [--tag _cell]`). Plot `videos/scatter_scaling_nsweep.png`; rollout grids `videos/scatter_scaling_n{1000,10000,100000}.png`. Scenes per point: 16 (N<=1000), 8 (<=1e4), 4 (1.8e4), 1 scene at N>=3.2e4 (no seed-noise estimate there). ratio = err@20 / constant-velocity err@20. self-noise = err@20 between two identical model runs (GPU index_add nondeterminism amplified by chaos; in the JSON). Chunked and cell pair terms give identical forces (max diff 0.0 at N 600..1e5) and identical err@k to 3 digits, so accuracy rows are shared.

## Results

### 1. N sweep at the trained world (extent 64, G 128, h 0.5), cold collapse family (vfac 0.3), seed-mean

| N | err@5 | @20 | @100 | ratio@20 | dE/E (truth) | frac bodies outside grid | ms/tick chunked | ms/tick cell | peak MB chunked / cell |
|---|---|---|---|---|---|---|---|---|---|
| 2 | .0005 | .011 | .22 | .27 | .029 (9e-5) | 0 | 2.4 | | 33 |
| 10 | .0014 | .017 | .33 | .07 | .016 | 0 | 2.4 | | 33 |
| 100 | .0040 | .031 | 3.3 | .02 | .012 | 0 | 2.5 | | 33 |
| 316 | .0080 | .067 | 5.5 | .02 | .018 (3.6e-4) | .03 | 2.5 | | 34 |
| 1000 | .016 | .58 | 12.9 | .05 | .12 (1.2e-3) | .08 | 2.6 | 3.0 | 35 / 59 |
| 3162 | .064 | 6.8 | 59 | .39 | 1.1 (7.5e-3) | .24 | 2.5 | 3.0 | 196 / 363 |
| 1e4 | 1.8 | 42 | 309 | 1.71 | 3.0 (.06) | .70 | 6.6 | 5.2 | 1911 / 1478 |
| 3.2e4 | 12.6 | 93 | 600 | 2.2 | 3.5 (.61) | .78 | 51 | 19 | 4779 / 1486 |
| 1e5 | 35 | 237 | 1370 | 1.67 | 4.5 (1.5) | .90 | 491 | 452 | 4789 / 1493 |

Warm family (vfac 1.0, no collapse): err@20 .0303 (N 100), .307 (1e3), 5.85 (1e4), 63.9 (1e5); ratio@20 .02 / .03 / .14 / .49; ms/tick with cell 2.9 / 3.6 / 94.5 (chunked 2.5 / 6.6 / 490). No non-finite value at any N, either family (errors saturate at box size instead). |dP| is 3e-5 at N=316, 0.02 at 1e4, ~1 at 1e5 (momfix cancels mean model acceleration, but mass-weighted removal does not make it zero once bodies leave the grid).

Usable (ratio@20 < 0.1): N <= ~1e3 cold, ~5.6e3 warm. err@20 exceeds constant velocity at N ~ 1e4 (cold). Between N=316 and 3e3, err@20 grows ~N^1.7. The trained envelope was 10-200 bodies plus clusters to ~300.

### 2. Force probes (model acceleration vs exact at the model's own positions; acceleration-weighted relative error; tick 1 = initial Gaussian)

Cluster sigma 8: N = 75 / 300 / 1200 / 4800 / 19200 gives .144 / .104 / .063 / .052 / .069 at tick 1 (cosine .99+). So the initial force is good up to 2e4 bodies in the 64 box; at N=1e5 it is .31 (cold) / .15 (warm), concentrated in the core (.44 vs halo .16). Ablating the pair term (`nopp`) doubles the error at N<=300 (.17 vs .10) and changes nothing at N>=1e3. Failure at large N is therefore dynamic, not a bad initial force: cold N=3162 tick 10 error .15 uniformly; tick 50 core .07 but halo 1.9 (cos .75) where |a| is ~1/100 of the core; N=1e4 tick 10 halo 2.15 vs core .10. Error sits in the low-acceleration outskirts and coincides with collapse plus ejection of bodies (fraction outside the grid .24 to .70 for N 3e3 to 1e4; those bodies get zero force while the truth keeps attracting them). Untested: which of (zero force outside the grid) vs (outskirt noise floor) dominates. Section 4 partly separates them.

### 3. World-scale sweeps (N=300 unless stated; same weights)

| family | setting | err@5 | @20 | ratio@20 | tick-1 force err | verdict |
|---|---|---|---|---|---|---|
| R resolution only (extent 64) | G32 h2 | .128 | 1.79 | .46 | .53 | poor |
| | G64 h1 | .093 | 1.12 | .29 | .38 | poor |
| | **G128 h.5 (trained)** | .0076 | .063 | .02 | .10 | good |
| | G256 h.25 | .227 | 1.93 | .50 | 1.30 | fails |
| | G512 h.125 | .86 | 7.9 | 2.0 | 5.1 | fails |
| D domain only (h .5, scene fixed) | extent 48 | .0084 | .071 | .02 | | ok |
| | 64, 96, 128, 256, 512 | .00755 | .0631 | .02 | .10 | identical to 4 digits |
| | 32 | .027 | .29 | .07 | | worse: 19% bodies leave the box |
| J joint constant density (h .5, N=300 (L/64)^2, sigma L/8) | L16 N19 | .0104 | .084 | .03 | | ok |
| | L32 N75 | .0084 | .063 | .02 | | ok |
| | L128 N1200 | .0080 | .074 | .02 | .092 | ok |
| | L256 N4800 | .213 | 3.5 | .93 | .85 | fails |
| | L512 N19200 | 4.3 | 79 | 21 | 16.6 | fails |
| S2 scene scaled with box, h .5, N 300 | L32 / 64 / 128 / 256 / 512 | .019 / .0076 / .0035 / .0136 / .066 | .57 / .063 / .030 / .21 / 1.06 | .07 / .02 / .02 / .48 / 6.7 | | ok to L128, fails L>=256 |
| S scene scaled with box, G128 (h varies) | L16 h.125 / L32 h.25 | 3.4 / .53 | 32 / 4.9 | 4.7 / .64 | | fails |
| | L128 h1 / L256 h2 / L512 h4 | .038 / .029 / .069 | .47 / .43 / 1.1 | .39 / .96 / 6.9 | .42 (L128) | poor |

Conclusions:
- Domain extent is free at fixed cell size: D is identical to four digits for extent 64..512 (zero-padded convs/FFT see an empty periphery). Cost is only G^2 (G=1024: 9.6 ms/tick, 2.1 GB).
- Cell size is not free: error has a V shape in h with the minimum at the trained 0.5 (tick-1 force error .10 at h .5, .17 at .7, .38 at 1, 1.3 at .25, 5 at .125). Going finer is much worse than coarser. In code, `kmlp`, the pair window pp=2.0, eps and the density `in_scale` are in absolute units and the U-Net was trained at h=.5. So "fixed grid count at another extent" fails; "fixed cell size, G scaled with extent" works.
- Large scenes fail even at fixed h and constant density (J, L>=256; S2 with N fixed, L>=256), though the mesh is happy. The sigma sweep at N=300 (tick-1 force error, cells in parentheses): .074 (sigma 4), .104 (8), .143 (16), .57 (32, halo error 1.5), 8.3 (64, cosine negative, force points the wrong way). Scenes wider than ~16-32 units (trained half-width 32) get a wrong far field. Code-consistent but untested cause: `kernel_acc` evaluates `kmlp(r)` for r up to the box size but training only saw r <~ 64; direct test would compare `kmlp` against the true 1/r form for r in 40..500.

### 4. N sweep at extent 256, G 512 (h .5, same sigma-8 scene), cell pair term

Cold: err@20 .031 (100), .067 (316), .58 (1e3), 6.75 (3.2e3), 39 (1e4), 83 (3.2e4), 244 (1e5); ratio .02 / .02 / .05 / .39 / 1.6 / 2.0 / 1.7. Warm: err@20 .030 / .070 / .26 / .90 / 2.4 / 7.5 / 26.9; ratio .02 / .02 / .03 / .04 / .055 / .10 / .20 (extent 64: .49 at 1e5). Fraction outside the grid at 1e5: cold .998, warm .22 (extent 64: .90 / .39). ms/tick 3.3 (100), 4.4 (1e3), 7 (1e4), 16.7 (3.2e4), 69 cold / 87 warm at 1e5; peak 530 MB to 1.5 GB. A 4x bigger box helps warm (fewer bodies lost) and does nothing for cold: the cold failure is collapse density, not the boundary.

### 5. Time and memory

- Mesh stack (scatter, FFT kernel, U-Net, gather): ~2.3 ms per force eval independent of N to 1e5 (job 3304 components at 1e5: .63 / .38 / .72 / .55 ms). Tick time is launch-bound at 2.4-2.6 ms up to N ~ 3e3. O(N + G^2 log G) as designed.
- Stock `pp_acc`: dense N x N tensors. 8.5 ms/tick at 1e4, 22 at 1.8e4, 64 at 3.2e4, 205 at 5.6e4 (peak 72 GB), OOM at 1e5 (single 80 GB allocation). Peak ~7 bytes per pair (23 GB at 3.2e4).
- My chunked exact-kNN (`scripts/scatter_scaling_ops.py`, equal output): memory capped at 4.8 GB, 10-20% faster at N>=1e4, still O(N^2): log-log slope of ms/tick for N>=1e4 is 1.89 (all-N slope 0.31); at 1e5 491 ms/tick, 99.5% in the pair term (445 of 447 ms per eval).
- scatter-opt `cell`: warm 1e5 94.5 ms/tick (5x faster than chunked, slope ~1.4), cold 1e5 452 ms (slope 1.9): when many bodies collapse into a few pp-sized cells the per-query candidate count 9*M (M = max cell occupancy) grows. Sent to agent 1 with numbers.
- The model is therefore O(N) in the mesh, O(N*M) in the pair term, O(N^2) in a collapsed cluster until the cell search is density-adaptive.

### Prediction vs measured
- Memory: predicted OOM near 5-6e4 (about 10 N^2 floats); observed 72 GB at 5.6e4 and OOM at 1e5 (about 7 bytes per pair). Chunked memory flat: confirmed (4.8 GB). Time slope 2: confirmed (1.89).
- Accuracy: predicted breakdown near N ~ 3e3 and no skill at 1e5: confirmed for cold (ratio >1 at 1e4), better than predicted for warm (ratio .49 at 1e5). Hypothesis (b), knn=16 truncation of the near field: refuted by the probe (`nopp` has no effect at N>=1e3; error appears dynamically in the outskirts).
- Joint constant density: predicted ok to L 128-256; observed ok to 128, fails at 256. D: predicted ok for L>=64, worse at 32: confirmed.
- R: predicted h .25 and 1.0 within 2x of .5: WRONG (h .25 is 25x worse in err@20, h 1 is 5x worse). The model is much more tied to cell size than predicted.
- First non-finite tick: none anywhere (predicted possible blow-up in dense cold runs: wrong, errors saturate at the box size).

### 6. Visual review (unbiased subagent, N=1e5 cold, `videos/scatter_scaling_n100000.png`, prompt in `scatter-scaling-review-prompt.md`)

Reference: smooth blob collapses to a very bright compact core with a faint speckled halo filling the frame by tick 5; core stays compact to tick 100. Model: no bright core at any step; bodies spread diffusely over the box by tick 5, nearly flat dim density at ticks 10-20, a faint large roundish cloud re-forms near the centre by ticks 50-100. Per-body error is unimodal and broad (not a few outliers); mean error exceeds the box size within a few ticks and reaches ~1400 at tick 100, so most model bodies are outside the displayed frame. (The reviewer also noted a tick-0 mismatch between the error histogram and the mean-error plot; that plot indexes tick 0 as the starting state in the histogram (1e-6 floor) and the err curve I plotted shows the first-step jump, not a true tick-0 error.) Matches the numbers: the model never reproduces the collapse at 1e5.

### Decisions for the user
- Zero-shot N scaling holds to ~1e3 (collapse) or ~5e3 (warm) bodies per 64-unit box. Beyond that needs training on dense/large clusters (scene agent) or an N-aware input normalisation; not tested here (no fine-tune).
- World rescaling is free only at fixed cell size 0.5 and scenes up to ~16-32 units wide; otherwise retrain or make the kernel/pair terms scale-free (express r in cell units, train with random h; test kmlp extrapolation at r>40).
- Agent 1: make the cell-list pair term density-adaptive.
