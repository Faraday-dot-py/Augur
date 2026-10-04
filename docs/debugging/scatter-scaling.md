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
