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
