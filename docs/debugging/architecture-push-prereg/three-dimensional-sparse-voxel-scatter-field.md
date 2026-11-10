# three-dimensional-sparse-voxel-scatter-field, stage 1 pre-registration (written 2026-11-10, before any run)

Spec: `docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md` section 5 (cost 1.5 GPU-h stage 1, 20 GPU-h full run). Stage 1 reports back to the user before the full run starts.

Definitions. N: body count. Tick: one step of dt 0.1. err@k: mean position error per body after k self-fed ticks against the exact fp64 simulation (softened gravity, G=1, eps 0.5, 4 substeps per tick). Voxel: a 3D grid cell. Halo: the one-voxel ring of empty neighbours stored around every occupied voxel. CIC: cloud-in-cell, trilinear spreading to 8 corners. Plummer sphere: the standard self-gravitating ball with an exact density profile and distribution function. the scatter-field control (ms_kp_pot_v_g128): the pinned 2D gravity checkpoint `ms_kp_pot_v_g128`.

## Model (stage 1)

- `model/sparse_voxel_scatter_field.py`: only occupied voxels plus a halo exist. Voxel coordinates are packed into int64 keys (19 bits per axis, batch index above bit 57), so memory is independent of the box size.
- CIC deposit of mass and momentum to 8 corners. Sparse 3x3x3 convolutions through a searchsorted neighbour table. Coarsening by halving coordinates until at most 64 voxels per scene remain. Sparse U-Net up the levels. Potential output, central-difference gradient, trilinear gather.
- Far field: a coarse learned radial kernel summed directly over the coarsest voxels (piecewise-constant, no FFT, no recurrence). Pair term: cell-list over cells of edge 2 (pp=2), 16 nearest neighbours (knn=16), cell capacity 32. Same `step` interface as `ScatterField`, momentum fix on, Verlet.
- Training: the scatter-field control's scene mix in 3D (N in 10..200; half Plummer spheres, half cold balls), exact 3D truth, batch 8, k ramp 4 to 20, lr 5e-4, 10,000 iterations in stage 1 (user decision 2026-11-10; the control's count is 60,000, extension to 60,000 needs a separate user go and resumes the same checkpoint). Seed 4738.

## Lines (copied from the spec)

PASS 3D Plummer-sphere err@20 within 2x the 2D ratio to exact at equal N AND cost at most 12x the 2D model. KILL cost above 12x or err@20 above 4x. Fails pass: 3x.

Not loosened. Any amendment goes to the user as a proposal with a concrete value that would still fail it.

## Measurement and interpretations (stated now, flagged)

- Control: the scatter-field control (ms_kp_pot_v_g128) (2D), Polaris `~/augur_nfr/kern/E_frozen.pt` (3,200,305 bytes, verified read-only 2026-11-10); same file at `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`. Grid 128, extent 64.
- Equal N: N=1000 for accuracy (a 2D Gaussian cloud, sigma 8, velocity factor 0.3, versus a 3D cold Plummer sphere, scale 8, velocities scaled by sqrt(0.3) of the virial distribution, about 0.55x; the two cold families are matched in regime, not in a shared definition).
- "Ratio to exact" is read as err@20 divided by the zero-force error at tick 20 (norm of exact position at tick 20 minus initial position minus initial velocity times elapsed time, mean over bodies), on the same scene. This is an interpretation: the spec does not define it. The per-scene numbers are saved so another reading (for example error over the exact-vs-4x-substeps floor) can be recomputed without a rerun.
- Accuracy pass test: ratio3 / ratio2 at most 2.0; KILL above 4.0 (3.0 is the spec's fails-pass point). Primary family: cold. A virial family (velocity factor 1.0) is reported, not scored.
- Cost: ms per tick of the 3D model over ms per tick of the 2D control at the same N, measured at N=1e5 (also N=1e3 to 1e5 for the exponent, reported). Cost at 1e5 is the scored cost; 1e6 is not measured in stage 1.
- Scoring seeds 9100, 9200, 9300, 4 scenes per seed. Fresh seeds 12000, 12100, 12200 (`--heldout`) before a result is called real, on the same unchanged checkpoint.

## Tie-break rule

PASS lines inclusive (ratio exactly 2.0, cost exactly 12.0 pass); KILL lines exclusive (4.0 and 12.0 do not kill). Order as coded in `verdict()`: cost above 12 is KILL; ratio above 4.0 is KILL unless the plateau guard fires (then INCONCLUSIVE-undertrained); ratio at most 2.0 with cost at most 12 is PASS; otherwise INCONCLUSIVE-undertrained when the guard fires, else INCONCLUSIVE-between-lines (3x is reported as a fail of pass, not a kill). A missing 1e5 timing is INCONCLUSIVE-no-1e5-timing.

Stage 1 verdict rule (user decision 2026-11-10): at fewer than 60,000 iterations the accuracy outcome is INCONCLUSIVE-undertrained by design. The cost line is the stage 1 decider: cost above 12x is KILL; otherwise INCONCLUSIVE-undertrained with the cost reported against the line. KILL-nonfinite still applies.

## Ratio to exact (worker's interpretation, user-accepted 2026-11-10)

Ratio to exact = err@20 divided by zero-force err@20, the same quantity in 2D and 3D. Zero-force err@20 is the mean over bodies of |exact position at tick 20 - initial position - initial velocity x 0.1 x 20|.

The 2D comparison value (the denominator of the PASS line, ratio2) is measured in the same job, not taken from a file (no earlier measurement exists for this control at N=1000): 2D Gaussian cloud, sigma 8, velocity factor 0.3, N=1000, `ss.make_scene`, eval seeds 9100, 9200, 9300 with 4 scenes each (12 scenes), control `~/augur_nfr/kern/E_frozen.pt` (md5 9b3cb8bb1de47eb6ebf5f708d98e94b7, verified on the Polaris copy 2026-11-10). Averaging: ratio of means, mean over the 12 scenes of err@20 divided by mean over the 12 scenes of zero-force err@20. The mean of per-scene ratios is also reported (`ratio2_mean_of_per_scene_ratios`), not scored. The 3D ratio uses the same ratio-of-means form over its 12 Plummer scenes. The scorer logs both on a line starting "2D control ratio to exact" and stores them with the control md5 in score.json.

## Plateau guard

Validation err@20 (48 3D scenes, seed 9400) logged 50 times. INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window. Per user decision 10a.1, one extension at double length (120,000 iterations) is then allowed, candidate and control together. Control status: the 2D control is the frozen 60,000-iteration checkpoint, scored only; the extension needs it retrained at 120,000, a needs-user decision.

Non-finite handling: a NaN or infinite ratio, or a non-finite candidate rollout, returns KILL-nonfinite. A CUDA out-of-memory error in the bench stops the job nonzero and records oom_at_n in bench.json.

## Prediction (stated before running)

PASS about 25%. Expected: cost ratio between 3x and 8x at N=1e5 (sparse bookkeeping dominates; untimed guess), ratio3/ratio2 between 1.5 and 4 after 10,000 iterations; the stage 1 outcome is INCONCLUSIVE-undertrained by rule, so the cost line decides (cost KILL about 20%). The coarse far-field kernel is the weak point at N=1000.

## Constraints check (re-read 2026-11-10)

- pp=2 knn=16 stays: yes, same pair-term radius and neighbour count as the 2D model.
- Generalization: no force law hard-coded; the 2D-to-3D move is the generalization test. O(N): memory and cost scale with occupied voxels and bodies, not box volume (tested: node count independent of extent).
- Tileable: not addressed in stage 1. Interpretability: unchanged from the scatter field.
- No pairwise N^2 term: the pair term is the cell-list, bounded to 16 neighbours.

## Known weaknesses (kept in the notes)

- Stage 1 trains 10,000 iterations against 60,000 for the control (about 2 h untimed estimate). The control is frozen and scored only. Extending the same run to 60,000 restarts the cosine lr schedule (it depends on the iteration count), so the extended run is not identical to a fresh 60,000 run.
- The far-field kernel is coarse and has no recurrence over scales.
- Cost is measured at 1e5 only; the dense 3D comparison (audit row 33) is not run.
- Full run (20 GPU-h): same script with batch 16, k-end 30 (stage 1 uses batch 8, k-end 20); needs periodic full-state checkpoints (already written every 100 iterations by `tsf.train`) and `--resume` via requeue. Needs the user's go-ahead after this report.

## Files and run

`scripts/three_dimensional_sparse_voxel_scatter_field.py` (modes train, score, bench, all; resumable; SIGTERM saves and exits 143). Job `scripts/polaris_three-dimensional-sparse-voxel-scatter-field.sh`. Tests `tests/test_sparse_voxel_scatter_field.py` (12). Estimate (untimed; no logged 3D rate): about 2 GPU-h at 10,000 iterations; the job logs s_per_it_first100= to measure the rate.
