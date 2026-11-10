# ball-tree-particle-attention-tier-0, stage 1 pre-registration (written 2026-11-10, before any run)

Branch `ball-tree-particle-attention-stage-1`. Spec: `docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md` section 4, with user decision 5 in section 10a (stage 1 only, then report back). The full run happens only after the user reviews the stage 1 report.

Definitions. N: body count. Tick: one step of dt 0.1. err@k: mean position error per body after k self-fed ticks against the exact fp64 simulation (softened gravity, G=1, eps 0.5, 4 substeps per tick). Cold family: Gaussian cloud sigma 8, velocity factor 0.3 (the regime the scatter-field control (ms_kp_pot_v_g128) was selected on); warm family: same with velocity factor 1.0. the scatter-field control (ms_kp_pot_v_g128): the pinned gravity checkpoint `ms_kp_pot_v_g128` (grid 128, extent 64, FFT kernel, pair term). The spec calls the control "G"; the task assigns the scatter-field control (ms_kp_pot_v_g128).

## Model (stage 1)

- One network trained end to end (42,834 parameters, `model/ball_tree_attention.py`). No grid, no FFT, no pair term (the ball tree is a different family, user-approved).
- Bodies are sorted by the widest axis recursively into balls of 16 (padding last). Attention runs inside each ball. Between levels, pooling merges two child balls into a parent; unpooling adds the parent feature and an asinh-scaled position offset back to children.
- SUMMED attention, not normalised: per-edge softplus weights multiplied by the source ball mass over the ball size, summed over sources with no softmax or division by the weight total. At each merge, node mass is the SUM of the children, and node position and features are MASS-WEIGHTED means.
- Per-body force readout; the last layer is zero-initialised; momentum fix on (net force removed); same `step` interface as `ScatterField` (Verlet, dt 0.1).

## Lines (copied from the spec)

PASS err@20 at most 1.15x G at N 1000 AND cost exponent at most 1.10 from 1e3 to 1e6 AND at most 100 ns per body per tick at 1e6. KILL err@20 above 2.0x or exponent above 1.25. Fails pass: err@20 1.4x.

Not loosened. Any amendment goes to the user as a proposal with a concrete value that would still fail it.

## Measurement

- Control: the scatter-field control (ms_kp_pot_v_g128), Polaris `~/augur_nfr/kern/E_frozen.pt` (3,200,305 bytes, verified read-only 2026-11-10); identical file at `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (3,200,305 bytes). Loaded with the repo `scatter_scaling.load`, retargeted to grid 128, extent 64.
- err@20 ratio = mean candidate err@20 over scenes divided by mean control err@20 over the same scenes. Primary family: cold, N=1000, held-out seeds 9100, 9200, 9300, 4 scenes per seed (12 scenes). Warm family reported alongside, not scored.
- Cost exponent: log-log slope of ms/tick against N over N = 1000, 3162, 10000, 31623, 100000, 316228, 1000000 (the 1e3 to 1e6 slope only; the 1e4 and 1e5 slopes are reported, not scored). ns per body per tick at 1e6 is ms/tick over N, from the 1e6 cell. A missing 1e6 cell (OOM) gives INCONCLUSIVE-no-1e6-timing, never PASS.
- Fresh seeds (copy the results directory, then rerun `--mode score --heldout 12000,12100,12200 --out <copy>`; score needs val_curve.json for the plateau guard): before a result is called real, rerun scoring with `--heldout 12000,12100,12200` on the same unchanged checkpoint. A PASS or KILL that flips on fresh seeds is reported as not real.
- Tiled-versus-global difference: reported without a line (curiosity for that metric). Not measured in stage 1; recorded as open for the full run.

## Tie-break rule

Values exactly on a boundary: PASS lines are inclusive (ratio 1.15, exponent 1.10 and 100 ns pass); KILL lines are exclusive (2.0 and 1.25 do not kill).

Verdict order, as coded in `verdict()`: (1) exponent above 1.25: KILL regardless of training length (cost does not depend on training). (2) Error ratio above 2.0: KILL, unless the plateau guard fires, then INCONCLUSIVE-undertrained. (3) Ratio at most 1.15 AND exponent at most 1.10 AND at most 100 ns: PASS. (4) Ratio above 1.15 with the guard firing: INCONCLUSIVE-undertrained. (5) Anything else (for example ratio 1.4, the spec's "fails pass" point, or ratio at most 1.15 with exponent 1.10 to 1.25): INCONCLUSIVE-between-lines, reported as a fail of pass and not a kill.

## Plateau guard

Validation err@20 (48 scenes, seed 9400, N in 10..200) is logged 50 times over training. INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of the iterations is more than 3% below the mean over the preceding 10% window (5 points each). Per user decision 10a.1, one extension at double length (120,000 iterations) is then allowed for candidate and control together. Control status: the control is the frozen 60,000-iteration checkpoint, scored only, never retrained in stage 1; the candidate therefore trains 60,000 iterations to keep the control's count. The extension needs the control retrained at 120,000, which is a needs-user decision before it is run.

Non-finite handling: a NaN or infinite err@20 ratio, or a non-finite candidate rollout, returns KILL-nonfinite.

## Prediction (stated before running)

PASS is unlikely (about 15%). Most likely outcome: err@20 ratio 1.5x to 4x (60,000 candidate iterations, the control's count, training on N up to 200 and scoring at N=1000), so INCONCLUSIVE-undertrained or KILL on error. Cost exponent expected 1.0 to 1.1 (sorting and fixed-size balls); ns per body per tick at 1e6 expected above 100 because sorting and several attention blocks run as separate PyTorch kernels (untested guess).

## Constraints check (project constraints re-read 2026-11-10)

- Generalization: no force law is hard-coded; the model learns from trajectories. Satisfied in design; the N=200 to N=1000 gap tests it.
- O(N): tree sort O(N log N) in the sort only, attention cost linear in N for fixed ball size. The exponent line measures it.
- Tileable: at risk (tree partitions are not translation-invariant); reported, not scored.
- Interpretability: not addressed in stage 1.
- pp=2 knn=16 stays for scatter-field models; this model has no pair term; the ball tree is a different family, user-approved (decision 5).
- No hard-coded conservative forces: the momentum fix is the same net-force removal the scatter field uses.

## Known weaknesses (kept in the notes)

- Unmatched training N range against the control (candidate N up to 200 in training). Iteration count matched at 60,000.
- The control is a frozen checkpoint, scored only; the double-length extension would need the control retrained, a user decision.
- Sorting by the widest axis changes ball membership discontinuously as bodies move.

## Files and run

- Script `scripts/ball_tree_particle_attention.py` (modes train, score, bench, all; resumable; SIGTERM saves and exits 143). Job `scripts/polaris_ball-tree-particle-attention-tier-0.sh`. Tests `tests/test_ball_tree_attention.py`. Tarball `scripts/build_architecture_push_stage1_code_tarball.sh`.
- Estimate (untimed): about 0.3 s/it from the nearest logged rate (width-32 scatter field, 0.28 to 0.31 s/it): 30 min training, with scoring and the 1e6 bench 1.0 to 1.5 GPU-h (spec: 1.5).
- Outcome gets logged to `docs/debugging/experiment-log.md` by the supervisor.
