# Overnight 8-GPU plan 2026-11-06

Supervisor: overnight-8gpu-supervisor. Status: scripts written and committed (9dde077, 77ac6a1), NOT SUBMITTED. Main session submits.
Pass/kill lines below are fixed before launch (amendment guard: an amendment is a logged deviation that names a concrete value that would still fail it). Full pre-registrations are in docs/debugging/experiment-log.md under "2026-11-06 overnight".

## Run rules (all 8 jobs)

- One job per H200 on gpu01, `--gres=gpu:1`, qos student-pleiades, partition gpu, `--time=12:00:00`, `--requeue`, `--signal=B:TERM@120`.
- Hard stop 09:00 PST 2026-11-07 (epoch 1794070800). Every script: exit 3 if 300 s or less remain at start; payload runs under `timeout -k 120 -s TERM <remaining>` so the SIGTERM trap saves a full-state checkpoint; steps that would start after 08:55 are skipped; exit 4 means "stopped at deadline, unfinished, state saved".
- Finishing early is fine. Planned runtime is capped at about 10 h (submission about 22:30-23:30 PST gives 9.5-10.5 h).
- Per-user cap observed: 4 running jobs on student-pleiades (QOS raise pending). Submit order below puts the two long jobs and two mid jobs first; the rest start as slots free. A job still queued after 08:55 exits 3 and costs nothing.
- Every script: `--resume` or skip-if-output-exists, periodic full-state checkpoint (tiered trainers every 250 iterations and on SIGTERM), SIGTERM trap, input check (MISSING then exit 2), CPU smoke tests run in-job before the GPU work.
- Checkpoint/resume command for every job: `sbatch <script>` again (or `scontrol requeue <id>`, not after the deadline). Scripts resume from saved state and skip finished outputs.
- Resume is not bit-identical (no RNG state in checkpoints).
- Seed convention: 4738 is the base; replicate seeds here are fresh (4739-4746, 12100-12300) so nothing reuses a tried seed.

## Jobs

CPU = `--cpus-per-task`. Priority 1 (background) to 5 (urgent). "Serial GPU-h" is the estimate for the job alone on one H200 (two processes on one GPU gave 0.96x aggregate, so no concurrency credit).

| # | Name (script scripts/polaris_<name>.sh) | Priority | CPU | Est. | Lower value? |
|---|---|---|---|---|---|
| 1 | tiered-tier0-third-training-seed | 4 | 1 | 6.4 h | no |
| 2 | tiered-tier0-fourth-training-seed | 3 | 1 | 6.4 h | no |
| 3 | tiered-replay-on-second-training-base-with-fresh-seed-eval | 4 | 1 | about 2.0 h | no |
| 4 | scatter-core-drift-arm-a-fresh-held-out-seeds | 3 | 1 | 2-3 h (UNVERIFIED) | no |
| 5 | solar-n-and-m-and-k-regimes-second-training-seed | 3 | 4 | about 3.1 h | no |
| 6 | neuron-graph-extra-training-seeds-bundle | 2 | 1 | 1-1.5 h | YES |
| 7 | tiered-sharpness-aware-minimisation-adam | 2 | 1 | about 2.8 h | YES |
| 8 | interp-and-pm-baseline-completion-bundle | 1 | 1 | 45-75 min | YES |

Total about 25.7 serial GPU-h of about 80 available (8 x 10 h). This is an underutilization anomaly, flagged: no further non-repeat, non-sweep, single-GPU experiment with a clear question was found that fits. Swap candidates are the three lower-value jobs.

### 1. tiered-tier0-third-training-seed
- Question: does the tiled closed-loop tier 0 beat the frozen tiled baseline (BL = job 4060, e44b796) on a third independent training seed? Seed 4818 (seed 4739) passed thinly (0.0001 margin at eval seed 9100).
- Setup: job 4818 recipe at training seed 4740, validation seed 4741: matched-60k base from scratch (60000 it), tiled closed-loop fine-tune (5000 it, no early-stop probes), eval seeds 9100/9200/9300.
- PASS: tiled err@20 (position error at tick 20) <= 1.0x BL on every eval seed AND plain err@20 <= 1.10x matched-60k. KILL: tiled err@20 > 1.25x BL on any seed. Otherwise INCONCLUSIVE_PARTIAL. Tie-break: final checkpoint, no argmin on eval seeds. Prediction PASS 55%.
- Time basis: job 4818 took 6:24:35 (base 18030 s, fine-tune 4831 s at 0.97 s/it, plus eval). Not timed at 1 CPU.
- Files: `bash scripts/build_replay_code_tarball.sh tiered_overnight_code.tgz` from the committed tree (646 files), upload to `$HOME`. Already on Polaris from earlier jobs: `$HOME/E_ms_kp_pot_v_g128.pt`, `$HOME/t0n/data4`, `$HOME/augur_tiered_c2_tier0_matched_60k.pt`, harvest shards.
- Results: `$HOME/augur_tiered_tier0_third_training_seed/results/tier0_third_training_seed`.

### 2. tiered-tier0-fourth-training-seed
- Same question and lines as job 1, training seed 4742, validation seed 4743 (chosen to avoid 4740/4741 used by job 1 and 4739/4740 used by job 3). Copy of job 1 with names changed. Results in `results/tier0_fourth_training_seed`.
- Together with 4738, 4739, 4740 this gives n=4 training seeds. Same tarball, same time basis (6.4 h).

### 3. tiered-replay-on-second-training-base-with-fresh-seed-eval
- Question (a): does the visited-state replay that passed on seed 4738 (job 4089, 1 seed, control generic 2.10x) replicate on the second training base (4818)? Question (b): do both frozen bases (4060, 4818) hold on fresh eval seeds 12000/12100/12200?
- (a) PASS per eval seed: free-rollout force <= 1.25x BL, dE@100 (relative energy drift at tick 100 against truth E(t)) <= 2.5x BL, err@20 <= 1.0x BL, plain <= 1.10x the 4818 value. KILL: mean dE@100 > 3.5x BL or err@20 > 1.25x BL on any seed. Prediction PASS 40%.
- (b) PASS: err@20/BL <= 1.0x on seeds 12000/12100/12200 for BOTH checkpoints. KILL: > 1.25x on any (checkpoint, seed). Else INCONCLUSIVE. Floor: a seed whose BL err@20 lies outside 0.5-2x of the 9100-9300 mean is flagged, not gated. Prediction PASS 55%.
- Time basis: 4089 took 1h32 for 4500 it; plus evals about 0.5 h. Checkpoint paths for the 4060 and 4818 outputs are inferred from their job scripts and not verified on Polaris; the input check exits 2 if missing.
- Files: same tarball as job 1.

### 4. scatter-core-drift-arm-a-fresh-held-out-seeds
- Question: does the symmetrised C4 U-Net potential (arm A) fix core drift on fresh seeds 12100/12200/12300? Prior result: thin PASS on 3 seeds.
- Per fresh seed gates: core drift <= 0.007/tick; err@20 <= 1.05x stock; bh300 dE@1000 ratio <= 1.15x stock; nr100 judged on the lower end of its 95% bootstrap interval. KILL: any value above 1.5x its gate (drift above 0.0105, err@20 ratio above 1.575, dE ratio above 1.725). PASS: all gates met on every seed and pooled bh300 dE ratio <= 1.095. INCONCLUSIVE: between. A value exactly on a gate counts as pass.
- Floor: a seed counts for drift only if stock drifts at least 0.03/tick there; PASS needs at least 2 such seeds; the absolute drift gate stays. Prediction PASS 25%, INCONCLUSIVE 55%, KILL 20%.
- Time 2-3 h is UNVERIFIED (basis: job 3868 bh300 time and the 1.1 GPU-h escape study; the N=20000 exact-truth time was never measured). The deadline guard bounds it.
- Files: `git archive --format=tar -o armA.tar 0cb0471 scripts tests model conftest.py results/far_kernels results/chaos_floors.json`, then `tar rf armA.tar scripts/scatter_core_drift_fresh_seeds.py scripts/scatter_core_drift_fresh_seeds_verdict.py`, `gzip -c armA.tar > armA_fresh_code.tgz`; upload to `$HOME`. Also needs `$HOME/augur_far/kern/E_frozen.pt` and `$HOME/edge_ck/ghost_lsq.pt` on Polaris.
- Note: the merged scripts/scatter_field.py and scripts/train_scatter_field.py in the tarball are shared code that affect other users.

### 5. solar-n-and-m-and-k-regimes-second-training-seed
- Question: job 4918 left n-and-m-and-k at 940.8 moon survival INCONCLUSIVE (kill 938.9, pass 1100). Does a second training seed settle it? Runs 8 arms, CPU 4.
- PASS: survival >= 1100 and all guards. KILL: survival <= 938.9 (841.6 control plus floor 97.3) or any guard failure. Guards: held-out planets kept mean >= 0.67 and original solar >= 7 of 8 on seeds 9300 and 12000. INCONCLUSIVE otherwise. Replicates only if both new seeds give the same verdict class. Tie-break: higher survival, then narrower CI, then fewer guard failures; a gap counts only if it exceeds the CI half-width and the control's across-seed spread. Prediction: n-and-m-and-k PASS on at least one seed 15%.
- Time basis: job 4918 took 3h06 for 8 arms concurrent (23 min per arm).
- Files: `git archive -o solar_nkm_second_seed_code.tgz HEAD scripts model tests conftest.py results/solar_moon_diagnosis/moon_arm_enlarged_eval_repeat3.json results/solar_nkm_regimes/verdict.json`; upload to `$HOME`. Needs on Polaris: `~/augur-solar-moon-arms` (scene_kinds_256.pt), `~/augur-solar-more-moons`, `~/bounce-scenes/checkpoints_scenes/orbit_ft2.pt`.

### 6. neuron-graph-extra-training-seeds-bundle (LOWER VALUE)
- Question: do the seed 4738 verdicts (slow-patch PARTIAL, windowed attention KILL, plasticity KILL, bit-output PASS) hold at training seeds 4739 and 4740, each against a baseline re-trained at that seed? Mostly confirms KILLs.
- Slow-patch: PASS = slope300 >= 0.3, corr300 >= 0.3, rate ratio 0.7-1.3, err@20 <= 1.03x baseline, f1@20 >= baseline - 0.03, both N; KILL = slope < 0.2 at either N or ratio < 0.5; else PARTIAL. Windowed: PASS = err@20 <= 0.95x baseline and horizon >= 1.15x, both N; KILL = err@20 > 0.97x at either N. Plasticity: pre-registration of 2026-10-29 (job 4091): KILL if plastic forward-minus-reversed contrast <= 0 at both probes and N, or fixed contrast >= 0.75x plastic. Bit-output: PASS = bits/float particle-force-error ratio <= 1.5 global and centered; KILL if either > 3. Replicates = same verdict class as seed 4738 at both new seeds. Final checkpoint, no argmin.
- Time about 1-1.5 h (jobs 4893, 4914, 4091; baseline about 3 min; bit-output 3.5 min CPU).
- Files: `git archive -o neuron_extra_training_seeds_code.tgz HEAD model scripts tests conftest.py`; upload to `$HOME`.
- Open low findings: non-atomic torch.save; a TERM during pytest skips requeue.

### 7. tiered-sharpness-aware-minimisation-adam (LOWER VALUE)
- Question: does sharpness-aware minimisation (SAM) with Adam, rho 0.05, change drift on the job 4060 recipe (seed 4738, 5000 it)? Optimizer lever is otherwise closed (Muon swaps 4975 and 5073 null).
- PASS: dE@100 <= 2.5x BL AND err@20 <= 0.82x BL on seeds 9100/9200/9300. KILL otherwise. Caveat: frozen 4060 is 0.8215 at seed 9100, already 0.0015 above the err line; line kept as written. Prediction KILL 80%.
- Time about 2.8 h (2 x 0.97 s/it x 5000 + about 0.1 h). Files: same tarball as job 1.

### 8. interp-and-pm-baseline-completion-bundle (LOWER VALUE)
- Step 1, Yukawa lambda protocol: yukawa_lambda_protocol.py, mixed_s4738, laws yuk8/yuk32/yuk128, fresh placement seeds 4744/4745/4746, n=1024, 9 cells. Both verdicts reported, neither chosen. Literal 4832 line: wrong = selected lambda != true; PASS if wrong == 0 and no cell with p_err > 0.10 or eps_err > 0.20; KILL if wrong >= 5 of 9; else INCONCLUSIVE_PARTIAL. Tie-aware 4841 line: tied set = candidates with residual <= 1.02 x min; a cell is ok iff the true lambda is in the tied set and the set has <= 2 members; same thresholds. These lines were taken from the scorer code, not verbatim registry text; the registry floor clause was not checked.
- Step 2, low-dead mesh sparse autoencoder (SAE): CURIOSITY, no pass/kill, never to be reported as a verdict. Approved by me as supervisor; listed for the main agent to approve or escalate. Checkpoint 4738, SAE seeds 4740 and 4741, mesh sizes 16/32/64, 24 scenes with 6 held out; matched against existing SAE seeds 4738/4739; reports dead fraction, nmse, median best |cos| vs random baseline.
- No PM-baseline step (row 2 has no missing family).
- Time 45-75 min (upper bound from windows of 4832, 4834, 4841). Files: runs from `$HOME/augur_nfr` (needs `ckpt/main_s4738.pt`, `ckpt/main_s4739.pt` and the scripts in its input check); no tarball.
- The Yukawa and scorer steps were not verified on real input (the n=4 CPU smoke failed on a random checkpoint earlier).

## One-command submit (8 separate sbatch jobs, submit order = start priority)

Upload first (from the committed tree): tiered_overnight_code.tgz, solar_nkm_second_seed_code.tgz, neuron_extra_training_seeds_code.tgz, armA_fresh_code.tgz (build commands above) into `$HOME` on Polaris. Then, from the directory holding the scripts:

```
for s in tiered-tier0-third-training-seed tiered-tier0-fourth-training-seed solar-n-and-m-and-k-regimes-second-training-seed scatter-core-drift-arm-a-fresh-held-out-seeds tiered-sharpness-aware-minimisation-adam tiered-replay-on-second-training-base-with-fresh-seed-eval neuron-graph-extra-training-seeds-bundle interp-and-pm-baseline-completion-bundle; do sbatch scripts/polaris_$s.sh; done
```

Expected packing on the 4-running cap: slots 1-2 run jobs 1 and 2 (6.4 h each); slot 3 solar then SAM (3.1 + 2.8 h); slot 4 gravity then replay, neuron, interp (2.5 + 2 + 1.5 + 1 h). If the cap is raised, all 8 start at once and finish earlier. To put the higher-value replay ahead of SAM, submit replay before SAM.

## Caveats

- GPUs: the 2026-11-06 standing-rules note says Polaris GPU nodes are down; check node status before submitting. gpu01 as an H200 is unverified (scripts print nvidia-smi at start).
- Tier-0 jobs were not timed at 1 CPU; they ran with more before. If the base stage is slower, the deadline guard stops them with state saved (exit 4) and a later requeue is needed.
- Tarballs must be built from the committed state (HEAD 77ac6a1 or later), not the dirty tree.
- Checkpoint paths on Polaris for 4060/4818 outputs are inferred, not verified.
- Muon resume restores a stale learning rate (note); not relevant to these jobs unless the SAM job resumes.
- Held-out seed 12000 and the make_scenes question are open (needs-user).
