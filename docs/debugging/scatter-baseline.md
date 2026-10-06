# Scatter-field frozen baseline / regression harness

Script: `scripts/scatter_regress.py` (Polaris wrapper `scripts/polaris_scatter_regress.sh`). Baseline: `results/scatter_baseline.json` (Polaris job 3296, H200 NVL, idle GPU, git of harness 'fbf3b79').

Pinned checkpoint: Polaris `~/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (Exp E, 60769 it). Load flags: `tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", dt=0.1)` (grid 128, extent 64, U-Net, kernel, pp 2.0, split, potential, Verlet), eps 0.5; identical to `scatter_bh.py`. These are the script defaults.

## Usage

```
# on Polaris (~/bounce), one GPU job at a time
CKPT=checkpoints/scatter_bh/<cand>.pt OUT=results/<cand>_regress.json BASE=results/scatter_baseline.json sbatch scripts/polaris_scatter_regress.sh
# or directly
PYTHONPATH=. python scripts/scatter_regress.py run --ckpt <ckpt> --out <out.json> [--exp E --variant ms_kp_pot_v_g128]
PYTHONPATH=. python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate <out.json>   # exit 1 on FAIL
```
Log lands at `~/bounce-scatter-regress-<id>.log` (home, not `~/bounce`). Runtime ~2 min. Timing needs an exclusive GPU (wrapper prints nvidia-smi first; check no other compute apps are listed). An optimized candidate with the same weights is expected to reproduce errors to float noise; a retrained candidate is judged by the gate.

## Protocol (frozen)

Seeds 9100, 9200, 9300, 4738; 100 ticks, dt 0.1, eps 0.5.
- `two_body`: 48 scenes/seed, 2 unit-mass bodies (`gs.make_dataset`, scale off), batched.
- `expB`: 48 scenes/seed, 10-100 bodies, scale-init, batched (padded).
- `bh300`: 1 scene/seed, 300 bodies, sigma 8, vfac 0.3, c 10, relativistic kick-drift-kick on momentum with the learned force (scatter_bh.py logic), truth = exact softened all-pairs relativistic sim, 4 substeps/tick.
- err@k = mean per-body position error vs truth at tick k (sim units). dE/E, |dP|, dL/L at tick 100 (A/B: nonrelativistic; BH: relativistic energy, momenta p = gamma v); model and truth both reported.
- Timing: whole-rollout wall / 100, CUDA-synchronised, 1 warm-up rollout, median of 5. Peak memory = max allocated above the pre-run allocation. Entries: two_body (batch 48), expB (batch 48, N<=100), bh300 (batch 1), expB_batch1_n100 (batch 1, N 100).

Gate (`compare`): err@5/10/20 seed-mean per regime <= baseline x1.15 (+1e-5 abs); tick_ms <= x1.05; peak memory <= x1.10, for each regime. err@50/100 printed, not gated (chaotic).

## Baseline numbers (mean over 4 seeds)

| regime | err@5 | @10 | @20 | @50 | @100 | dE/E@100 (truth) | |dP|@100 | dL/L@100 |
|---|---|---|---|---|---|---|---|---|
| two_body | .00105 | .0031 | .0091 | .0408 | .131 | .129 (6e-5) | 3.5e-10 | .67 (truth 4e-12) |
| expB | .00301 | .00813 | .0240 | .157 | 1.42 | .0108 (1.1e-4) | 3.3e-6 | .40 (1e-12) |
| bh300 | .00795 | .0209 | .0622 | .912 | 4.77 | .0478 (2.5e-4) | 5.6e-5 | .30 (2e-4) |

Per-seed values, |dP|, dL/L and BH vmax/c are in the JSON. Seed spread of err@5/10/20: two_body ~30% (err@5; ~15% at @10/20), expB ~4%, bh300 ~10%; the 15% gate is therefore tight for two_body err@5 only if a retrained candidate is judged (the seed-mean is gated, not single seeds).

| timing | ms/tick | peak mem above inputs |
|---|---|---|
| two_body (B 48, N 2) | 4.73 | 830 MB |
| expB (B 48, N<=100) | 4.95 | 835 MB |
| bh300 (B 1, N 300) | 2.48 | 33.5 MB |
| expB_batch1_n100 | 2.45 | 33.2 MB |

## Sanity check vs committed BH result

Seed 4738, bh300: err@5/10/20/50/100 = .00828/.0227/.0673/.895/4.93 vs committed (`results/scatter_bh_300.json`) .008/.023/.067/.90/4.9. Match.

## Notes
- two_body/expB numbers are for the Exp E checkpoint (trained on 10-200 bodies incl. clusters), so they are worse than the Exp A/B-specialised numbers in scatter-field-fixed-budget.md (e.g. Exp B err@5/10/20 .0006/.0014/.0039); not comparable across checkpoints.
- Timing is launch-latency bound (batch 1 N 100 and N 300 are the same ~2.4 ms/tick; batch 48 ~2x).
- Timing run-to-run noise: see rerun section below.
