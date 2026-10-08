# Scatter joint from-scratch training

Branch `scatter-joint-scratch` (base `scatter-integration-retrain` 024dd0a). Question: is "the scatter field is good at one thing at a time" an architecture limit, or an artifact of short-unroll fine-tuning from existing checkpoints? Prior: 7 single fixes fail the 15% gate; combined fine-tune int_full1 passes accuracy but breaks 1000-tick energy rows (nr100 dE .017->.287, bh300 dE +35%, vir_gap .037->.271), regresses largen cold_1e4 (1.77->2.62) and plummer a=.25 (3.3->5.2); a plain fine-tune control fails the same dE rows.

## Design
- Model: variant `ms_kp_pot_v_g128_cu_eps_mnorm` (opt-in input flags only, no new modules), RANDOM init (no `--init-ckpt`). Same variant for joint and control so the only difference is the data mix.
- Joint (`--fix all`): E-style replay 40% (70% h 0.5, 15% each G256/G64), rest masses 0.1-10 (25%), 0.01-100 (12%), eps {.1,.2,.35,1,2} (6% each), dense/Plummer (20%), large-N 1000-6000 (13%). Control (`--fix ctrl`): E-style scenes only (box N 10-200 + 50% cold clusters, unit mass, h .5, eps .5).
- Long-horizon remedy, identical in both runs: scenes of 100 ticks, unroll k 4 -> 40 (large-N groups capped 12/8), plus soft dE/dL penalty (`--long-windows 2 --long-k 20 --lam-e/--lam-l 0.3`, truncated BPTT past the supervised window, groups with N<=400, on from 25% of the schedule). No hard-coded conservative force. Penalty weight 0.3 chosen from the earlier note (lam 3 over-traded accuracy, 0.3-1 suggested); not swept.
- Optimiser: Adam lr 5e-4 cosine to 5%, warmup 200, per-group EMA loss norm + spike skip (needed in the integration run), batch 16, seed 4738, 1500 base scenes. Budget 12000 s each (Exp E was B-pretrain + 9000 s). Full-state checkpoint (model+Adam+iter) every 50 it, plus `.it<N>` snapshots every 5000; resumable.
- Eval: full harness (`scatter_regress.py run --battery`, held-out seeds 9100/9200/9300/4738, 1000-tick conservation rows), `scatter_ood.py` all axes, scaling r46/n46/n46big; compared against baseline_v2, control, int_full1 via `scripts/scatter_joint_table.py`.

## Predictions (before running)
Pilot (600 s, joint, ~3.5k iters): pipeline runs, no non-finite; accuracy far from baseline (err@20 several x baseline) because of the short budget.
Full joint vs control vs int_full1:
- Control from scratch: err@5/10/20 within 0-40% of baseline_v2 (cost of random init vs B-pretrain + longer E schedule); dE nr100@1000 within ~3x baseline (.017) thanks to penalty and long unroll.
- Joint: P(passes 1000-tick energy rows within gate vs baseline_v2) ~0.3; P(accuracy rows within 15% of baseline_v2) ~0.2 (from scratch with 12000 s vs E's total budget); mass/eps/cu rows at least as good as int_full1 (P 0.6); largen cold_1e4 and plummer a=.25 no worse than baseline (P 0.4).
- Verdict rule: if joint matches control on energy and keeps the mass/eps/cu gains with no row worse than baseline beyond tolerance -> artifact of fine-tuning. If energy rows or largen/plummer still trade against mass/eps/cu (joint worse than control on energy rows or on E-style accuracy) -> architecture/capacity limit at this size. Mixed -> report per row.

## Results (2026-10-08)
Jobs (Polaris, private copy `~/augur_joint`): 3419 pilot (600 s, joint, tests 230 pass, pipeline OK, 2242 it, err@20 2-5x baseline as predicted), 3421 `joint_full` (`--fix all`, 12000 s, 45164 it, 2584 skipped spike steps), 3422 `ctrl_full` (`--fix ctrl`, 12001 s, 45762 it, 59 skipped), 3423 1000-tick snapshots. Wall 3.60 + 3.56 + 0.22 + 0.01 = 7.4 GPU-h. Checkpoints: `~/augur_joint/checkpoints/scatter_joint/{joint_full,ctrl_full}.pt` (full state, `.it<N>` snapshots every 5000). Raw: `results/scatter_joint/`, tables `results/scatter_joint/tables.md` (`scripts/scatter_joint_table.py`), snapshots `snaps_{nr100,bh300}.npz`, rendered `videos/scatter_joint_{bh300,nr100}{.mp4,_grid.png,_series.png}`.

Gate vs baseline_v2 (compare): joint FAIL 18 rows (all err@5/10/20 +31..+72%, bh300 dE/dL@100, nr100 and bh300 dE@1000, bh300 bound/vir/outgrid gaps). Control FAIL 6 rows (bh300 err@5/10/20 +18..+22%, nr100 dE@1000 .017->.074, bh300 dE@1000 .48->.67). int_full1 (earlier): accuracy PASS, nr100 dE@1000 .287, bh300 dE@1000 .65, vir_gap .271.

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| two_body err@5/20 | .00105/.0091 | .00100/.0081 | .00156/.0119 | .00108/.0083 |
| expB err@5/20/100 | .0030/.0239/1.42 | .0033/.0262/1.53 | .0050/.0394/1.63 | .0032/.0270/1.54 |
| bh300 err@5/20/100 | .0080/.0622/4.73 | .0095/.0759/4.97 | .0128/.107/6.95 | .0091/.0691/5.08 |
| nr100 dE@1000 | .017 | .074 | .167 | .287 |
| bh300 dE@1000 | .48 | .67 | 2.25 | .65 |
| bh300 vir_gap@1000 | .037 | .010 | 4.06 | .271 |
| bh300 bound@1000 (truth .88) | .75 | .72 | .17 | .74 |
| mass q100 err@2 | 7.0 | .70 | 1.85 | .48 |
| eps2 err@2 | .73 | .79 | .026 | .016 |
| cu G64 / G256 err@20 | 1.12 / 1.93 | 2.31 / 2.04 | .32 / .17 | .12 / .97 |
| dense sigma1 / sigma2 err@2 | 1.15 / .40 | 1.09 / .40 | 5.12 / 2.93 | 1.03 / .35 |
| plummer a.25 err@2 | 3.33 | 3.58 | 8.27 | 5.22 |
| largen cold_1e4 err@5 | 1.77 | 2.02 | 2.44 | 2.62 |
| tick_ms two_body / expB / bh300 | 4.72/4.95/2.49 | 4.81/5.03/2.60 | 4.80/5.03/2.61 | 4.81/5.03/2.56 |

Predictions vs result: control err@5/10/20 within 0-40% of baseline (HIT except bh300, +19-22%); control nr100 dE within 3x baseline (MISS, 4.4x); joint passes energy rows (MISS, P was .3: bh300 virial 54 vs truth .93, bound .17, outgrid .83); joint accuracy within 15% (MISS, P .2); mass/eps/cu at least as good as int_full1 (MISS: eps2 and cu G256 yes, mass q100 1.85 vs .48, cu G64 .32 vs .12); largen/plummer no worse than baseline (MISS).
