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
