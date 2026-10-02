# Scatter-field vs CentralForceDynamics at a fixed 900 s training budget

Budget: 900 s wall-clock training on one H200 per model (data generation and evaluation excluded), cosine lr and k curriculum scheduled by elapsed time. Eval: 48 scenes, seeds 9000/12000, err@k = mean per-body position error (sim units), 100-step rollouts. Scripts: `scripts/train_scatter_field.py --time-budget 900`, `scripts/train_gravity_dynamics.py --time-budget 900`, `scripts/summarize_budget.py`. Raw log: `scatter-field-overnight-notes.md`.

## Exp A (2 bodies), seed 9000

| run | iters | err@5/10/20 | err@50 | err@100 | dE/E@100 |
|---|---|---|---|---|---|
| CentralForce (steps 30, k<=20) | 4766 | .0001/.0002/.0006 | .002 | .006 | .004 |
| CentralForce (steps 100, k<=50) | - | .0002/.0004/.0013 | .006 | .019 | .014 |
| scatter, kick-drift, k<=50 | 6756 | .0008/.0014/.0037 | .010 | .054 | .065 |
| scatter, kick-drift, k<=20 | 12187 | .0004/.0011/.0033 | .015 | .084 | .112 |
| **scatter, Verlet, k<=20** | 10953 | **.0001/.0003/.0013** | .004 | .015 | .018 |
| scatter, Verlet, k<=50 | 6202 | .0002/.0007/.0018 | .007 | .026 | .023 |

## Exp B (10-100 bodies), seed 9000

| run | iters | err@5/10/20 | err@50 | err@100 | dE/E@100 |
|---|---|---|---|---|---|
| CentralForce (steps 30, k<=20, batch 16) | 4332 | .0006/.0011/.0025 | .015 | .278 | .005 |
| scatter, Verlet, G128, k<=20, batch 16 | 8087 | .0006/.0014/.0039 | .025 | .374 | .006 |

## What made the difference

1. The scatter integrator was first-order: with the exact force it has err@20 .017 (Verlet: .0002). The network had to learn a force that cancels the integrator error, which is slow to learn and looked like a near-field defect (7-15% force error at separations <= 0.5). Fix: velocity Verlet with one force evaluation per step (state carries the previous acceleration). Near-field force error dropped to ~1% at all separations.
2. Training time was the earlier lever (old integrator: err@20 .0049 at 4k it, .0030 at 12k, .0015 at 30k).
3. Not helpful: potential output alone, force-weighted loss, orbit-mix data, batch 128, long windows at short budgets. torch.compile is unavailable on Polaris (no Python.h); per-step CUDA graphs give 1.45x (`scripts/test_graph_step.py`), not integrated.
