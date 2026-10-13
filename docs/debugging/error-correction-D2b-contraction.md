# D2b: recurrent-state contraction retry (2026-10-13, interim)

Code (branch worktree-agent-a226cff35f933b29f, 2ef2d1d; Polaris dir ~/augur-d2b): `scripts/scatter_field.py` opt-in `add_gate(bias, gmax)` (keep = 1 - gmax*sigmoid(conv1x1([x, log1p|field|])), weight 0, bias -6, nonzero gradient at init, keep ~0.9975), `scatter_d2b.py` (builder + `growth_pen`), `scatter_contact_d2b.py` (D5 staged recipe + gate + penalty; `--gate 0 --lam 0` = control path), `scatter_d2b_verify.py`, `scatter_d2b_eval.py` (adds `--inject tick:lam` field-scaling stress, off-grid, err median), `scatter_d2b_regress.py`.

Penalty: from the detached state after the k-tick loss rollout, optionally scale the carried field by lam (log-uniform 3..1000, p 0.5), roll 12 more ticks (reaches tick ~32 at k=20), per tick e = relu(log(R_out+5) - log(R_in+5))^2, R = field rms and max (mean of both). Penalty grads clipped separately (norm 1) before adding to the main grads; gate params get lr x20.

## Step 0 verification (jobs 3547, 3550, frozen C, gate-only)
- Natural field scale: D5ctrl field max 50-73 (rms-max ~28, a constant potential offset) for ticks 5-60; C field max 223 (t5), 2.5e3 (t10), 2.2e4 (t20), 5.5e4 (t40), 1.9e5 (t60).
- Penalty nonzero under the runaway state: 0.02 (first hinge, rho .9), gate gradient nonzero (norm 0.004 to 3e4, heavy-tailed); gate weights moved (|dw| 0.21 at 40 it lr 1e-2; 0.83 at 400 it lr 3e-2).
- Gate-only did NOT remove the runaway on frozen C: field max t60 1.9e5 -> 1.9e5, t100 5.2e5 -> 7.1e5; bias moved more negative (gate opening), i.e. gate alone has no useful direction.

## Full fine-tune attempts (staged recipe, lr 5e-5, k 4->20, 10000 it, seed 4738)
1. Linear-ratio hinge, gate lr x200 (job 3555): penalty 300-1000 from the start; loss diverged at it ~2200-3400 (gn 1e7-1e14), 1230+ rollbacks; killed at it 3900.
2. Log hinge, gate lr x20 (job 3564): penalty 3e-4..1e-3, injected large fields (1e3-3e4) decay to ~100 within 12 ticks; stable to it ~4400 (0 rollbacks), then diverged at k=20 (loss 45 -> 89 -> 264 -> 5e3, gn 1e8-1e11, 2518 rollbacks by it 9154); killed. Not evaluated.
3. Same-code control (`--gate 0 --lam 0`, D2bctrl, job 3573): finished with ~20 rollbacks, loss ~70-90 at it 8000, no divergence. So attempt 2's divergence is attributable to gate/penalty, not the recipe (single seed).
4. Attempt 3 (gate bounded gmax 0.5, otherwise as 2) + evals + harness: queued as chain job 3577 (log ~/augur-d2b-chain-3577.log). RESULTS NOT YET READ (Polaris Duo session dropped; re-auth needed).

Chain evals: C, D5ctrl, D2bctrl, D2b2 on c_id/c_nonconvex/c_count/box15 seeds 9400/9500/9600 with injection stress (field x100, x1000 at tick 25), gravity expB, harness rows vs E (scatter_d2b_regress).

## Predictions (before attempt 3)
- D2b2 trains without divergence (55%). Natural-rollout runaway 0% for control and D2b2 (control already 0%). Injection stress: control runs away at x1000 (|a|>200 in >50% of scenes, 60%); D2b2 recovers (<25%, 35%). err@20 within +-10% of ctrl; dE rows fail for all fine-tunes (90%).
