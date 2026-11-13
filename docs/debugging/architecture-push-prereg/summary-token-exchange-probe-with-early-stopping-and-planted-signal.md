# summary-token-exchange-probe-with-early-stopping-and-planted-signal (pre-registration)

Written 2026-11-13 before submission. Follow-up to job 9914 (summary-token-exchange-between-neighbour-tiles stage 1: delta_net -0.032 pooled on eval seeds 9100/9200/9300, KILL by rule). A fresh reviewer found the probe overfits (64x64 MLP, 30 epochs, no early stopping; the shuffled control also hurt), so it may be blind to a weak real neighbour signal; the oracle gate is trivially satisfied (target near-linear in inputs); fresh seeds were never run. Prereg of the base probe: summary-token-exchange-between-neighbour-tiles.md (definitions of e, f, Delta, Delta_net, oracle unchanged).

## Changes (opt-in flags; defaults unchanged)
- `--early-stop`: MLP refitted with early stopping (max 100 epochs, patience 8) on validation scenes held out of the TRAIN seeds (scene index mod 6 equal 0, 4 of 24 per seed); no eval-seed rows touch the validation loss.
- `--ridge`: closed-form ridge on standardised features; alpha from 1e-2..1e5 (factors of 10) picked on the same validation scenes, then refitted on all train rows.
- `--planted-share 0.05`: planted-signal validity gate. Target becomes e + c*(Wz), z = standardised neighbour summaries (n1, n2), W fixed random (seed 4738), c set on train so the planted term is 5% of target error energy (same W and c on eval). Same pipeline (own, own+neighbour, shuffled-neighbour) per learner; recovered delta = delta_net pooled.

## Gate
Probe is VALID only if the planted delta_net pooled is at least 0.025 (half the planted share) for the early-stopped MLP or ridge. A recovered 0.01 would FAIL. If invalid: no verdict on neighbour information (combined_verdict = NO-VERDICT-probe-failed-planted-signal-gate); the real numbers are still saved.

## Real measurement (only counted if the gate is valid)
Eval seeds 12000/12100/12200 (fresh), MLP train seeds 7100/7200/7300, checkpoint replay seed 4738 (job 4089). ORIGINAL line unchanged: PASS delta_net pooled at least 0.19 and each seed at least 0.10; KILL pooled below 0.06 with oracle f at least 0.90; else INCONCLUSIVE. A value that would still fail PASS: 0.15 pooled, or 0.19 pooled with one seed at 0.08.
- Learner rule (tie-break, fixed before launch): the line is applied to each learner. PASS if the early-stopped MLP passes; KILL only if both the early-stopped MLP and ridge are KILL; anything else INCONCLUSIVE. The fixed MLP of job 9914 is reported for comparison only. No tuning on eval seeds.
- Numerical-reference floor: shuffled-neighbour control gain (delta_net already net of it) and the planted-signal gate.

## Prediction (before the run)
delta_net 0.0-0.05 pooled, P(PASS) under 10%, KILL most likely. Halo exchange of the recurrent field already carries much of the neighbour information. Gate expected to pass (a linear 5% signal is easy for ridge); if it does not, the probe is the problem.

## Run
Script scripts/neighbour_tile_force_error_predictability_probe.py; job scripts/polaris_summary-token-exchange-probe-with-early-stopping-and-planted-signal.sh (slot 2, 1 GPU, est 0.2 GPU-h, limit 1 h, requeue, SIGTERM trap); output results/summary_token_exchange_probe_with_early_stopping_and_planted_signal/summary_token_exchange_probe_with_early_stopping_and_planted_signal.json.
