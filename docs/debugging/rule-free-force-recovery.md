# Rule-free force recovery (in progress, uncommitted)

Pre-registration and floor run: docs/debugging/experiment-log.md, 2026-11-08 rule-free-force-recovery entries. Not yet scored against the model.

## Status
- Exact fp64 floor run done: results/rule_free_force_exact.json (see log entry). Pipeline recovers 7 of 8 true laws exactly; yuk128 comes back as a plain softened power (exponent 2.04), range unidentifiable.
- Job 7190 (slot 2 chain, afterany:7187): curves for 8 laws x 2 checkpoints. Estimate ~40 min, ~0.7 GPU-h. Log: ~/augur-rule-free-force-curve-extraction-7190.log on Polaris home. Output: ~/augur_nfr/results/rule_free_force_curves/ (16 npz).

## Resume steps
1. polaris_job_status 7190 == COMPLETED and log ends EXIT 0 (an INCOMPLETE OUTPUT line means rerun: `cd ~/augur_nfr; sbatch scripts/polaris_rule-free-force-curve-extraction.sh` after editing the dependency line; per-cell files are skipped). Check slot file first.
2. polaris_download each of the 16 npz into ~/polaris-mcp-files/rule_free_force_curves/ then copy to results/rule_free_force_curves/.
3. PYTHONPATH=. python3 scripts/rule_free_force_discovery.py model --curves results/rule_free_force_curves --out results/rule_free_force_model.json   (about 10-15 min, local CPU, 8 workers)
4. PYTHONPATH=. python3 scripts/score_rule_free_force_recovery.py --curves results/rule_free_force_curves --exact results/rule_free_force_exact.json --model results/rule_free_force_model.json
5. Fresh reviewer subagent recomputes the headline numbers; log verdict in experiment-log; fill this writeup (level three design only if levels one and two are clear).
