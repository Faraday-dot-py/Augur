# SAE dose-response with a larger random pool (curiosity, no pass/kill)

Label: curiosity (user-approved 2026-11-08). Observations only; no pass or kill verdict.

## Pre-stated read-out rule (written 2026-11-08 before launch, not to be loosened)

A top-k law-aligned set is "special" at one (run, k) if its median force error is above the 95th percentile of the 200 frequency-matched random draws (percentile rank 95 or higher) in at least 6 of the 8 laws. "Any feature removal hurts" is read if the top-k set sits inside the random range (rank below 95) in at least 6 of 8 laws. Tie-break (5 of 8 or 4 of 8): the mean percentile rank of top-k across the 8 laws, 95 or higher is "special", otherwise not. Percentile rank = (random draws below top-k plus half the ties) / draws.

Random-pool rule: per slot (one slot per top feature) the candidate pool is the 50 live non-top features nearest in log activation frequency to that slot's top feature, or all live non-top features if fewer than 50 exist (a 64-feature dictionary has about 40 live). Within a draw members are sampled without replacement. 200 draws per k, k in 1, 2, 5, 10, 20 (k capped at half the live count). Seed 4738. Force error: median over 8 scenes per law against exact fp64 force (same scenes as earlier dose-response runs).

Outlier flag: a random draw is an outlier if its error exceeds max(Q3 + 3 IQR of the random draws, 1.2 x the SAE-reconstruction error).

(Results, job id, reviewer verdict: appended below after the run.)

## Run record

Job 7183 (slot 2, no dependency because tail was none), submitted 2026-11-08 17:08, first law of the first run done after about 2 minutes (about 16 min per dictionary run, estimated 2 to 2.5 GPU-h for 8 runs, limit 6 h). Script scripts/sae_ablation_random_pool_dose_response.py, job script scripts/polaris_sae_random_pool_dose_response.sh (both uncommitted; imports sae_ablation.py, sae_ablation_dose_response.py, sae_interp.py, all committed). Log: ~/augur-sae-random-pool-dose-response-7183.log on Polaris (download with polaris_download). Outputs: ~/augur_nfr/results/sae_random_pool_dose_response_<tag>_<key>.json (8 files: 4738 dec0 mesh_256/mesh_512/mesh_128/mesh_64, 4739 dec0 mesh_64, 4739 dec2 mesh_256/mesh_128, 4738 dec0 heldout16 mesh_256).
Resume: a run's JSON has "complete": true when finished; resubmitting the same job script (slot 2 chain, afterany on the current tail) skips completed runs. Then download the JSONs, summarise, spawn the fresh reviewer, fill the results here and add the curiosity entry to docs/debugging/experiment-log.md.
