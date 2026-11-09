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

## Results (job 7183 COMPLETED exit 0:0, 2 h 25 min 33 s; all 8 JSONs "complete": true, every STEP exit 0)

Numbers: results/summary_sae_random_pool_dose_response.txt (script scripts/summarise_sae_random_pool_dose_response.py recomputes ranks from raw per-arm errors). Curiosity: the readings below apply the pre-stated rule and are not a pass or kill verdict. "recon" = error with the SAE reconstruction and nothing removed; ratios are averaged over the 8 laws.

Reading by run and k (S = special by the rule, H = any-removal-hurts, T- = not special by tie-break). Mean percentile rank of top-k in brackets.

| run (live features) | k=1 | k=2 | k=5 | k=10 | k=20 |
| --- | --- | --- | --- | --- | --- |
| 4738 dec0 dict 128 (84) | H (53) | H (48) | S (100) | S (100) | S (100) |
| 4738 dec0 dict 256 (115) | S (93) | S (100) | S (100) | S (100) | S (100) |
| 4738 dec0 dict 512 original (184) | H (45) | S (100) | S (100) | S (100) | S (100) |
| 4738 dec0 dict 64 (58) | H (50) | H (48) | H (89) | H (75) | S (100) |
| 4738 dec0 dict 256 held-out ranking (88) | H (44) | H (37) | H (30) | H (82) | S (100) |
| 4739 dec0 dict 64 (43) | H (72) | H (84) | S (99) | S (100) | S (100) |
| 4739 dec2 dict 128 (50) | H (29) | H (48) | T- (58) | T- (56) | T- (53) |
| 4739 dec2 dict 256 (65) | S (92) | S (89) | T- (85) | S (95) | S (88) |

Observations.
- Top-k removal is above the random 95th percentile from k=5 (dict 128, 256, 512, 4739 dec0 dict 64) or k=2 (256, 512) in dec0. At k=1 it is special only where the top feature is itself a frequency-0.87 feature (dict 256 feature 229; the 4739 dec2 dict 256 top feature has frequency 0.75).
- The k=1 "any-removal-hurts" cells are not a sign that random features are harmful in general: random median is 1.00x recon, but a few random single features are very damaging (p95 6x to 20x recon in dec0). Examples: dict 512 feature 448 (frequency 0.87), mean pow2 error 2.52, pow1.5 1.41, yuk8 3.19, about the 0.05 to 2.5 jump logged earlier for the "mesh_512 top feature" (earlier ranking used 12 activation scenes; here the top-1 feature is 322, frequency 0.125, only 1.16x recon). Dict 128 feature 118 (freq 0.87): yuk8 2.21. Dict 64 features 49, 53, 39 (freq 0.85 to 0.87), dict 256 held-out feature 54 (freq 0.83) and 249 (0.82), 4739 dec0 dict 64 features 5 and 60 (0.87 to 0.88).
- Every strongly damaging random feature found is a high-frequency (0.82 to 0.88) feature; the lj law is the exception (its worst random features are low-frequency, 0.12 to 0.14, damage 0.28 to 0.45). Because the top-1 feature in most dec0 dictionaries has frequency 0.12, its matched pool contains none of these, which explains the k=1 and k=2 ranks near 50.
- k=20 "special" in the small dictionaries is partly by construction: removing the 20 top features also removes the load-bearing high-frequency features, and the remaining random pool is nearly exhausted (23 to 38 non-top live features, random median and p95 both 1.00x recon in dict 64). Do not read k=20 in dict 64 as evidence beyond that.
- 4739 dec2: dict 128 is not special at any k (top within 0.98x to 1.09x recon; random p95 within 1.01x to 1.09x). Dict 256 is special at k=1, 2, 10, 20 by a 6 to 7 of 8 laws count but the effect is small (top 1.27x to 1.47x recon vs random p95 about 1.01x to 1.04x). dec2 differs from dec0: no random feature has a large effect.
- Outlier random draws are saved per law and k (summary[law][k].outlier_draws, worst_random_features); members of every draw are in draw_members.

Caveats. Force-error medians use 8 scenes per law (same scenes as earlier runs); the top-feature ranking and frequencies use all 24 activation scenes including SAE training scenes (except the held-out-ranked runs); 4739 dec0 dict 64 pool at k=20 is 23 features for 20 slots; ranks tie heavily when many random draws equal recon. No second SAE seed.

## Independent review (fresh reviewer, ran 2026-11-08)

Recomputed from raw per-arm errors (not the saved summary blocks): rule counts and mean ranks reproduce for the three checked cells (dict 256 k=2: 8 of 8, rank 100; dict 128 k=1: 1 of 8, rank 52.9; 4739 dec2 dict 128 k=5: 3 of 8, 57.5, not special by tie-break). Feature 448 (frequency 0.868) pow2 mean 2.520 reproduces. Draws: 200 per k in all 8 files, no duplicate members inside a draw, no top or dead members, rand keys 0-199 only. Code check: rank counts ties as half, baseline is the SAE hook with nothing removed, seed 4738 everywhere (the 4739 files use the other checkpoint with the same seed).
Caveats from the reviewer: the dict 512 top-1 mean ratio of 1.16x is driven by lj (2.31x); on the other 7 laws top-1 is 1.00x recon. Many random draws tie exactly with recon (features silent in the force scenes), producing mass ties and a rank near 45 even when top does nothing. The 50-nearest pool is loosely frequency-matched for a top feature at frequency 0.87. Medians use 8 scenes. Review used one inline python3 call to list JSON keys (reviewer's own rule slip; output only key names).
