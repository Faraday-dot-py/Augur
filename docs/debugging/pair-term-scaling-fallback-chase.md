# Pair-term scaling fallback chase (curiosity diagnosis, 2026-11-08)

Status: step 1 (stored data and code reading) done; step 2 job 7187 submitted, results pending. Curiosity-labelled: no pass or kill verdict. Pre-registration: docs/debugging/experiment-log.md, section "2026-11-08 collapsed-pair-term-fallback-chase: PRE-REGISTRATION". Nothing committed.

## 1. What the code does at N = 3e6 (scripts/scatter_field.py, pp_acc_adapt)

- The adaptive pair term builds cell lists at levels l = 0..8 (cell size pp/2^l, pp = 2, so the finest cell is 2/256 = 0.0078). A body is resolved at the first level where the largest cell count in its window is at most 64 at level 0 and at most 32 at levels 1-8 (the candidate ladder is 1, 2, 4 ... 64 at level 0 and stops at 32 after that, line 350; found by the reviewer) and, for l >= 1, its 16th neighbour lies inside the window's guaranteed radius (lines 365-368).
- A body with more than 32 bodies in a window cell at every finer level, or one failing the radius test at the last level, stays unresolved. After the loop (lines 377-389) every unresolved body is sent to a chunked all-pairs loop against all n bodies: chunk size Q = max(1, budget // n) = max(1, 2^25 // 3e6) = 11 bodies, so the work is (unresolved bodies) x n pair evaluations. This is O(N^2) when the unresolved count grows with N.
- At 1e6: Q = 33, 2,380 unresolved bodies, 73 chunks. At 3e6: Q = 11, 1,199,131 unresolved bodies, about 109,000 chunks.

## 2. What the stored data show (job 6240, augur_probe/results/collapsed_pair_term_scaling_probe/probe.json, downloaded 2026-11-08 to ~/polaris-mcp-files/pairterm/)

adapt_stats was saved per cell (the earlier review note "not read" is resolved). Counters are from the last timed call.

| N | cloud | seconds per call | ns per body | bodies sent to all-pairs | peak GB |
| --- | --- | --- | --- | --- | --- |
| 3e5 | collapsed | 0.0485 | 162 | 0 | 1.5 |
| 1e6 | collapsed | 0.282 | 282 | 2,380 | 2.4 |
| 3e6 | collapsed | 222.4 | 74,145 | 1,199,131 (40%) | 5.8 |
| 1e6 | reference | 0.136 | 136 | 2 | 2.4 |
| 3e6 | reference | 0.397 | 132 | 0 | 5.8 |

- Fallback count rises about 500x for a 3x increase in N (3e6 vs 1e6). Time per call rises 790x.
- Excess time over the reference cloud divided by (unresolved bodies x N): 0.146 s / 2.38e9 = 6.1e-11 s at 1e6 and 222.0 s / 3.6e12 = 6.2e-11 s at 3e6. The same constant at both sizes means the data fit the all-pairs cost (fallback x N) with no other term needed.
- Memory is not the limit (5.8 GB peak). Warm-up: the timing helper runs one untimed call first, so the 3e6 figure is the second call, not a cold start; a single timed call is still one sample.
- Not recorded in the stored data: why bodies reach the fallback (more than 32 in a cell vs radius test failing at level 8), per-stage times, steady-state spread. The chase job adds these.

## 3. Step 2: instrumented rerun (job 7187)

- Instrumentation: scripts/scatter_field.py gated by model.adapt_profile (counts per level of bodies over the cap (64 at level 0, 32 after), seconds per level, fallback bodies, chunks, seconds). Off by default; output compared against the plain call (max abs diff) inside the job. Script scripts/collapsed_pair_term_fallback_chase.py, job script scripts/polaris_collapsed-pair-term-fallback-chase.sh.
- Cells: N = 1e6, 1.5e6, 2e6, 3e6, collapsed and reference clouds, 1 warm-up + 3 timed + 1 profiled call each, seed 4738.
- Job id 7187, slot 2, `#SBATCH --dependency=afterany:7183`, estimate under 40 min (collapsed 3e6 alone about 19 min), limit 1 h 30 min, 32 GB host memory, peak GPU memory expected 6-8 GB. Rating 2, nice 1000.
- Log: Polaris home `augur-collapsed-pair-term-fallback-chase-7187.log` (use polaris_download; polaris_job_logs reads the wrong path). Results: `augur_probe/results/collapsed_pair_term_fallback_chase/chase.json` and `chase.log`. Download to ~/polaris-mcp-files/ first.
- Resume steps: (1) check sacct for 7187 (exit 0, ALLDONE in log); (2) download chase.json; (3) evaluate the interpretation rule in the pre-registration (fallback at 3e6 >= 100x at 1e6; fallback_s share of the profiled call >= 95%; fallback_s / (fallback x N) within +-25% of 6.1e-11; steady within 10% of warm-up; over_cap at level 8 vs fallback for trigger label cap / cover / mixed; max abs diff 0); (4) fill the Results section below and the experiment-log entry; (5) update priority-registry row 7187 and gpu-slot-chains.md; (6) fresh reviewer on chase.json plus script.
- Incident: the first submit (7185) passed the dependency through the MCP tool's sbatch_args, which did not apply it; it started while two other GPU jobs ran and was cancelled after 20 s (three GPU jobs overlapped for about 20 s). Resubmitted as 7187 with the dependency as an SBATCH line in the script and verified PENDING (Dependency) before moving on. Other agents using sbatch_args for dependencies should check state after submit.

## 4. Results (pending job 7187)

To be filled.

## 5. Ranked fix proposals (not implemented; based on the code reading, to be re-ranked when the trigger is known)

Constraints checked: model must learn dynamics (none of these touches the learned net or adds a force); stay pp = 2 and O(N), no pairwise pp = 8 (feedback_no_pairwise_pp8_keep_onlogn); no hard-coded conservative force (feedback_no_hardcoded_conservative_forces); soft priors would be allowed but none is needed.

1. Deepen the exact multilevel search for crowded cells (best): extend the candidate ladder past 32 (64, 128, 256) and raise lmax from 8 to 10 or 11, so dense-core bodies resolve at a finer level or a wider list instead of falling through. Output stays the exact top-16, so numbers match the all-pairs result up to tie order; this is a performance fix only. Expected cost: a few lines in pp_acc_adapt, one extra sort and search per added level over the unresolved bodies only (untested; the chase job's per-level seconds will give the real figure); chunk size Q = budget / (25 x m) stays in the thousands. Expected effect: fallback near 0 at 3e6, time back to about 0.4-1 s per call (about 130-300 ns per body). Risks: cell index size grows (C^2 at finest level about 1e9, int64 keys still fine), float32 positions at cell sizes near 0.002 need a check, and a true point-mass cloud still defeats any finite ladder. Validation: exact-match test against the all-pairs result on a 2e4 collapsed cloud, then the chase job.
2. Hard bound on fallback work: keep the exact path, but if (unresolved x N) exceeds a set budget, resolve the rest with a cap-neighbour rule (candidates limited to the nearest K in sorted cell order, approximate top-16). Guarantees O(N) in every cloud, including point-mass collapse. Changes numerics for those bodies (approximate neighbours, model was trained on exact top-16), so it needs an accuracy gate before use. Cost: moderate (one new branch plus a test). Lower rank because it alters behaviour.
3. Hierarchical exact neighbour search (Morton-ordered list or adaptive equal-occupancy cells) replacing the fixed-size cell ladder: robust to any density, O(N log N). Cost: largest (days, new kernel, new tests). Rank third; worth it only if fix 1 leaves a pathological case.
4. Cheap guard, not a fix: when unresolved x N is huge, log a warning (or abort) instead of silently taking minutes, and report adapt_stats in every scaling probe. A larger chunk budget only shrinks the constant by a small factor and the cost stays quadratic.

## 6. Independent review

Review of step 1 (fresh reviewer, no hypothesis primed, read probe.json, script, scatter_field.py diff, job script): ran. Agreed: all recomputed numbers (fallback 0 / 2,380 / 1,199,131; excess constant 6.14e-11 and 6.17e-11 s), the all-pairs cost path, instrumentation is numerics-neutral. Found: (a) the candidate ladder stops at 32 for levels 1-8, so the over-cap counter was testing 64; fixed before launch (threshold now 64 at level 0, 32 after), logged as an amendment in the pre-registration; (b) the rerun draws different scenes than job 6240 (same generator and seed, different draw order), so fallback counts are comparable in scale, not identical; (c) the first cell warm-up includes CUDA start-up, so the warm-up-artefact test applies to later cells only; (d) chase.log is appended on rerun, a stale DONE line could match (job 7185 left a partial log, check the final lines). Review of the job 7187 results: not run yet.

## 7. Adaptive-depth fix feasibility (pair-term-adaptive-depth-fix, 2026-11-08)

Option added to scripts/scatter_field.py, off by default: deeper cell levels (up to 14), and longer candidate ladders (to 64 or 128). Flags were renamed after the first smoke run (now adaptdeep, adaptwide64, adaptwide128); the smoke numbers below are from the renamed flags. Default output is bit-identical to git HEAD on CPU at N 3e4 (torch.equal True).

CPU emulation of the 3e6 collapsed density (N 3e4, core scale 0.025, random nonzero pair weights, seed 4738; default fallback 46.1% vs 40% on the real 3e6 cell):
- default: fallback 13,837 bodies (46.1%), 4.32 s per call.
- deeper levels only: fallback 10 bodies (0.033%), 0.62 s. The leftover tail is outskirts bodies with steep density gradients that need a longer ladder.
- deeper levels plus ladder 64: fallback 0, about 0.95 s in the earlier all-variants run.
- Every variant returns the exact top-16: max abs error vs an independent brute-force top-16 is 4.8e-07 (float round-off) for default, same order for the others; so accuracy equals the current method. Raw: /tmp scratchpad out/cpu_30000_all.json.

Verdict so far: feasible and exact on CPU emulation. GPU time at the real 3e6 against the 0.794 s (2.0x of 0.397 s) limit is NOT measured. Pre-registered pass/kill is not yet evaluated (time and real-density fallback fraction need the GPU job).

GPU job: scripts/polaris_pair-term-adaptive-depth-fix.sh (dependency afterany:7193 inside the script, 1:30 limit, three cells: collapsed 1e6 with head bit-identity, reference 3e6, collapsed 3e6). Files are uploaded to ~/augur_probe_deep on Polaris. Submit FAILED: QOSMaxSubmitJobPerUserLimit (account job submit limit). Not worked around. Not on the slot chain. Resume: when the submit limit clears, re-read gpu-slot-chains.md, set the dependency line to the then-current slot-2 tail, and submit script_path pairterm_deep/polaris_pair-term-adaptive-depth-fix.sh (the uploads persist). Results land in ~/augur_probe_deep/results/pair_term_adaptive_depth_fix/*.json.

Review of section 7 (fresh reviewer, no hypothesis primed): ran. Agreed: default path unchanged and bit-identical to HEAD (rerun printed True); flags only add levels or longer ladders and keep the guaranteed-radius exactness test; no new all-pairs path; fallback 13,837 vs 10 and max error 4.8e-7 recomputed. Notes: the longer ladder at 64 alone leaves 3,556 bodies (11.9%), so deeper levels are required; at 14 levels cell indices stay exact in float32 (below 2^24) but edge rounding was not stress-tested; the max-error column is a weak discriminator (identical across variants), the diff to default is the better evidence. scripts/scatter_field_head.py is only a staged copy of git HEAD scatter_field.py (not in the repo); it is already uploaded to Polaris. The job script has not run.
