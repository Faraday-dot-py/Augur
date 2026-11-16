# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | three-part-control-and-score-agent (claimed by main session 2026-11-16 11:45 for job 10404 resubmit, Pilot-approved; class-project work finished, harness 10444 COMPLETED), then staggered-grid-cost-fix-gpu-timing 11098 and nested-refinement-levels-stage-1-ceiling-probe 11099 (afterany chain, Pilot-approved) | reserved | 2026-11-16 11:45 |
| GPU 2 | voxel-divergence-and-grokking-agent (claimed 2026-11-16 for voxel-divergence-diagnosis-inference-on-checkpoints; prior: released after scatter-field-2d-cost-blowup rerun 11092 ended 11:47), then neighbour-context full run 11101 (Pilot-approved; 11100 FAILED in 38 s, so 11101 started 12:19) | reserved | 2026-11-16 11:56 |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
