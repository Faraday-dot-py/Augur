# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | nested-refinement-levels-ceiling-probe-harness-fix agent (claims before its first submit; prior chain 11098 and 11099 COMPLETED) | released | 2026-11-17 12:07 |
| GPU 2 | timestep-as-input-model-stage-1 agent, job 11162 (started 12:05, predicted end 13:05, limit 1.5 h); prior voxel-divergence-and-grokking-agent slot released (11111 done, 11101 COMPLETED) | reserved | 2026-11-17 12:05 |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
