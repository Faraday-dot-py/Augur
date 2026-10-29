# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | experiments | reserved | 2026-10-29 13:05 (tiered-drift-mechanism-agent, diagnostics job) |
| GPU 2 | showcase videos | released | 2026-10-29 13:54 (neuron-graph-smoke-agent done, nothing queued) |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
