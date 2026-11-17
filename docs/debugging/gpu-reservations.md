# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | interior-force-offset all-bodies rerun (agent a8fc66180b239c8a5; job id pending) | reserved | 2026-11-17 12:55, est <1 GPU-h, limit 1.5 h |
| GPU 2 | none (chain 11164-11166 COMPLETED; released 2026-11-17 12:50) | released | 2026-11-17 12:50 |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
