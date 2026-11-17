# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | job 11170 (error bars + chaos widening; started 12:59, predicted end 14:45, limit 15:29) then job 11172 (Barnes-Hut rerun, afterany:11170, ~5 min, limit 30 min) | reserved | 2026-11-17 13:15 |
| GPU 2 | none; RESERVED FOR THE PILOT (11171 COMPLETED 13:03; Pilot order 2026-11-17 13:12: all agent/main work on one GPU) | released | 2026-11-17 13:12 |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
