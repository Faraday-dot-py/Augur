# Polaris GPU reservations

Rule (user, 2026-10-27): showcase videos have equal priority, attention and GPU share with experiments. The account has 2 GPUs (student-pleiades QOS). Each running agent has one GPU reserved; an agent may release its slot to others by editing its row to `released` (with a time), and must claim it back by editing it to `reserved` before submitting again. Never submit a job on a slot that is not yours or released. Update this file in the same turn you change a row.

| Slot | Owner | State | Since |
| --- | --- | --- | --- |
| GPU 1 | headline-error-bars-and-chaos-widening agent (gap items 1, 11; job id pending) | reserved | 2026-11-17 12:58, est 1-2 GPU-h |
| GPU 2 | accuracy-at-1e6-and-baselines agent (gap items 2 accuracy, 7; job id pending) | reserved | 2026-11-17 12:58, est 0.5-2 GPU-h |

Priority registry: `docs/debugging/priority-registry.md` (design in `priority-system.md`). Every slot change and every job submission also updates the registry in the same turn. Check with `python3 scripts/check_priority_registry.py`; it flags unrated running items, starving items (waiting over 12 h), one owner on both GPUs, idle slots and stale rows. A released slot goes to the highest effective rating among waiters.
