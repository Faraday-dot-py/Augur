# Priority system for jobs and agents

Scope: 2 Polaris GPUs account-wide (QOS = Slurm quality-of-service class, here `student-pleiades`), about 5 agents running at once, one human-in-the-loop supervisor (the main Claude session). Everything is a text file; a person or an agent can edit it without tools. Registry: `docs/debugging/priority-registry.md`. Checker: `scripts/check_priority_registry.py`.

## Research summary (from background knowledge, not web-verified in this pass; nothing fetched)

- Slurm multifactor priority = weighted sum of job age, fair-share (past usage by account/user), job size, partition, QOS and `nice`. At 1 account with 2 GPUs the fair-share and size terms carry no information, and our QOS has a single priority value (30). So Slurm can only order our own jobs through `--nice` (higher = later), `--begin`, `--dependency` and the per-user cap (`QOSMaxGRESPerUser`). Slurm priority is therefore the enforcement layer; the decision of what goes first is made here.
- MoSCoW (must / should / could / won't) is the right granularity for a human reading a table; numeric scores invite false precision. Weighted shortest job first (WSJF = cost of delay divided by job duration) and value-versus-effort say: short, high-value, unblocking work goes first. We keep that as three yes/no questions, not a formula with weights.
- Aging (priority rises with waiting time) prevents starvation; Slurm does the same with its age factor. We cap aging so it never beats a user deadline.
- Preemption (Slurm `PreemptMode=REQUEUE`, here `GraceTime=0`, no QOS preempt lists) is configured but never triggered by our QOS; requeue-etiquette: jobs must be restartable (`--requeue`, SIGTERM trap, full-state checkpoint) or must not be preempted at all.
- Reservations: one GPU per running agent (`gpu-reservations.md`) is a static partition, simple and fair at this scale.

## Rating scale (one number, 1 to 5, with a name)

| Rating | Name | Meaning |
| --- | --- | --- |
| 5 | urgent | the user asked for it with a deadline inside 24 h, or it blocks the user right now |
| 4 | high | high scientific value and it unblocks other work, or user-requested without a hard deadline |
| 3 | normal | planned work with clear pass/kill criteria (default) |
| 2 | low | exploratory, repeat of a known result at new settings, or large cost for modest value |
| 1 | background | fill-in work for otherwise idle GPUs; always the first to wait or be requeued |

Rating = 3, then adjust by the five questions below, clamped to 1-5. Write the adjustments in the rationale column.

| Question | Adjustment |
| --- | --- |
| Deadline: did the user name a time that falls within 24 h? | +2 (within 7 days: +1) |
| Value: does a pass/kill answer change a decision or fix a user-visible defect? | +1 if yes, -1 if it only repeats or tunes a known result ("low-value rerun") |
| Unblocks: do other jobs or agents wait on its output? | +1 |
| Cost: expected GPU-hours above 4 | -1 (above 12: -2) |
| Risk: untested script, no checkpoint/resume, or no pass/kill criterion written | -1 (fix the cause before launch instead of paying the penalty) |

A user statement of priority overrides the arithmetic: record it as rating plus `pinned` in the rationale; only the user unpins.

## Who assigns, when

- The supervisor (main session) assigns the rating when it dispatches an agent or approves a job, and writes the row. The agent may propose a different rating for its own jobs in its report; the supervisor decides.
- Each agent rates its own new Polaris jobs when it submits them (using the table), inheriting at most its own rating, and adds a row. It never rates itself above the supervisor's rating for it.
- Showcase videos and experiments use the same scale and the same questions; there is no class bonus either way (user rule 2026-10-27: equal priority, attention and GPU share).

## Ordering, tie-breaking, aging

1. Higher effective rating first. Effective rating = rating + aging bonus.
2. Aging: an item that has been pending or idle (waiting for a GPU, not blocked on its own dependency) for 12 h gets +1 effective rating, for 24 h +2, but never above 4 (only a user deadline makes 5). The checker flags any item waiting over 12 h as `STARVING`.
3. Ties: earlier `queued_since` first; then shorter expected runtime first (shortest job first); then the class that has had fewer GPU-hours in the last 24 h (keeps the equal-share rule honest).

## Equal share and reservations

- Default: GPU 1 for experiments, GPU 2 for showcase videos (table in `gpu-reservations.md`). Neither class takes the other's slot while the owner has a runnable job.
- A released slot goes to the highest effective rating among waiters, not to the class that asked first. An agent claiming its slot back edits the row to `reserved` before submitting; a running job on a released slot is never killed for the claim, the claim waits for it to finish (or the job's next checkpoint, if it is requeueable and the owner's rating is higher by 2 or more).
- One owner holding both GPUs at the same time is a defect unless the other class has nothing runnable; the checker flags it.
- Idle reserved slot: an owner with nothing to submit for 2 h releases the slot (checker prints `IDLE_SLOT` hint when a slot has no running job and the owner has no pending job).

## Preemption and requeue rules

- Default is no preemption: running jobs finish.
- Exception: a rating 5 item waits behind a rating 1 or 2 item for more than 30 min. The supervisor may `scancel` the lower job only if it is requeueable (`#SBATCH --requeue`, SIGTERM trap, checkpoint and `--resume`, as required for any job over 1 h) and then resubmits it with a later `--begin`. A job that is not requeueable is never cancelled for priority; the waiter waits.
- Never cancel another agent's job without telling its owner in the registry rationale and the supervisor log the same turn.
- Requeued jobs keep their original `queued_since`, so aging continues.

## Mapping to Slurm (flags actually used in this project, jobs 4053-4056 and earlier)

| Rating | `--nice` | Other |
| --- | --- | --- |
| 5 urgent | 0 | submit first; no `--begin` |
| 4 high | 0 | |
| 3 normal | 100 | default |
| 2 low | 1000 | |
| 1 background | 10000 (as in jobs 4053-4056) | `--begin=<off-hours time>` allowed; `--requeue` required |

- `--nice` only orders our own pending jobs (the QOS has one priority value); it does not preempt. `--dependency=afterok:<id>` expresses "unblocks"; `--begin` delays a job so it does not take a GPU someone else is waiting for; `--requeue` plus `--signal=B:TERM@120` makes preemption or node failure survivable. The QOS cap (2 GPUs, 4 running, 8 submitted) cannot be changed from here.
- Pending reason `QOSMaxGRESPerUser` just means both GPUs are in use by our own jobs; do not resubmit, do not poll in a tight loop (use one wait tool call with a long timeout), and do not leave the turn.
- Flag support was not re-verified with `sbatch --help` in this pass because the Polaris tools exposed to this session have no raw command runner; the four flags above are all present in accepted jobs 4053-4056. Do not submit test jobs to check.
- Auth: if Polaris reports disconnected, `polaris_start` at most once; never override the 3-fail/900 s lockout.

## How agents and the supervisor use it

- Dispatch: the supervisor adds the agent row (rating, rationale, owner, GPU slot) before sending the prompt; the prompt carries the "priority and GPU slot" block from `research-agent-prompt.md` / `implementation-agent-prompt.md`.
- Submit: before `sbatch`, the agent checks `gpu-reservations.md` and the registry, adds a row for the job, chooses `--nice` from the table, then submits. On job end it sets the row to `done` (rows older than 24 h in a terminal state are deleted at the next edit, the history lives in the experiment log and supervisor log).
- Supervisor sweep every status update, and at least every 6 h overnight: run `python3 scripts/check_priority_registry.py`, fix every `ERROR`, act on `STARVING`, `UNRATED` and `BOTH_GPUS` lines, update the Priority column of the Live status table in `docs/theory/supervisor-log.md` from the registry.
- Failure modes this targets: jobs pending behind `QOSMaxGRESPerUser` while a lower-value job runs (rating makes the order explicit); agents polling or ending turns while pending (rule above); idle agents or slots (IDLE_SLOT); one agent holding both GPUs; low-value reruns (-1 value question, check "already tried" list in `experiment-log.md`); unlogged results (an item cannot go `done` in the registry until its experiment-log entry exists, the row's `log` column names it).

## Registry format

One markdown table, one row per running or pending job or agent. Columns, in order: `id` (job number or agent name), `kind` (`job` or `agent`), `name`, `owner` (agent name, `supervisor`, or `unknown`), `rating` (1-5), `rationale` (adjustments applied, short), `gpu_slot` (`1`, `2`, `none`, or `unknown`), `state` (`running`, `pending`, `idle`, `released`, `done`), `queued_since` (`YYYY-MM-DD HH:MM`, PDT), `updated` (same format), `log` (experiment-log heading or `-`). Do not use `|` inside cells. Update `updated` on every edit; rows not updated for 12 h while `running` or `pending` are flagged stale.

## Logging

Rating changes, preemptions and overrides get one line in `docs/theory/supervisor-log.md` Log. Process-level changes to this system get an entry in `experiment-log.md`.
