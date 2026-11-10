# finite-speed-local-field-medium: CURIOSITY (written 2026-11-10, before any run)

Label: curiosity. No prediction is pre-registered as a pass or kill, so there is no pass/kill line. Results are observations and are never presented as a verdict. Spec: section 7, item 20; user decision 6 in section 10a (approved as curiosity, capped at 1 GPU-hour, observations only). Standing rule: a curiosity experiment records what was run and what was observed.

Definitions. N: body count. Tick: one step of dt 0.1. Propagation speed bound: the field is advanced n_sub substeps per tick by two 3x3 convolutions and a 1x1 head, so influence moves at most 2 cells per substep, at most 2*h*n_sub/dt length units per time (h = cell size). Retarded-force reference: the exact softened force using each source's position at the retarded time, found by iterating s = t - |x_i(t) - x_j(s)| / c with history interpolated between ticks and clamped to tick 0.

## What is built

`model/local_field_medium.py`: a grid (64 cells, extent 64) with its own state (potential phi, one auxiliary channel psi, 4 hidden channels), carried between ticks. Bodies only deposit mass (CIC) and read -grad(phi) (bilinear gather). Update rule: S += (dt/n_sub) * head(conv(conv([S, density, log n_sub]))), learned, local, zero-initialised head. No FFT, no global operation anywhere.

Training (`scripts/finite_speed_local_field_medium.py`): teacher-forced on exact trajectories (positions come from the exact simulation; only the force is predicted), 32-tick windows, loss on ticks 8 onward, relative squared force error against the exact instantaneous force, N alternating 2 (random circular-ish pairs) and 20, batch 8, Adam 1e-3, seed 4738, 480 s of training per speed. One model per n_sub in {1, 2, 4, 8} (speed bounds 20, 40, 80, 160).

## Observations to record (reported, not scored)

1. Force error against exact at N = 2, 20, 1000 as a function of propagation speed: RMS relative force error (norm of the difference over norm of the exact force, all bodies and ticks after the skipped warm-up), seeds 9100, 9200, 9300, 64 evaluation ticks. N=1000 is out of the training range (zero-shot).
2. Cost per tick: ms per tick at N=1000, per speed.
3. Whether the filtered front stays within 5% of the retarded-force reference at N = 2 and 20: fraction of ticks (after skipping 2 domain crossing times, at most half the run) whose RMS relative error to the retarded reference is at most 5%, for the raw force and for a 3-tick moving average ("filtered"); also the median per-tick error. The retarded reference is computed at the model's own speed bound. Audit row 40's line is the origin of the 5%; here it is reported, not scored.

## Cap and resources

1 GPU-h hard cap (Slurm time limit 01:00:00). Estimate (untimed; no logged rate for this model): 32 min training plus under 10 min evaluation. The script stops training per speed at its time budget (`--train-seconds`), saves each speed's results as it finishes, and skips finished speeds on rerun. Not a long run (under 1 h), so no periodic checkpoints; per-speed checkpoints are saved at the end of each speed.

## Constraints check (re-read 2026-11-10)

- Local (Til): yes by construction, receptive field bounded per substep. O(N): deposit and gather are O(N); the grid update is O(cells), independent of N. Generalization: the rule is learned; no force law is built in. pp=2 knn=16: not applicable, there is no pair term in this model (a missing near field is expected to dominate the error at N=20 and 1000).
- Known gap in the prototype lineage: the earlier two-body prototype (`scripts/causal_field_2body.py`) used a hand-written wave equation; the rule here is learned.

## Known weaknesses (kept in the notes)

- Teacher-forced only: no rollout error is measured, so nothing here says how the medium behaves in closed loop.
- The medium is trained against the instantaneous exact force, not the retarded force, so at low speed its error against the retarded reference and against the exact force are different questions; both are recorded.
- Short training (480 s per speed) means differences between speeds partly reflect optimisation, not capacity.
