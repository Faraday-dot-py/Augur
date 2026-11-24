# neighbour-context-pair-branch-full-run: pre-registration (written 2026-11-16, before any run)

Status: APPROVED by Pilot 2026-11-16, pooled momentum. No GPU job submitted yet (script staged 2026-11-16, see experiment-log). Follows neighbour-context-pair-branch-replacing-central-pair-term (stage 1), -train-seed-replication, -with-soft-tangential-penalty. Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md section 7 item 9, 10a item 9 (full run as specified).

Definitions as in the stage 1 prereg (Exp E = ms_kp_pot_v_g128; err@20 = mean per-body position error after 20 self-fed ticks; tangential/radial = rms tangential over rms radial pair output on truth states at ticks 0/10/20; momentum error = |sum m a| / sum m |a| with momfix off).

## Evidence carried in (checked against local files 2026-11-16)
- Stage 1, train seed 4739, job 9277: err@20 0.792x control, momentum 0.876x, tangential/radial 0.064 (pass below 0.05, fails) -> INCONCLUSIVE. Fresh seeds (job 9621): 0.803x, 0.868x, 0.078. Source: docs/debugging/architecture-push-results.md:20. Raw stage 1 files are on Polaris only (results/pair_context/results.json, results_fresh.json in $HOME/augur_pair_context): not checked here, numbers from the results table.
- Replication job 9688 (seeds 4738, 4740), local raw results/pair_context_replication/{results,results_fresh}.json, verdict blocks: err@20 ratio 0.788/0.817 held-out, 0.796/0.827 fresh; tangential/radial 0.0558/0.0519 held-out, 0.0684/0.0632 fresh; momentum ratio 1.696 (4738), 0.741 (4740); candidate/frozen Exp E 0.869/0.896 held-out. Control err@20 spread (cv) across eval seeds 0.042 held-out, 0.098 fresh; across train seeds 0.003.
- Penalty job 9689 (weight 0.3, seed 4738), local raw results/pair_context_tangential_penalty/{results,results_fresh}.json: err@20 ratio 0.771 held-out, 0.781 fresh; tangential/radial 0.0054 held-out, 0.0068 fresh (per eval seed 0.0047-0.0082); momentum ratio 1.237/1.240; candidate/frozen 0.850/0.857. Single seed.
- Training rate: 0.146 s/it (pairctx-cand-s4738 219.7 s / 1500 its; ctrl 209-222 s), results/pair_context_*/train/*.json "seconds".

## Answer to the open question
Penalty variant 9689 cleared tangential/radial < 0.05: yes, 0.0054 held-out and 0.0068 fresh on seed 4738 only (results/pair_context_tangential_penalty/results.json:446-448, results_fresh.json:446-448, verdict.tangential_over_radial). The unpenalised candidate misses it on all 6 pooled cells (4738/4740 held-out 0.056/0.052, fresh 0.068/0.063; stage 1 0.064/0.078); only single held-out eval seeds dip below 0.05 (4738 seed 9200 0.0486, 4740 seeds 9100 and 9200 0.0489, 0.0446).

## What a full run means
Stage 1 recipe at 4x length, as in the draft scripts/polaris_neighbour-context-pair-branch-replacing-central-pair-term-full.sh (comment "4x iterations", ITERS=6000, SCENES=1500; the 6000 figure comes from that draft and the stage 1 prereg cost line, not from the spec, which gives only cost, 2 GPU-h for the full run). Everything else as stage 1: warm start Exp E, 50-tick scenes, batch 16, k 4->20, lr 2e-4 cosine, loss cap 3, same data and batch stream per train seed across arms, identical iteration count in all arms.
- Train seeds: 4738, 4739, 4740 (4738 start seed; 4739 and 4740 already used; spec says two seeds, three now exist).
- Arms per seed (3): matched control (`--fix ctrl`), candidate (`--fix pairctx`), candidate+penalty (`--fix pairctx --tan-penalty 0.3`, weight unchanged, not tuned). Reference only: frozen Exp E.
- Eval: held-out 9100/9200/9300 and fresh 12000/12100/12200, 32 scenes x 30 ticks; err@20 = mean over the 3 eval seeds per set; ratio per train seed per set = arm / its own control.
- Checkpoint used for scoring: final iteration only (no best-of selection).

## Proposed lines (proposed, awaiting Pilot approval; fixed once approved)
Headline arm: candidate+penalty (the only arm that can meet the tangential clause; soft penalty, no projection). The unpenalised candidate gets the same lines and is reported beside it.
R = err@20 ratio arm/control, per train seed, per eval set.
- PASS: R <= 0.90 on all 3 seeds on both sets AND tangential/radial < 0.05 on all 3 seeds on both sets AND arm err@20 / frozen Exp E err@20 <= 1.00 on all cells (guards "control got worse by fine-tuning": control is 1.10x frozen at 1500 its) AND not KILL.
- KILL: R >= 0.95 on any seed on either set, OR momentum error above control: seed-pooled ratio (mean of the 3 per-seed momentum ratios, per set) strictly above 1.0 on either set.
- INCONCLUSIVE: otherwise.
- Ties: R exactly 0.90 passes, exactly 0.95 kills, tangential exactly 0.05 fails, momentum exactly 1.0 is not above. Overall label: worse of the two sets (KILL if either, PASS only if both). Report seeds lowest index first. No argmin or best-checkpoint selection exists; if one is added it must break ties by lowest iteration.
- Floor lines: control err@20 spread across eval seeds (cv) reported next to every ratio; ratios are paired on identical scenes, line applied literally. Frozen Exp E is the numerical floor for the fine-tune family: arm/frozen reported for all cells; the 1.00 clause above is the floor line.
- Plateau guard as stage 1: candidate validation mean over iterations 5400-5999 more than 3% below mean over 4800-5399 turns a non-PASS into INCONCLUSIVE-undertrained (seed 4740 declined 2.7% at 1500 its, close). One extension at double length for all arms together allowed.
- A value that would still fail PASS: R 0.93 on one seed; tangential/radial 0.06 with R 0.85; arm/frozen 1.02 with R 0.88.
- Momentum choice for the Pilot: the original line (spec item 9) is "momentum error above control" with no pooling rule. Read per seed it would already have killed 4738 (1.696 candidate, 1.237 penalty). Pooled over seeds is proposed because the metric is switched off by momfix in deployment and the control's own momentum varies 0.045 vs 0.075 across seeds at identical recipe (results/pair_context_replication/results.json arms ctrl-s4738, ctrl-s4740). The Pilot may instead require per-seed (stricter; probability of PASS drops to about 10%).

## Prediction (stated before running)
R 0.77-0.88 (gain persists; some chance control catches up with 4x length and R rises toward 0.90); tangential/radial penalised 0.003-0.015, unpenalised 0.04-0.08; momentum pooled ratio about 1.0-1.3 (existing unpenalised seeds pool to 1.10 held-out, 1.11 fresh, already above 1.0) (seed-dependent, 1.70/0.74/0.876 unpenalised, 1.24 penalised). Probabilities, headline arm, pooled momentum reading: PASS about 30%, KILL about 55% (nearly all through the momentum clause; err@20 kill about 5%), INCONCLUSIVE about 15% (R 0.90-0.95 on some seed, or arm/frozen > 1.00). Per-seed momentum reading: PASS about 10%. Honest basis: err@20 gain and tangential clearing replicated at 1500 its but untested at 6000 its; momentum has only 1 seed for the penalised arm.

## Cost and limits (estimate, untested at 6000 its)
- 9 trainings x 6000 its x 0.146 s/it = 880 s each; x1.15 for 1500-scene data generation and validation = about 1000 s; total about 2.5 h, plus scoring about 0.2 h (stage 1 job 9277 took 711 s for 2 trainings plus eval, so about 270 s non-training overhead) = about 2.7 GPU-h. Spec 2 GPU-h; this is about 1.35x because of the 3rd seed and the penalty arm. Cheaper option: seed 4739 only with ctrl, cand, penalty (3 cells) about 0.9 GPU-h, loses the seed replication.
- Layout: 3 per-seed training jobs (3 cells each, about 0.9 h, limit 1:45) plus one scoring job; per-seed jobs chained so at most 2 GPUs run at once (seeds 4738 and 4739 in parallel, 4740 after one finishes via afterany), scoring afterany on all. Wall about 2 h after start. No `--nodelist`, `--nice` per the priority registry rating.
- Checkpoints: train_scatter_ft.py saves model, optimiser, iteration, RNG, validation log every 50 its (scripts/train_scatter_ft.py:118), `--resume` (:103-114), SIGTERM handler (:79-80, :176-181, installed :235), save frequency set by --log-every default 50 (:184-186, :210); job scripts need `#SBATCH --requeue`, `--signal=B:TERM@120`, a TERM trap that forwards to the child, per-cell skip of finished `*_train.json`. Cells are under 20 min each, so the 1 h rule is covered anyway.
- Before submit (not done here): rewrite the draft full script (it has seed 4739 only, no penalty arm, 3:30 h limit), extend scripts/pair_context_eval.py with a full-run `--lines` mode (3 seeds, pooled momentum, frozen clause, both sets) and CPU-test it, confirm tarball contents with `tar -tzf`, add registry rows.

## What each outcome means
- PASS: free tangential pair output with a soft central-ness penalty beats the stock central pair term by at least 10% err@20 on 3 seeds at full training length; adopt `pairctx` + penalty as the near-field pair form for later experiments; candidate must still pass the O(N) cell/adaptive-depth port (not implemented; dense path only).
- KILL via momentum: the accuracy gain is real but the pair force breaks momentum balance more than the control; the gain would rely on momfix, so record as "gain with momentum cost", not adoption.
- KILL via err@20: gain at 1500 its was a short-training effect; drop.
- INCONCLUSIVE: report the cell that failed; no loosening; a rerun needs a new prereg.

## Constraints check
pp=2, knn=16 kept; no hard-wired conservation (soft penalty, user rule); one change vs matched control per arm; no sweep (weight 0.3 fixed); start seed 4738 then 4739/4740; Polaris only, 2-GPU cap respected by chaining; dense path only (limit stated).

## Already tried (experiment-log checked 2026-11-16)
Stage 1 (9277, 9621), replication (9688), penalty (9689), all logged 2026-11-12. No 6000-iteration run exists.
