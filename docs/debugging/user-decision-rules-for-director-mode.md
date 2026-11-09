# User decision rules for director mode

How awebb decides after an experiment or job result, drawn from past prompts and decision memories. Evidence only. Sources: memory files in `/home/awebb/.claude/projects/-home-awebb-Research-Augur/memory/` (cited as `memory/<file>`), `docs/debugging/experiment-log.md`, and session transcripts (cited as `transcript <first 8 chars of session id>`). Each rule is marked clear or ambiguous. Thin evidence (one instance) is stated in the rule.

## Verdict handling

1. **Amend a mis-calibrated gate, never loosen silently (clear).** When a registered line fails because the line itself was wrong and the data are otherwise fine, the user approves a logged amendment after being asked. The as-registered verdict stays on record.
   - "Option 1, amend the gate" (transcript c04c1712; moon-arm rejection-rate gate 0.6 vs actual 0.992, agent had not loosened it itself).
   - "Yes, adopt the Row 3 replacement" (transcript ece964ce). Logged: "user approved, post hoc ... As-registered verdict INCONCLUSIVE stays on record" (experiment-log.md:3575; memory/project_overnight_results_2026_11_03.md).
   - Director consequence: ask before amending; never amend unprompted. Amendment prompts carried a concrete replacement line and a recommendation.

2. **A pass on a different line than the registered one is not a pass (clear).** The amended Row 3 line is recorded as PASS only with the as-registered INCONCLUSIVE kept next to it. Source: memory/project_overnight_results_2026_11_03.md.

3. **KILL stands; do not chase it when it is a cheap or marginal idea (clear, one instance of the wording).** "Try the cheap token wins, come up with another solution for polaris waits. If the result is a kill either way, let's not worry about it." (transcript b2f767fa). In contrast, for ideas he cares about: "Keep pushing and augmenting the idea until it's exhausted whether it works or not" (transcript 801bda9c, kmlp fix).
   - Ambiguous part: what distinguishes "don't worry about a kill" from "exhaust the idea". Likely his own stated interest, not result type.

4. **Early-kill or unreplicated runs are not replications (clear).** Seed 4739 EARLY_KILL at it 1500 was recorded "not replicated, final lines never measured" and the user held further replay rounds. Source: memory/project_tiered_replay_result_2026_10_29.md.

5. **Do not claim validated or mechanism-specific when a control reads generic (clear).** Truth-state control 4104 read GENERIC (2.10x vs 2.5x line); standing instruction: "don't call replay validated or visited-state-specific; ask before merging the branch to master". Source: memory/project_tiered_replay_result_2026_10_29.md.

## Thin margins, seeds, replicates

6. **Thin PASS gets recorded as thin, not upgraded (clear).** Row 4 seed 4739 was logged "PASS thin" (err@20/BL 0.999/0.988/0.876, pass line <=1.0x BL). The user did not push back or ask for more seeds after it. Source: memory/project_overnight_results_2026_11_03.md. Ambiguous: whether he would have wanted a third seed; no statement either way.

7. **Single-seed or single-scene wins need a replicate before counting (clear).** After arm B showed a single-scene result, the agent's doc said it needed the 4-scene median and seed 4739; user: "run the 4-scene median and seed 4739 replicate" (transcript 92b38cc3). For tier-0 the second seed was approved with explicit lines: "Pass tiled err@20<=1.0x BL, plain<=1.10x; kill >1.25x" (memory/project_decisions_2026_11_03.md).

8. **Replication seed order: 4738 first, 4739 as the replicate (clear).** Standing rule "Start with seed 4738" (global CLAUDE.md); used consistently, eval seeds 9100/9200/9300 held out.

9. **Select on held-out seeds; call effects real only if they beat a same-length control on most sets by >5% (clear).** Fine-tunes differ about 10-15% by noise. Source: memory/feedback_select_on_heldout_seeds.md.

10. **Every experiment needs its own noise floor (clear).** "Each exp should have its own chaos floor calculated so we know what to measure against" (transcript 34f69960). Also: "Make sure that each fix we implement is tested individually" (34f69960), later softened to a recommendation, bundling OK with a reason (memory/feedback_test_each_fix_individually.md).

## What he approves

11. **Batched approval of agent-proposed, pre-registered work (clear).** Short replies on lists: "Approve both" (transcripts 5e32e79e, 7b0a6d52), "Approve all three" (ece964ce), "Accept all 4, queue the retrain, next agent" (34f69960), "approve all" (197caaca). He approves when proposals come with pass/kill lines and a recommendation. Many such approvals are given while the item has a clear single recommendation.

12. **Approves with a single subtraction or tweak (clear).** "Accept 1 and 2, drop 3" (34f69960); "Approved on 1-3 ... Hold 6 for now" (ec5b8f43); "Approved, go with your recommendations except for 4. Interpolate between them" (c04c1712); "Accept B with A as ablation" (34f69960). Director consequence: partial approval is the normal form; execute exactly the approved items.

13. **Queue ordering is user-set and then binding (clear).** "Submit only AFTER the drift jobs are queued"; drift jobs queued BEFORE the near-field run (memory/project_decisions_2026_10_22.md). Another: "Hold till the no-velocity retrain finishes" (transcript 49e2af9c).

14. **Approves one-arm, scoped versions of ladders (clear).** Euclid heads-only: "Try it" meant one arm at the full schedule, not the whole ladder, with pass = "ab err@20 <= .0076 AND dE gate" (memory/project_decisions_2026_10_22.md).

15. **Diagnostic follow-ups to understand a result are approved quickly (clear).** Near-field and replay follow-ups ("Yes, chase them now", 65480f12; replay validation and control, "Approved on 1-3", ec5b8f43). Seed-replication and control arms are the common type.

## What he rejects or defers

16. **Rejects anything that violates the pairwise/O(N) or learned-dynamics constraint, even to launch (clear).** "No, id rather not include it due to the pairwise concern. Otherwise, launch it" (transcript 5fba138e). "I don't want to hard-code conservative forces, as I want the model to be able to learn uneven interactions. Past that, I approve the fixes" (34f69960). Row 8 exact-mesh-pair complement "NOT acceptable (central/reciprocal form)" (memory/project_decisions_2026_11_03.md). Also: item 6 "violates the pairwise interaction prohibition" (transcript 3bb41af9).

17. **Rejects ideas that only work for one interaction type or let the model lean on enforced rules (clear).** "3 doesn't generalize well outside velocity, 4 lets the model rely on enforced rules instead of representing the structure itself ... 7 is specific to certain interaction types" (transcript 3bb41af9).

18. **Defers (holds) rather than rejects when a precondition is unmet, with a stated release condition (clear).** "Hold 6 for now" (ec5b8f43). dt 0.05 "diagnostic only, stays held until replay plateaus >2x BL"; time-reversal loss "per-sim opt-in OK, stays deferred" (memory/project_decisions_2026_11_03.md). Held items in the tiered track: dt 0.05 fine-tune, curl/drag penalty arm, more replay rounds (memory/project_tiered_replay_result_2026_10_29.md).

19. **Pauses a whole track after repeated no-gain results, and asks for save and push (clear, one instance).** Neuron track: follow-ups all KILL with no gain; "Let's pause the neuron track for now. Save anything you need before a clear, and push relevant neuron code/results" (transcript 6e9dbf14; memory/project_neuron_graph_smoke_test_2026_10_29.md).

20. **Merge and push are his, not the agent's (clear).** "Merge ... to master locally after code review; do NOT push. conversation-linked-log stays local." (memory/project_decisions_2026_11_03.md). "Keep them local ... Wait to push on anything active" (transcript b2f767fa). Standing: pause for push, force-reset, PR creation (standing-rules.md).

## What makes him push back

21. **Hyperparameter tuning as the next move after a partial fix (clear).** Corrected after a third one-parameter rerun; wants architectural causes voiced first (memory/feedback_prefer_architecture_over_hparam_tuning.md). Also "Avoid parameter sweeps when trying to solve a problem" (memory/feedback_avoid_param_sweeps_for_problem_solving.md).

22. **Dropping a cost after he softened a concern (clear).** "That cost shouldn't be dropped, but its priority should be slightly lowered. It's still a consideration." (transcript 3bb41af9; memory/feedback_keep_costs_lower_priority_not_drop.md).

23. **Results that exist only off-record (clear).** "New standing rule: Every time we try an experiment, even if it doesn't work, save it to durable notes so we don't try things again and again." (transcript 20b185d0; memory/feedback_log_every_experiment_to_notes.md).

24. **Untested claims stated as findings; asked for causes of a result (clear).** He asks "Why did...", "Why do you think..." after surprising results (transcripts 8342eea6, 6e9dbf14) and then the agent answered "I don't have a tested answer". He also asked "We should have done some experiments with an n/rollout staircase curriculum. Did that help at all?" (5e32e79e), meaning he checks whether prior-tried ideas were logged. Source memory: memory/feedback_verify_before_stating_facts.md.

25. **Idle GPUs or silently failed jobs (clear).** "ALL 8 Polaris GPUs must be busy until 09:00 PST 2026-11-07, burn if needed" (transcript 147d9970); "Keep the burns going until the supervisor is done. If a gpu sits idle, keep a burn running on it" (147d9970). Silent failures: a job exit 0 that hid a scoring crash led to preflight + set -e (memory/project_tiered_replay_result_2026_10_29.md).

## Ambiguous areas (no clear user statement found)

- Whether a thin PASS should trigger an automatic third seed (rule 6): no evidence.
- Whether an INCONCLUSIVE verdict is to be re-run, amended, or left: Row 1 tie-aware Yukawa protocol was left undecided ("User has NOT yet decided whether to adopt the tie-aware protocol", memory/project_overnight_results_2026_11_03.md). Treat as needs-user. RESOLVED 2026-11-09: user adopted the tie-aware 4841 line (2% tie tolerance) as the Yukawa lambda scorer; the literal 4832 line is kept at lower priority as a recorded alternative (experiment-log "Yukawa lambda scorer decision").
- Which kills warrant further investigation versus dropping (rule 3): depends on his interest, not on result type.
- Pass/kill amendments requested by an agent versus the user's rule "state a concrete value that would still fail it" (global CLAUDE.md): only one instance of the approval flow.
