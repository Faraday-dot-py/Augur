# Time-estimate accuracy review

Status: research only, uncommitted. Nothing applied (no memory, template or registry edits). Written 2026-11-02.

Ratio = actual minutes / midpoint of the estimate range. Below 1 means overestimate (job finished early). Basis "measured" = the estimator had a logged s/it (seconds per iteration), a same-recipe prior run, or a running-job pace; "unmeasured" = a guess from prereg budgets or memory. Source ids are transcript line (session-id:line) or doc file:line; transcripts are in ~/.claude/projects/-home-awebb-Research-Augur/. Nothing in the table was computed from sacct re-queries; actuals are the elapsed values the sessions recorded. Rows marked inferred in the cause column below were read from context rather than a stated duration.

## 1. Pair table

60 verified pairs (Sep 21 - Nov 3) after excluding: failed or early-stopped jobs (j2890, j2850, j2868, C01, C07), gate stops, rows without a numeric estimate (C04) and unverified rows. Three auth/channel-wait rows (C08, C09, C10) are listed but kept out of the statistics (see 3d). Full machine-readable rows: scratchpad final_pairs.json.

Note on D2/D1/D4: job elapsed includes about 6 min of startup and scene generation; training alone was 12 m 6 s for 4740 (registry says 19 min = whole job). Actuals here are whole-job elapsed, as the estimators' budgets also referred to the job.

| id | type | estimator | estimate | actual | basis | ratio | source (est) |
|---|---|---|---|---|---|---|---|
| C08 | auth wait | main assistant | not retrying for another 891s | connected 03:31:01 after user /mcp reconnect; override attempts at 03: | unmeasured | 0.41 | c04c1712 sub:136, 2026-11-03 03:25:00 |
| C10 | auth wait | main assistant | wait 13s | agent resumed 20:47:58 after connect 20:47:55; Ready at 20:46:59 | unmeasured | 3.73 | c04c1712:1171, 2026-11-02 20:47 |
| j2818_datagen | data gen | main assistant | ~53min gen | dataset gen finished in 14.5min (not 53min) | unmeasured | 0.27 | c62c8e82:319, 2026-09-22T03:07 |
| j2812_stage1 | data gen | main assistant | ~23min dataset gen + fast GPU training | {"job_id":2812,"state":"COMPLETED","elapsed":"00:25:47"} | measured | 1.01 | 7d1c6dc0:1662, 2026-09-21T22:56 |
| D7 | eval | subagent | 10-20 min | elapsed 00:02:16 | unmeasured | 0.15 | c04c1712:1894 11-02T23:21Z |
| D9 | eval | subagent | about 3 h | elapsed 00:28:03 | unmeasured | 0.16 | c04c1712:1983 |
| D3 | eval | subagent | Eval: about 20 min | COMPLETED 3 m 13 s | unmeasured | 0.16 | c04c1712:1343; experiment-log.md:3479 |
| B20 | eval | subagent | It should take roughly 1.5-2 hours. | {"job_id":3636,"state":"COMPLETED","elapsed":"00:23:26"} | unmeasured | 0.22 | 92b38cc3:301, 2026-10-15T18:11 |
| D6 | eval | subagent | about 1 h (log: roughly 1 h) | elapsed 00:13:54 | unmeasured | 0.23 | c04c1712:1775; experiment-log.md:3507 |
| B09 | eval | subagent | Eval: 6 scenes x 3200 model steps up to 100 tiles + 100-frame truth reruns: ~0.5 h. | about 7 min to evaluate | unmeasured | 0.23 | 9eb2ca8c:112, 2026-10-23 (doc scatter-lj.md text) |
| B13 | eval | subagent | The job should take well under 1 hour, so no SIGTERM trap. | {"job_id":3830,"state":"COMPLETED","elapsed":"00:11:53"} | unmeasured | 0.40 | 3e784190:146, 2026-10-21T20:27 |
| D11 | eval | subagent | about 30 min | elapsed 00:22:45 | unmeasured | 0.76 | c04c1712:2838 |
| j2970_1m_10k | other | main assistant | it should take about 20-30 min, but that's unmeasured | {"steps_done": 10000, "wall_s": 1030.07} | unmeasured | 0.69 | 8822c031:531, 2026-09-25T06:24 |
| bh_1b_meshonly | other | main assistant | a 1B run would take about 1.7 s per tick, so 100 ticks is about 3 min | 1325 ms/tick median ... 263 s for the loop incl. diagnostics (137 s ti | measured | 0.76 | 5c57629a:385, 2026-10-08T23:11 |
| j3118_bh100k | other | main assistant | 250 steps ... ETA ~3 min compute | Job 3118 completed (2:37 elapsed, ~0.6s/step) | measured | 0.95 | fbd423a2:409, 2026-09-28T03:28 |
| C09 | other | main assistant | transient ... next check about 15:55 | recovered only after polaris stop then start at 00:02:29-46 (about 12. | unmeasured | 2.50 | c04c1712:2044, 2026-11-02 23:50:59 |
| B14 | queue wait | subagent | 3642, 4h14m in, up to ~6h left | Job 3642 elapsed 04:33:54 COMPLETED (4:13:49 at 00:52) | unmeasured | 0.11 | a5d11050:919, 2026-10-16T01:01 (also 878: "up to about 6.4 hours left" |
| B23 | queue wait | subagent | I looped on job 3950 through many poll rounds, roughly 2 to 3 hours by my estimate | <duration_ms>953162</duration_ms> (436) and <duration_ms>1946599</dura | unmeasured | 0.32 | 49e2af9c:526, 2026-10-24T22:38 |
| v12_queue_wait | queue wait | main assistant | v12 ... auto-start once v11 frees its slot; v11 is at epoch 12/15, ~15-20 min from done | 2856 PENDING (QOSMaxGRESPerUser) at 08:58 with 2855 at 1:07:50; 2855 C | measured | 0.87 | 82b05453:914, 2026-09-23T08:58 |
| B17 | render | subagent | 4053 solar system, about 5 min estimated | {"job_id":4053,"state":"COMPLETED","elapsed":"00:00:34"} | unmeasured | 0.11 | 49e2af9c:5178, 2026-10-26T22:29 |
| B18 | render | subagent | 4054 1e9 black hole, 300 ticks, about 25 min estimated | {"job_id":4054,"state":"COMPLETED","elapsed":"00:17:31"} | measured | 0.70 | 49e2af9c:5178, 2026-10-26T22:29 |
| B19 | render | subagent | 4055 1e9 clumpy collapse, 200 ticks, about 15 min estimated | {"job_id":4055,"state":"COMPLETED","elapsed":"00:12:13"} | measured | 0.81 | 49e2af9c:5178, 2026-10-26T22:29 |
| B22 | render | main assistant | the agent will rerun the 1e9 simulation on Polaris ... That rerun should take about 4 min. | job 3443, 336 s total across two passes | unmeasured | 1.40 | 5c57629a:491, 2026-10-09T00:19 |
| B21 | training | user | save a periodic full-state checkpoint if the run exceeds ~1 hour (it should not at ~45 min | A 1e9 pass is about 5 minutes of wall time, not ~45 | unmeasured | 0.11 | ef199bc0:480, 2026-10-12 |
| C05 | training | subagent | expected 1-1.5 GPU-h | about 11 min wall, about 0.2 GPU-h | unmeasured | 0.15 | ec5b8f43/agent-aac2a44fdcc5cc04b registry row, 2026-10-29 ~20:42 |
| D10 | training | subagent | expected about 1 h | elapsed 00:12:56 | unmeasured | 0.21 | c04c1712:2838 11-03T01:12Z |
| B08 | training | subagent | Train 6000 it, batch 8, k<=32 two-force Verlet steps: guess 0.5-1 s/it = 1-1.7 h per arm. | 6000 it, 0.18 s/it = ~18 min train + ~7 min eval per arm; est. had bee | unmeasured | 0.22 | 9eb2ca8c:112, 2026-10-23 (doc scatter-lj.md text); also 65480f12:835 |
| j2863_v16 | training | main assistant | Given the ~2h+ expected runtime | {"job_id":2863,"state":"COMPLETED","elapsed":"00:28:10"} | unmeasured | 0.23 | b0f744bc:333, 2026-09-23T23:20 |
| B16 | training | subagent | then E (about 15 minutes) | Both jobs completed (3950 in 40 minutes, 3953 in 4) | unmeasured | 0.27 | 49e2af9c:436, 2026-10-24T22:22 |
| D2 | training | subagent | Control: about 30-60 min (prereg 1 to 3 GPU-h per arm) | COMPLETED 13 m 11 s | unmeasured | 0.29 | c04c1712:1343; experiment-log.md:3479 |
| j3378_scatter_fixes | training | subagent | I estimate about 1-1.3 h per run, so 8-10 h in total, with a 12 h time limit | {"job_id":3378,"state":"COMPLETED","elapsed":"03:47:42"} | unmeasured | 0.42 | 5f0d1d08:295, 2026-10-07T02:29 |
| j3397_pp8_wrapper | training | subagent | Runtime estimate is about 1.5-2 hours, so I'm requesting 5 h | I submitted the wrapper as job 3397 (MODE=all, 5h limit); it ran in ab | unmeasured | 0.43 | 5f0d1d08:253, 2026-10-07T01:33 |
| D1 | training | subagent | Long-horizon: about 1-2 h (prereg: training 1 to 3 GPU-h per arm) | COMPLETED 40 m 38 s | unmeasured | 0.45 | c04c1712:1343 11-02T20:56Z; experiment-log.md:3479 |
| C06 | training | subagent | about 1.2 GPU-h | about 40 min wall, about 0.7 GPU-h | unmeasured | 0.56 | ec5b8f43/agent-aac2a44fdcc5cc04b:558, 2026-10-29 23:39:42 |
| C02 | training | subagent | expected 2-3 GPU-h | COMPLETED 1h32m (about 1.55 GPU-h) | unmeasured | 0.61 | e1aaaa18/agent-a85ea1501bbd185f0:305, 2026-10-29 20:10:40 |
| D8 | training | subagent | about 15-25 min | elapsed 00:12:54 | measured | 0.65 | c04c1712:1983 11-02T23:25Z |
| j2857_v13 | training | main assistant | Training + eval typically takes ~20-30 min based on prior runs | {"job_id":2857,"state":"COMPLETED","elapsed":"00:18:37"} | measured | 0.74 | 3ab3cc1b:283, 2026-09-23T19:19 |
| B07 | training | main assistant | ETA is about 18:00-18:15 PDT. That is my estimate from the job log | Agent "Euclid heads-only ladder runs" finished (notification 2026-10-2 | measured | 0.82 | b64ff1a9:796, 2026-10-19T23:58 |
| v12_train | training | main assistant | v12 (state_vel_weight, ~20 min) | {"job_id":2856,"state":"COMPLETED","elapsed":"00:16:56"} | measured | 0.84 | 82b05453:888, 2026-09-23T08:31 |
| D4 | training | subagent | 15-30 min | elapsed 00:19:07 | measured | 0.85 | c04c1712:1775 11-02T23:08Z |
| j2817_200ep | training | main assistant | Expect it done in ~20 min | {"job_id":2817,"state":"COMPLETED","elapsed":"00:17:16"} | measured | 0.86 | c62c8e82:205, 2026-09-22T02:42 |
| B15 | training | subagent | roughly 45 minutes on one GPU | Both jobs completed (3950 in 40 minutes, 3953 in 4) | measured | 0.89 | 49e2af9c:411, 2026-10-24T22:10 |
| D5 | training | subagent | 15-30 min | elapsed 00:20:02 | measured | 0.89 | c04c1712:1775 |
| j2898_2899_ft | training | main assistant | jobs 2898 (v35) and 2899 (v36) run next (~10 min each) | 2898 COMPLETED 00:08:54; 2899 COMPLETED 00:08:56 | measured | 0.89 | 9e78283d:1961, 2026-09-24T15:03 |
| B06 | training | subagent | each arm takes roughly 20-28 min | Arms took 18 min (heads-only) and 26 min (full fine-tune) of training  | measured | 0.92 | b64ff1a9:796, 2026-10-19T23:58 |
| v11_remaining | training | main assistant | v11 (epoch-count, ~45 min left) | {"job_id":2855,"state":"COMPLETED","elapsed":"01:23:04"}; elapsed 39:5 | measured | 0.96 | 82b05453:888, 2026-09-23T08:31 |
| j2859_v15 | training | main assistant | should take ~20 min based on v14's run | {"job_id":2859,"state":"COMPLETED","elapsed":"00:19:18"} | measured | 0.97 | 0c67dbca:1387, 2026-09-23T21:57 |
| B02 | training | subagent | At the current pace of about 3.3 iterations per second it should finish around the 3:30 ma | Job 3592 finished (3:26 elapsed). | measured | 0.98 | 57a58a98:420, 2026-10-14T02:17 |
| j2821_200ep | training | main assistant | At ~14:47 for 80 epochs, expect ~37min for 200 | {"job_id":2821,"state":"COMPLETED","elapsed":"00:36:26"} | measured | 0.98 | c62c8e82:906, 2026-09-22T04:38 |
| j3080_6000step | training | main assistant | costs 1.05 s/step, so about 1.75 h for 6000 steps | {"job_id":3080,"state":"COMPLETED","elapsed":"01:45:03"} | measured | 1.00 | dc6f3875:597, 2026-09-26T20:15 |
| j3272_train | training | main assistant | Training (job 3272) is running, about 2.5 h | {"job_id":3272,"state":"COMPLETED","elapsed":"02:30:29"} | measured | 1.00 | a61541e2:583, 2026-10-03T06:14 |
| j2901_2902_ft | training | main assistant | ~9 min each, expected done by ~09:25 | 2901 COMPLETED 00:09:14; 2902 COMPLETED 00:08:58 | measured | 1.01 | 9e78283d:2253, 2026-09-24T16:08 (wakeup prompt; written earlier) |
| j2903_2905_ft | training | main assistant | each ~9 min serial, all expected done by ~09:52 | 2903 00:09:15; 2904 00:09:13; 2905 00:09:12 (all COMPLETED) | measured | 1.02 | 9e78283d:2328, 2026-09-24T16:38 (wakeup prompt) |
| C03 | training | subagent | I expect 1h30m to 1h40m | elapsed 01:38:48 | unmeasured | 1.04 | 8c916383:206, 2026-10-30 01:52:49 |
| j2885_v25 | training | main assistant | started ~04:42, 5h limit, expect ~40-60 min | {"job_id":2885,"state":"COMPLETED","elapsed":"00:52:13"} | measured | 1.04 | 9e78283d:1319, 2026-09-24T12:12 |
| j3089_6000step | training | main assistant | About 1 h 40 min. Job 3089 ... 1.06 s/step | {"job_id":3089,"state":"COMPLETED","elapsed":"01:45:02"} | measured | 1.05 | dc6f3875:1119, 2026-09-26T23:29 |
| j2819_train | training | main assistant | mostly just the ~14min training portion | {"job_id":2819,"state":"COMPLETED","elapsed":"00:14:47"} | measured | 1.06 | c62c8e82:669, 2026-09-22T04:00 |
| j2815_stage2 | training | main assistant | ~25 min projected for the full set (dataset gen), then 20 epochs of training | {"job_id":2815,"state":"COMPLETED","elapsed":"00:28:18"} (training 20  | measured | 1.07 | 5c008c95:663, 2026-09-22T00:32 |
| j2858_v10 | training | main assistant | v10 took ~20min historically | {"job_id":2858,"state":"COMPLETED","elapsed":"00:19:59"} | measured | 1.14 | 0c67dbca:739, 2026-09-23T21:10 (also :769 "v9 previously took ~15-20mi |
| B01 | training | main assistant | Each arm is about 3 hours on one H200. | Job 3592 finished (3:26 elapsed). | unmeasured | 1.14 | 57a58a98:196, 2026-10-13T23:03 |
| B12 | training | subagent | 80,000 iterations per seed ... (0.24-0.30 s/it, about 13.4 GPU-h total) | jobs 3898/3899 train, ~7 h each, 80k iterations | measured | 1.17 | 65480f12:1212, 2026-10-23T00:18 |
| B04 | training | subagent | At about 0.106 s/it that is roughly 1.9 h each, about 4 GPU-h with evals. | about 2.9 h wall per job including evals | measured | 1.53 | 65480f12:381, 2026-10-22T20:43 |
| B03 | training | subagent | past iteration 24,000 of 60,769 in stage E, about 10 iterations per second. I expect it to | Job 3592 ... "elapsed":"59:46" (line 357, same minute) ; finished (3:2 | measured | 2.43 | 57a58a98:376, 2026-10-14T00:18 |

## 2. Accuracy summary

Ratio = actual / estimate midpoint, n = 60 (auth rows excluded).

| group | n | median | IQR | min | max | over / under |
|---|---|---|---|---|---|---|
| all | 60 | 0.82 | 0.29-1.00 | 0.11 | 2.43 | 44 over, 16 under |
| training | 40 | 0.89 | 0.53-1.03 | 0.11 | 2.43 | 26 / 14 |
| eval | 8 | 0.23 | 0.16-0.27 | 0.15 | 0.76 | 8 / 0 |
| render | 4 | 0.76 | 0.55-0.96 | 0.11 | 1.40 | 3 / 1 |
| queue wait | 3 | 0.32 | 0.22-0.60 | 0.11 | 0.87 | 3 / 0 |
| data gen | 2 | 0.64 | 0.46-0.83 | 0.27 | 1.01 | |
| other (sim ticks) | 3 | 0.76 | 0.72-0.85 | 0.69 | 0.95 | 3 / 0 |
| basis measured | 32 | 0.96 | 0.86-1.03 | 0.65 | 2.43 | 26 within 0.8-1.25 |
| basis unmeasured | 28 | 0.27 | 0.20-0.48 | 0.11 | 1.40 | 2 within 0.8-1.25 |
| unmeasured training | 14 | 0.36 | 0.23-0.53 | 0.11 | 1.14 | |
| measured training | 26 | 0.98 | 0.89-1.05 | 0.65 | 2.43 | 22 within 0.8-1.25 |
| Nov 2 session (D1-D11) | 11 | 0.29 | 0.19-0.70 | 0.15 | 0.89 | 11 / 0 |
| estimate >= 60 min | 24 | 0.50 | 0.23-1.01 | 0.11 | 2.43 | |
| estimate < 60 min | 36 | 0.86 | 0.58-0.97 | 0.11 | 1.40 | |

Direction: the dominant bias is overestimation, and it is almost entirely a measured vs unmeasured split. Measured estimates are unbiased (geometric mean 0.97). Unmeasured ones run at about 0.3x (geometric mean 0.31). Evals are the worst type: every one of 8 finished early, median 0.23.

Worst overestimates (ratio): B14 0.11 (remaining-limit guess, see 3d), B21 0.11 (1e9 pass guessed ~45 min, actual ~5 min), B17 0.11 (solar render 5 min guess vs 34 s), C05 0.15 (1-1.5 GPU-h vs 11 min), D7 0.15 (eval 10-20 min vs 2.3), D9 0.16 (eval 3 h vs 28 min), D3 0.16 (eval 20 min vs 3.2).

Worst underestimates: B03 2.43 (60 min projected from an early rate; the k ramp raised cost per iteration), B04 1.53 (omitted per-job eval; wall 2.9 h vs 1.9 h of training), B22 1.40 (second pass omitted), B12 1.17, B01 1.14.

## 3. Causes (each tied to rows)

a. Unmeasured speed guesses. Preregs budgeted GPU-hours before any s/it was logged (experiment-log.md:3479 "training 1 to 3 GPU-h per arm ... speed untested; eval about 20 min"). Actuals: control arm 13 m 11 s (D2, ratio 0.29), long-horizon arm 40 m 38 s (D1, 0.45). The same pattern earlier: C05 (1-1.5 GPU-h, 11 min), C02 (2-3 GPU-h, 1 h 32 m), C06, B08 (81 min vs 18). Once the control run gave s/it (0.125 s/it, 749 s, sss_ctrl_train.json) the later estimate was accurate (D4 0.85, D5 0.89).

b. Evals estimated by feel, not scaled from the previous eval. 4716 ran in 3 m 13 s before the 4742 eval was estimated at 1 h (experiment-log.md:3507); it took 13 m 54 s (D6, 0.23). The translation eval was estimated 3 h and took 28 m (D9). 8 of 8 evals overestimated.

c. Growing cost per iteration. B03 projected 60 min at iteration 24,000 of 60,769 from the early rate; real time was 146 min because the unroll horizon k ramped (2.43). For solar N=210, k4->30 cost 0.125 s/it but k10->100 cost 0.40 s/it (749 s vs 2395 s, results/solar_stability/*_train.json). A flat s/it extrapolated across a ramp misses by 2-3x.

d. Waits not separated from compute. Rows: C08 "not retrying for another 891 s" (polaris-mcp lockout); it ended about 365 s later, via a manual /mcp reconnect, not natural expiry. C09 "next check about 15:55": a channel timeout cleared only after polaris stop then start, about 12.5 min from the first timeout. C10 "wait 13 s": Duo push plus two banner errors preceded the connect; the cooldown message covered only the gap between attempts. These three have no compute estimate to ratio cleanly and are outside the statistics; they show auth waits run minutes to tens of minutes and are set by the user or a restart, not by a timer. B23: an agent self-estimated 2-3 h of polling on job 3950; harness-measured duration was about 48 min (2.5-4x lower). B14: "up to ~6 h left" was the 10 h limit minus elapsed, not a runtime projection; job 3642 ended at 4 h 34 m. QOSMaxGRESPerUser (one GPU job per user) pends are bounded by the running job's remaining time; v12_queue_wait (0.87) was good because the estimator used v11's remaining ETA. Agents stopping at Duo failures are in the transcripts but have no numeric estimate (C04, excluded).

e. Whole-job vs training-only mismatch. Registry actuals are whole-job elapsed (about 6 min startup and scene generation included); experiment-log numbers are often training-only (4740: 19 min vs 12 m 6 s). A budget of "training 15-30 min" is not comparable to either unless stated.

f. Omitted stages (underestimates): per-job eval (B04), second simulation pass (B22), eval after training (B03/euclid). Small sample (4 rows), but all under.

Not supported: estimator type (user/main/subagent) as a driver. Main-assistant pairs are mostly the measured ones (Sep), subagent pairs mixed; the split is explained by basis, not by who wrote the number.

## 4. Proposed method (proposals only; nothing applied)

1. Measure first. Before any estimate over 30 min, time a 50-100 iteration slice, or take s/it from the nearest same-recipe run in a *_train.json, and multiply by the real iteration count. For a k ramp, weight by the k schedule (cost at final k, not first-slice rate). Evidence: measured training median 0.98 (n=26, IQR 0.89-1.05) vs unmeasured 0.36; B03; solar 0.125 vs 0.40 s/it. If no measurement exists, label the number "guess, 0.3-1.0x of budget" rather than a point value.
2. Scale evals from the last same-harness eval, plus the number of models/scenes ratio. Evidence: 8 of 8 overestimated, median 0.23; 3 m 13 s was already known before the 1 h estimate. Reference below.
3. Estimate three things separately: compute (range), queue wait (bounded by the running job's remaining time at one GPU per user, else unknown), and auth/review wait (unknown; state "unbounded until user acts" and do not poll for it). Evidence: C08-C10, B14, B23, v12_queue_wait.
4. State the scope (whole job vs training-only) and give a range with confidence, e.g. "12-16 min, 80%". Evidence: section 3e.
5. Record estimate and actual: add est_min/est_max/actual columns to docs/debugging/priority-registry.md and an "estimate / actual" line to each experiment-log entry. The registry has actuals in prose today; no column for the estimate, so this review needed transcript mining. Evidence: the 63 rows required reading 14 transcripts.
6. Key the >1 h checkpoint/--resume/SIGTERM rule off the measured projection, not the prereg budget. Evidence: preregs said 1-3 GPU-h and the jobs ran 13-40 min; the opposite risk (B03, 146 min vs 60) also exists.

Judged not worth changing: the existing s/it x iterations projection practice for fixed-k runs (22 of 26 within 0.8-1.25); agent templates already carry "record job id, expected runtime, log path".

### Reference runtimes (lookup; every row has a source above or below)

| job type | runtime | source |
|---|---|---|
| solar training N=210 k4->30, 6000 it | 0.125 s/it, 749 s | results/solar_stability/sss_ctrl_train.json |
| solar training k10->100 | 0.40 s/it, 2395 s | results/solar_stability/sss_long_train.json |
| generalized-systems training | 726-744 s (job 13-20 min incl. startup) | results/solar_generalized/*_train.json; D4/D5 |
| tiered visited-state replay | 92 min (4089), 98 min (4104) | priority-registry.md |
| solar eval, 5 ICs x 3 models | 3 m 13 s (4716) | priority-registry.md |
| generalized-systems eval | 13 m 54 s (4742) | D6 |
| translation eval | 28 m (D9) | D9 |
| moon-situations eval | 22 m 45 s (D11) | D11 |
| BH 1e9 render | 1.4 s/tick; 1e10 13.4 s/tick | 49e2af9c:5178 and rows B18/B19 |
| job startup + scene generation | about 6 min | 4740 (19 m 7 s job vs 12 m 6 s train) |
| fine-tune run | about 9 min each (8:54-9:15) | j2898-j2905 rows |

## 5. Independent review

A fresh unprimed reviewer subagent ran. It recomputed all 63 ratios (0 discrepancies; overall, per-type and per-basis medians and IQRs in section 2 reproduced exactly) and checked ten pairs against sources: B03, B04, B22, B14, B21, B17, B13, C03, D11, j3118 all confirmed as quoted, with these corrections:

- B14 is a remaining-limit bound (10 h minus elapsed), not a runtime forecast; B23 is a retrospective self-report of polling time, not a forecast. Both should be excluded. Recomputed without them (n=58): all median 0.83, IQR 0.29-1.00; measured 0.97 (n=33, IQR 0.86-1.04, geometric mean 0.97); unmeasured 0.27 (n=25, IQR 0.21-0.45, geometric mean 0.31); training median 0.89. Queue-wait type drops to one row (v12_queue_wait 0.87), too few to summarise. Conclusions unchanged.
- C03 is basis "measured" (estimate derived from the 1 h 32 m run of 4089); the table lists it as unmeasured. It is the 1.04 row.
- B04: estimate is training only (114 min), actual includes evals (2.9 h); the 1.53 reflects omitted eval. B21: the 45 min is the main assistant's own agent-prompt guess for the whole task, not user text. B22: actual is a memory-note recollection (336 s), excludes encode time. B13: the 1-59 min range is derived from "well under 1 hour"; treat as a bound. B12 actual is approximate (~7 h). j3397 actual is subagent prose, no sacct line.
- GPU-h budgets (C02, C05, D1, D2) are compared as wall-clock minutes; valid only for one-GPU jobs, which these were.

