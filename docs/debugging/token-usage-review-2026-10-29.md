# Token usage review, Augur project, 2026-10-15 to 2026-10-29 (UTC)

Scope: every transcript under `~/.claude/projects/-home-awebb-Research-Augur/` (25 main-session files, 207 subagent files) with message timestamps on or after 2026-10-15 00:00 UTC. 9,067 deduplicated assistant API calls (deduped by message id; 99 synthetic messages excluded). Other project directories had one 1.7 MB transcript and no other activity in the window, so nothing to compare against. Analysis scripts are in `~/.claude/jobs/8a5257ec/tmp/` (not committed).

## Definitions and weights

- SIT = Sonnet-input-token equivalents: every token category converted to the cost of one fresh input token on the Sonnet-class model. All "cost" numbers below are SIT.
- Weights assumed (standard published ratios, not re-verified online in this review): fresh input 1.0, cache write with 5-minute time-to-live (TTL) 1.25, cache write with 1-hour TTL 2.0, cache read 0.1, output 5.0. Opus assumed 5/3 of Sonnet (assumed). Your real allowance may weight things differently; treat shares as the reliable part and absolute SIT as relative.
- "Miss" = an API call whose cache read fell more than 30% (and more than 5k tokens) below the previous call's total context. In subagents, 442 of 453 such events re-wrote at least 80% of the lost tokens as cache writes, so the detection is sound. In main sessions a drop can also be auto-compaction, so main-session miss counts are an upper bound.
- Token estimates for tool outputs use 3.5 characters per token.

## 1. Totals

Raw tokens: fresh input 18,209; cache write 116,143,916 (5-minute TTL 107,296,595, all in subagents; 1-hour TTL 8,847,321, all in main sessions); cache read 1,360,280,524; output 6,845,678.

Weighted total: 327.46M SIT.

| Category | SIT | Share |
|---|---|---|
| Cache write | 155.0M | 47.3% |
| Cache read | 137.9M | 42.1% |
| Output | 34.5M | 10.5% |
| Fresh input | 0.02M | 0.0% |

Cache writes cost more than cache reads even though reads are 12x more tokens, because 79% of subagent cache-write tokens (84.6M of 107.3M) are re-writes after cache expiry (section 3b).

By model: claude-sonnet-5-5 314.0M SIT (8,804 calls); claude-opus-5-5 13.4M SIT (164 calls, 4.1%, all in 6 subagent runs on 2026-10-15). Effort setting is medium; no Haiku calls in the window.

Main vs subagents: main sessions 41.99M (12.8%, 1,587 calls); subagents 285.48M (87.2%, 7,480 calls, 207 runs). Subagents run on 5-minute TTL cache; main sessions on 1-hour TTL.

By day (UTC, SIT): 10-15 35.2M; 10-16 12.3M; 10-19 9.5M; 10-20 8.5M; 10-21 42.9M; 10-22 23.2M; 10-23 10.6M; 10-24 35.0M; 10-25 41.9M; 10-26 64.0M; 10-27 19.4M; 10-29 22.5M; 10-30 (UTC rollover of the evening of 10-29 PDT) 2.3M. No activity 10-17, 10-18, 10-28. Peak day 10-26 is the overnight run of the 10-24 agent family plus the supervisor/video agents.

## 2. Top sessions and subagent runs

A session's family = its own turns plus every subagent it spawned. 26 families.

| Session | Main | Subagents (count) | Share of total | Doing |
|---|---|---|---|---|
| 49e2af9c (10-24 to 10-29, one 5-day session) | 18.5M | 143.2M (69) | 49.4% | Heterogeneous tiered field: 5 parallel research agents, rate-limit stop, 5 replacements, supervisor-log, videos, loss analysis |
| a5d11050 (10-15/16) | 2.7M | 22.8M (12) | 7.8% | Contact dE harness, lr-warmup fix, few-shot adapter |
| 65480f12 (10-22/23) | 2.6M | 22.5M (16) | 7.7% | Drift fixes A/B, near-field retrain, tiled page |
| 801bda9c (10-20/21) | 1.6M | 22.0M (7) | 7.2% | Scatter grid-edge artifact, kmlp far kernel |
| 3e784190 (10-21/22) | 1.4M | 20.2M (9) | 6.6% | Tiled rerun, website model page, 1B galaxy sim |
| ec5b8f43 (10-29) | 1.1M | 9.2M (6) | 3.1% | Neural circuit smoke test, tiered diagnostics |

Top subagent runs (all general-purpose on Sonnet, turns = API calls):

| Run | Turns | SIT | Peak context |
|---|---|---|---|
| Tier-k lever continuation (C2) | 421 | 25.9M | 363k |
| Tier-0 extension continuation (E2) | 439 | 25.4M | 485k |
| Hole-1 continuation (A2) | 266 | 16.5M | 365k |
| Build tiered data and eval suite | 180 | 9.7M | 494k |
| Implement tier k>=1 core module | 241 | 9.3M | 393k |
| Fix scatter grid-edge force artifact | 188 | 9.0M | 401k |
| Fix kmlp far kernel | 244 | 8.0M | 406k |

Concentration: 17 subagent runs with more than 100 turns cost 154.3M (47.1% of everything); 45 runs with 41-100 turns cost 97.1M (29.7%); the other 145 runs (under 41 turns, including 84 with 5 turns or fewer) cost 34.0M (10.4%).

## 3. Waste patterns

### (a) Long contexts re-read many times

- Subagent first-turn context is 35k to 39k tokens (median 36.8k); by the end the big runs sit at 360k to 494k.
- Calls with context over 200k: 2,534 in subagents and 230 in main sessions; they carry 158.7M SIT (48.5% of all cost). Input-side cost attributable to tokens above 200k: 37.7M (11.5%); above 250k: 18.0M (5.5%).
- Turns beyond turn 100 of a subagent: 91.0M (27.8%); beyond turn 150: 60.6M (18.5%).
- `settings.json` sets `autoCompactWindow` to 400000 for claude-sonnet-5-5. Subagents peaked at 485k to 494k, so in practice the setting did not cap them (whether it applies to subagents at all is not verified).
- Main session 49e2af9c ran 456 calls over five days with median context 160k and peak 362k.

### (b) Repeated cache creation (cache expiry)

This is the largest single waste.

- Subagents: 453 miss events in 63 runs; 86.7M tokens re-written at 1.25 instead of read at 0.1, an excess of 99.7M SIT (30.4% of all cost). Average re-write per event is 191k tokens.
- Gap before the re-write: 403 events had gaps of 5 to 60 minutes (77.1M tokens), 38 over 60 minutes (7.0M), 12 under 5 minutes (2.5M).
- What preceded the gap: `polaris_wait_for_pattern` alone 265 events (50.0M tokens); a Bash call plus a Polaris wait 25 events (5.4M); Bash `sleep`/`until` loops about 49 events (about 9.8M); no tool call at all (agent resumed by SendMessage after idling) 58 events (10.9M).
- Gap histogram clusters at 28 to 30+ minutes (151 events). The Polaris wait tool aborts after about 1,800 s with no response (seen in error text), and 740 of 1,130 wait calls passed no timeout.
- Worst runs: C2 70 events (14.2M tokens), E2 50 (11.4M), A2 44 (8.8M).
- Main sessions: 43 events, 6.6M tokens (26 of them over 1 hour idle, 3.9M tokens). Extra cost at the 2.0 write weight is about 12.5M SIT (3.8%); some of these are compaction rather than expiry.
- The 1-hour TTL is already in use for main sessions only (main cache writes are all 1-hour, subagent writes are all 5-minute). That mismatch is the key fact.

### (c) Large tool outputs entering context

Cost counts the one-time write plus 0.1 per remaining turn of that agent.

- All tool results: 51.1M SIT (15.6%). Bash 31.6M (9.7%, 4,731 calls, 13.9 MB); `polaris_ls` 8.7M (2.7%, 293 calls, 3.1 MB); Read 3.7M (1.1%); Artifact 2.7M (0.8%); `polaris_job_logs` 1.7M (0.5%).
- By size: 295 results of 5k-15k tokens cost 19.9M (6.1%); 1,159 results of 1k-5k tokens cost 20.4M (6.2%); results under 1k tokens (9,809) cost 10.4M (3.2%). Results of 8k tokens or more: 137, costing 11.4M (3.5%).
- Worst single items: `polaris_ls` returning about 12k tokens, called repeatedly inside the same 400-turn agent (each stays in context for 150 to 417 more turns, 0.2M to 0.5M SIT each); `polaris_job_logs` 11.3k tokens; Artifact publish refusals that hand back the whole live page (about 14k tokens each).
- Subagent output tokens are mostly tool-call inputs and thinking: of 6.85M output tokens, about 2.33M are thinking (2.15M in subagents), about 1.96M are tool-call inputs (Bash 0.97M, Write 0.58M).

### (d) Repeated reads of the same file

Negligible: 35 repeat Reads of an already-read path inside one transcript, 0.77M SIT (0.2%). Largest repeats: a scratchpad `human_trim.md` 8 times, `supervisor-log.md` 4 times. Only 3 identical large tool results (same call, same output) in the window. Not worth a rule.

### (e) Subagents: small tasks, duplicates, model

- 207 runs, all `general-purpose` except 3 unnamed, 2 `claude`, 1 `fork`. All inherit your model; 6 runs (all on 10-15) used Opus: 13.4M SIT (4.1%), design/loss-engineering/substep research.
- 84 runs of 5 turns or fewer cost 4.5M (1.4%), almost all your required reviews: 48 unbiased frame-review runs 2.16M (0.66%); 15 prompting-review runs 1.00M (0.31%). These are cheap; the rules are not a cost problem.
- Supervisor-log republish agents (9 runs, 147 turns): 4.7M (1.4%). The supervisor-log/artifact bucket as a whole (13 runs including restyle and page redesign) is 20.0M (6.1%).
- Overlap: on 10-24 five agents (A, C, D, E, F) were launched in parallel, hit the rate limit (HTTP 429, session limit) at 23:34 UTC, then were stopped by you at 23:41 after the `rm -rf` scare. Stopped agents cannot be resumed, so 5 replacements (A2, C2, D2, E2, F2) started cold with a handoff. The five predecessors cost 30.6M; the four continuations cost 72.4M (22.1% of all cost) and ran 123 to 439 turns each. Duplicate work cost was not separately measured.
- Main-session turns triggered by task notifications: 313, about 22% of main-session cost (about 9M SIT, 2.8% overall). Main also issued 195 SendMessage calls to resume agents.

### (f) Failed calls, polling, noop turns

- Failed tool calls: 414 (Bash 247, `polaris_wait_for_pattern` 55, WebFetch 43, Artifact 15), 306k characters (about 87k tokens): under 0.1%. Not a waste source.
- `polaris_wait_for_pattern`: 1,130 calls with results, 651 timed out (58%), 407 matched, 56 errored. The turn after a wait costs 87.6M SIT in total across 826 subagent turns (26.8% of all cost), mostly the cache re-write in (b).
- Bash `sleep`/`until` polling in subagents: 151 follow-up turns, 11.2M (3.4%).
- 113 tool results were SSH/Duo connection failures (the 10-24 Polaris disconnect and "Error reading SSH protocol banner" loops); small in tokens.
- `ScheduleWakeup`: 11 calls in main sessions; wakeup/loop noop turns are not a material source.

### (g) Verbose assistant output

Not a source. Only 41 assistant messages (39 in subagents, 2 in main) exceed 4,000 characters of prose, holding about 110k of 6.85M output tokens (1.6%). Output is 10.5% of cost, and about two thirds of it is thinking plus code written through tool calls.

### (h) Startup overhead (injected context)

Measured sizes (characters; tokens estimated at 4 characters per token):

- Global `CLAUDE.md` 8,978 bytes plus `MEMORY.md` index 14,201 bytes (105 lines, 98 files in the memory directory, 3.7 MB, only the index is injected): about 5.8k tokens (the injected "instructions" block is 21k to 22k characters in subagents).
- Skill listing 22,125 characters: about 5.5k tokens, injected in every main session and every subagent. Skill use is tiny: across all history 12 `superpowers:brainstorming`, 11 `artifact-design`, 8 `systematic-debugging`, 7 `writing-plans`, 6 `frontend-design`, all others at most 5.
- Deferred-tool list 5.8k characters (about 1.5k tokens); superpowers SessionStart text 3.7k characters (about 0.9k tokens, main sessions only); MCP instructions 2k-4k characters; agent listing 3k-6k characters.
- First-turn context: subagents median 36.8k (injected, trimmable part about 12k; the other 25k is system prompt and built-in tool schemas); main sessions about 43k-55k.
- Subagent first turns: 5.75M cache-written and 3.67M cache-read, so about 60% of startup prefixes are re-written, not shared (launches are staggered more than 5 minutes apart or prefixes differ).
- Rule of thumb from this window: every 1,000 tokens removed from the always-injected prefix saves about 1.7M SIT (0.5% of total) over 14 days (9,067 calls read it at 0.1, 496 misses re-write it, about 230 session starts write it).
- The remember plugin runs nested `claude -p` summarizer sessions: 139 `tmp-remember-summarizer-cwd-*` directories were created between 10-23 and 10-29 (64 on 10-26, 30 on 10-29), containing no transcripts, so their token use cannot be measured here. Each is a full Claude call with its own startup prefix; the plugin comments say it defaults to a smaller model.

## 4. Highest-leverage changes, ranked by estimated SIT saved

Savings are estimates, overlap between items exists, and none requires dropping a standing rule.

1. Stop paying 5-minute cache expiry during subagent waits. Evidence: 99.7M excess (30.4%) in 453 events; 26.8% of cost is the turn after a Polaris wait. Actions, in order of preference: (a) check whether subagents can run on the 1-hour TTL as main sessions already do (data shows main = 1-hour only, subagents = 5-minute only; not verified that a setting controls it). Estimated net saving about 66M (20%) after the 1-hour write premium on new content. (b) If not possible, have agent prompts pass a wait timeout of 270 s or less and loop, which keeps the cache warm (about 0.1 x context per turn instead of a 1.25 x context re-write per 30-minute wait); estimated saving about 30M (9%). Prefer (a). This does not conflict with "arm a wait tool": the wait is kept, only its cache cost changes.
2. Cap agent length and context. Evidence: turns beyond 150 cost 60.6M (18.5%), context above 200k costs 37.7M (11.5%), runs peaked at 494k. Actions: lower `autoCompactWindow` from 400000 toward 200000-250000 (upper bound 18M-38M SIT; realistic about half after compaction cost) and tell long-running agents in the saved prompt templates to finish at a stage boundary (about 120 turns) and report, so the next stage starts from a 37k prefix plus a handoff instead of a 300k+ context. Estimated saving 20M-35M (6%-11%). Trade-off: lost detail after compaction; your call.
3. Make resumption cheap after an unplanned stop (caused by the 10-24 rate limit and the user stop, not by a rule). Evidence: the four continuations cost 72.4M (22.1%), versus 30.6M for the work they continued. The fix is in the handoff: one narrower scope per replacement with an explicit end stage (item 2), not re-running full plans. Saving depends on the next stop; the cost of one such event was about 72M.
4. Cut large tool outputs at the source. Evidence: 137 results of 8k tokens or more cost 11.4M; `polaris_ls` alone 8.7M. Actions: in agent prompts require `polaris_ls` on a narrow path or `ls | tail -n 30` style filters, `polaris_job_logs` with a line limit, and Artifact publish that does not re-read the live page after a refusal. Estimated saving 8M-12M (2.5%-3.5%).
5. Trim the always-injected prefix (your config choice). Evidence: 1.7M SIT per 1k tokens. Actions: reduce the skill listing from about 5.5k to about 1.5k tokens (disable plugin skills you do not call, such as unused superpowers/anthropic-skills entries; usage counts above), and shorten the `MEMORY.md` index from 14.2 KB toward 7 KB by moving old dated entries (2026-10-02 to 10-12 sessions, CFD archival block) into the topic files. Estimated saving 8M-10M (about 3%). Superpowers hook text is under 1k tokens; not worth touching on its own.
6. Review the remember plugin's nested summarizer calls. 139 directories in six days, token cost not visible. If it runs on every session end and tool batch, restrict it or confirm it uses the small model; saving unknown, possibly several percent. Needs measuring from the account usage page, not from transcripts.
7. Opus for research agents only when needed. 13.4M (4.1%) on 10-15, none since 10-19. No action unless it recurs. Haiku for the frame-review agents would save at most about 1.4M (0.4%) and risks the unbiased read; not recommended.

## 5. Attribution

User prompting and configuration choices: 400k auto-compact window; 5 parallel agents on a nearly exhausted session allowance; stopping agents (stopped agents cannot resume, forcing cold restarts); long unbroken agent assignments (17 runs over 100 turns hold 47% of cost); large injected skill listing and `MEMORY.md` index; remember plugin enabled.

Assistant execution failures: 30-minute blocking Polaris waits inside agents with a 5-minute cache; repeated `polaris_ls` and log reads of 9k-14k tokens kept in context for hundreds of turns; Artifact refusals re-reading the full page; main-session `SendMessage` resumptions of idle agents. Failed tool calls, repeated file reads and verbose prose are not material.

Costs caused by your standing rules, for your decision (none recommended for removal): unbiased frame reviews 2.16M (0.66%); prompting reviews 1.00M (0.31%); supervisor-log republish agents 4.7M (1.4%) plus Artifact tool results 2.7M (0.8%); agent-native heavy compute is what drives the Polaris waits in item 1 (so the rule's real cost is the cache expiry, 99.7M, not the waiting).

## 6. Caveats

- SIT weights are assumed ratios; Opus weight assumed.
- UTC days.
- Main-session miss counts include compaction.
- Subagent transcripts do not show thinking text length reliably (counted from usage detail).
- `settings.json` also holds a plaintext `PUSHBULLET_API_KEY` in the `env` block; unrelated to tokens, flagged as an anomaly.
