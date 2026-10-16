# Few-shot / one-shot adaptation and state assimilation (2026-10-15)

Question (user, 2026-10-13): have we tested one-shot / few-shot adaptation (a handful of fine-tune steps or scenes per new geometry or scene type, or conditioning on a few observed frames), and might it help? "Exp E" = checkpoint `E_ms_kp_pot_v_g128` (gravity only, pp=2, frozen). dE = relative energy drift; nr100 = 100 unit-mass bodies, 1000 ticks; err@k = mean per-body position error (sim units) at tick k; K = reset period in ticks.

## Part 1. Prior work and verdict (no GPU)

Searched experiment-log.md, error-correction-{plan,literature,unet,codebase-proposals}.md, error-growth-diagnostic.md, long-rollout-root-cause.md, memory dir. Terms: few-shot, one-shot, in-context, adaptation, assimilation, truth-reset.

Never run as such: no few-shot or in-context experiment exists. Nearest results:
- Gradient adaptation, many-scene scale: fine-tunes on scene mixes (orbit_ft2, orbit_full, bounce ft, planetary ft; scatter-scenes.md). orbit_ft2 loses 25% of planets on the planetary benchmark vs 100% zero-shot E, so adaptation helps the solar failure. Cost: every U-Net weight update (E1, fix-finetune-dE) takes nr100 dE@1000 .02 -> .2-.5 within ~25 steps (lr 1e-4 and 1e-5, seeds 4738/4739); soft dE loss, warmup, EMA, weight interpolation (25% already .063) all fail. A few-step version of this is therefore as bad as the many-step one on the dE gate.
- Heads-only adaptation (kmlp/ppmlp, U-Net frozen): passes nr100 dE (.0218 vs .0170) but no accuracy gain, fails nr100 dL, bh300 dL and dE@100. Rejected by user (D1, 2026-10-13).
- Parameter-isolated adaptation: B-frozen (learned SDF geometry branch on fully frozen E) is stable (tunnel 0%, runaway 0) but over-damped. This is the only adapter-style route that leaves E untouched; a few-shot variant (fit the branch on a handful of scenes of a new geometry) was never tried.
- Observation conditioning: the model state is Markov in pos/vel (error-growth-diagnostic.md): zeroing the recurrent field every 10 ticks changes err by +10%, the norec ablation is within 5% of E, so there is no hidden state in which a context could live. Resetting pos/vel to truth every 5 ticks gives ~200x lower err@100 (expB .0070, cluster .019, star .105 vs free 1.49/4.14/27.6); K=10: .018/.048; K=20: .047/.165 at tick 100 (sawtooth, tick 100 is a reset tick for K dividing 100). D4 concluded "only state assimilation from observations would give the near-field gain" and left it as follow-up #5 (needs a deployment with observations). The learned relabel curriculum (truth windows in training) was negative; Kalman and parareal were rejected earlier without a recorded probe.
- Literature (error-correction-literature.md 1.10): nudging / data assimilation noted, novelty none, requires reference data at deployment.

Verdict on viability under the standing constraints:
1. Gradient few-shot on U-Net weights: not viable (dE gate breaks in ~25 steps; heads-only rejected).
2. Few-shot fit of an isolated branch or adapter on frozen E: viable in principle (O(N), pp=2, learned not hard-coded), untested, and only for geometry/contact-type gaps (B-frozen is the template). It does not address solar or mass ratio, where the missing physics is the pair force at large mass contrast / near-field, which a handful of scenes cannot teach a frozen U-Net without weight updates.
3. In-context conditioning on observed frames: needs a new context-encoder input (architecture change); no hidden state exists today. Not a free probe.
4. Inference-time state assimilation (observing truth pos/vel): requires no weights, keeps dE (truth resets re-anchor energy but E itself is untouched), O(N). Whether it helps the OOD failures at sparse observation rates is what Part 2 measures. Honest framing: this is an observation-fed hybrid, not a model improvement; it answers "how much outside data is needed to keep E on track", relevant only for deployments with observations.

## Part 2. Probe design (recorded before launch)

Script `scripts/scatter_assim_probe.py`; single Polaris job, seed 4738 for the solar IC, eval seeds 9200 (star) and 9300 (nr100). Frozen E, Verlet integrator as `ScatterField._step`; at an observation tick the observed bodies' pos (and vel) are overwritten with the truth state and a_prev is recomputed at that state.
Scenes: solar (star mass 30, 8 planets, moon, 200 belt particles; hill-k 4.0 IC of job 3507; 1000 ticks); star family (heavy body mass 8-30 + light 0.01-0.5, mass ratio up to 3000:1; 12 scenes, 150 ticks); nr100 (4 scenes, 1000 ticks, dE). A contact scene is not run: E has no wall/contact terms (zero-shot walls fail by construction) and the contact models are other checkpoints.
Arms: free; full-state reset K = 5, 10, 20, 50; first-m only (reset every tick for ticks < m, m = 5, 20); K=10 position-only (velocity unobserved); K=10 and K=20 observing only the heavy bodies (solar: star + 8 planets = 9 of 210; star family: heaviest ~10%; nr100: random ~10%); K=10 observing the single heaviest body.
Metric: mean per-body error over a tick window (11-20, 81-100, 281-300, 981-1000) so the reset sawtooth does not hide the reading; solar adds planets lost / period error / dE, dL at ticks 100, 300, 1000 (solar_metrics from scatter_solar.py); dE vs truth over ticks 951-999 for the 1000-tick families.

Predicted outcome (one line): K<=20 full reset gives >=5x lower window err@100 on solar and star; K=20 heavy-only gives 2-4x on solar (planets are anchored, belt/moon drift); first-m only gives <1.3x (chaos plus OOD bias, not an IC problem); position-only K=10 clearly worse than full-state.
Pass: K=20 full reset reduces window err@100 by >=2x on BOTH solar and star, and solar planets lost at tick 300 drops below the free arm. Kill (assimilation is not worth pursuing as a sparse scheme): K>=20 gives <2x err@100 reduction on solar. Heavy-only arms are exploratory, no gate.

## Part 2 results (job 3643, 5 min, frozen E, seed 4738; raw results/assim/assim.json, table results/assim/summary.txt via scripts/scatter_assim_summarize.py)

Window err (mean per-body position error, ticks 81-100) and ratio vs free. Windows 11-20 are uninformative for K>=20 (first reset at tick 20). Single solar scene; star 16 scenes' worth (6,6,4 shown), nr100 4 scenes.

| arm (obs/body/tick) | solar err@100 (x) | solar planets lost @300 | star err@100 (x) | nr100 err@100 / @1000 (x) |
|---|---|---|---|---|
| free | 41.6 | 100% | 26.5 | 1.09 / 35.7 |
| K5 (.2) | .034 (1217x) | 0% | .050 (529x) | .0049 (222x) / .0050 (7078x) |
| K10 (.1) | .122 (340x) | 0% | .179 (148x) | .010 (109x) / .012 (2986x) |
| K20 (.05) | .486 (86x) | 0% | .70 (38x) | .0235 (46x) / .037 (962x) |
| K50 (.02) | 7.9 (5x) | 50% | 7.5 (3.5x) | .157 (7x) / .379 (94x) |
| first5 / first20 | 1.2x / 1.9x | 100% | 1.1x / 1.7x | 1.05x / 1.7x (no effect by tick 300) |
| K10 pos-only | 8.6x | 100% (period err .71) | 6.2x | 20x / 35x |
| K10 heavy-only (9 of 210; star: ~10%) | 1.65x err, but 0% planets lost to tick 1000, period err .001-.005 | 0% | 0.87x (worse) | 1.15x |
| K20 heavy-only | 1.57x, 0% lost to 1000 | 0% | 0.82x | 1.1x |
| K10 star-only (1 body) | 1.6x, 75-100% lost | 100% | 0.86x | 1.04x |

dE vs truth (last 50 ticks, relative): solar free 2.6e3 vs K5 .022, K20 .25, K10 heavy-only 400; nr100 free .052 vs K20 .0039.

Verdict against pre-registered criteria:
- Pass: K=20 full reset cuts window err@100 86x on solar and 38x on star, solar planets lost at tick 300 0% vs 100%. Kill criterion (K>=20 <2x on solar) not triggered. Prediction "K<=20 >=5x": confirmed (far exceeded). "first-m only <1.3x": confirmed at err@100 for first5 (1.2x), missed for first20 (1.9x; still no effect by tick 300, planets all lost). "pos-only clearly worse than full-state": confirmed (K10 pos-only 8.6x vs 340x). "K20 heavy-only 2-4x on solar": err 1.6x (belt/moon still diverge, missed low) but the planets themselves are retained 8/8 to tick 1000 with period error .005, better than predicted.
- Sparse observation works down to K=20 (5% of body-ticks) and degrades sharply by K=50 (solar 50% planets lost, i.e. one chaos time per window is too long).
- Information needed is velocity as well as position, and has to be continuous: initial-condition-only observation (first-m) is worthless, confirming that the failure is accumulated force error, not a bad start.
- Heavy-body-only observation is a real finding on solar: 9 of 210 bodies at K=10/20 (0.2-0.4% of body-ticks) keeps all planets for 1000 ticks, since planets are driven by star + mutual forces and light bodies only matter for belt/moon. It does not transfer to the mass-ratio star family (worse than free, 0.8-0.9x: the heavy star is observed but light-body errors are not fixed) or to nr100 (random 10% subset: no effect). A single star observation alone fails.
- Caveats: one solar scene (seed 4738), 16 star scenes, 4 nr100; truth resets are an oracle observation (noise-free, exact); no noisy-observation arm; err mean for solar is dominated by belt (200 test particles) so the heavy-only err ratio understates planet retention. Resets bound dE by re-anchoring, E itself is unchanged.

Net answer to the question: gradient few-shot is not viable and conditioning on a few initial frames does not help. Sparse continuous state assimilation helps enormously (including on the solar OOD failure) but is an external-observation scheme, not a model improvement; it only matters for deployments that observe the system. Next step if wanted: noisy observations (sigma ~ real sensor) and heavy-body-only on more solar ICs.
