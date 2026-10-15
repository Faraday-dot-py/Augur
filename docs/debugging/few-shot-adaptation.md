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

## Part 2 results

(pending)
