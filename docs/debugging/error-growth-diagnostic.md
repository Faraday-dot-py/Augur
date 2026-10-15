# Error-growth diagnostic, scatter field Exp E (2026-10-12, uncommitted)

Model: Exp E `E_ms_kp_pot_v_g128` (checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt, grid 128, extent 64, h 0.5, Verlet + a_prev, momfix, no far-field flag). Seed 4738 for envelope scenes, eval seeds 9000 (expB), 9100 (cluster), 9200 (star). dt .1, eps .5, 150 ticks. Truth = own fp64 KDK (4 substeps), asserted equal to gs.rollout_torch. Scripts (uncommitted): `scripts/scatter_error_growth.py` (gravity, items 1,2,3,4), `scripts/scatter_error_growth_contact.py` (item 3, option-C runaway). No code changed otherwise. Definitions: err = per-body Euclidean position error |x_model - x_truth|, pooled mean over bodies and scenes; N = number of bodies; "one-step" (TF) = state reset to truth every tick, one Verlet step, compared to truth at t+1 (recurrent field and a_prev carried).

## Premise check (flag)
The 17 -> 1150 a0 runaway at ticks 21-24 was measured on the option-C CONTACT model (E fine-tuned, k_end 20) in box/obstacle scenes (job 3496), not on E. scatter-contact-c.md: "E zero-shot is stable to 100 ticks on gravity scenes" and E "does not blow up in 1e4 ticks". So item 3 is run two ways: (a) E on gravity scenes (does E's input leave the training envelope; is there any runaway), (b) the C checkpoint on c_id scenes (the actual runaway), with the same input statistics. E zero-shot on contact scenes was not probed (E has no geometry input; its contact failure is tunnelling, not a runaway).

## Predictions (written before any result)
- P1 (error vs horizon). Free-rollout error grows roughly exponentially from tick ~5 then saturates near the chaos ceiling; TF one-step error is small and flat in t (no horizon dependence), larger for bodies with nn-distance < 1 and for the grid-edge band (26-32), smaller for heavy bodies in star. Prior numbers: expB .024@20 -> 1.4@100.
- P2 (oracle). exact force every tick with the model integrator gives err ~1e-3 or less (integrator floor). Reset-to-truth every K gives a sawtooth with per-segment growth the same as the free curve at the same age (error is injected locally and amplified, not stored in the recurrent field), so `fieldreset10` is within 20% of `free`. Of the spatial-class oracles, `near` (nn<1) removes the largest share in expB/cluster (>50% of err@50); `edge` matters only in cluster (bodies reach the band) (<20%); `iso` little.
- P3 (chaos vs drift). Exact fp64 sim with state noise equal to the measured model one-step error (x1) reproduces the model's free curve to within ~3x through tick 50 (error is chaos amplifying a seed of that size); the model exceeds it (>1.5x) at ticks 100+ because the model error is systematic (bias), not white noise. Noise x2 brackets it.
- P4 (OOD). On gravity scenes, E's grid-net inputs and outputs stay inside the train-seed envelope in expB and cluster (<5% of scene-ticks above envelope max), no runaway of |a|; star (heavy masses, OOD) exceeds the envelope in rho/mom. For C on c_id: the recurrent field channel (fld_max) and gacc_max exceed their tick 0-20 envelope first (positive feedback), then the a0 runaway follows; input density/momentum channels stay in range until balls are flung out. The max grid acceleration first appears at cells adjacent to a ball (nearest ball < 2 units), not at walls.
- P5 (noise sensitivity). IC position noise at 1e-5 L grows with a Lyapunov exponent for the model within 0.7-1.5x of the exact dynamics; the model does not amplify much faster. Larger noise (1e-3 L) saturates sooner for both.

## Runs
| job | what | output |
|---|---|---|
| 3508 | smoke (2 scenes) | reproduced the contact runaway a0 17.4 -> 65.4 -> 1149.9 (scene 0, ticks 20/22/24); truth reimplementation max diff 5e-9 |
| 3509 | gravity suite v1 (E; expB 24 / cluster 12 / star 12 scenes) + C probe on 24 c_id scenes | results/errgrowth/full_gravity.json, full_contact_C.json |
| 3510 | gravity suite v2 (adds exact-init / warm-start variants) + causal follow-ups on C (gravity scenes, field reset, field clamp) | results/errgrowth/r2_gravity.json, r2_contact2_job3510.log |
Tables below are from 3510 (gravity numbers identical to 3509 for shared variants). Summaries: `scripts/scatter_error_growth_summarize.py`, `_contact_summarize.py`, `_front.py`. No render/video was produced, so no image review was run. GPU time: ~3 jobs, ~25 min wallclock total.

## 1. Error vs horizon (E, free rollout from truth IC; mean per-body position error; TF = one step from truth state)
| family | err@5 | @20 | @50 | @100 | @150 | TF one-step pos (t=5 / 50 / 100) | TF one-step vel (mean) |
|---|---|---|---|---|---|---|---|
| expB (N 11-98) | .0034 | .026 | .17 | 1.49 | 4.58 | 2.8e-4 / 6.3e-4 / 1.2e-3 | .0081 |
| cluster (N 104-233) | .0063 | .057 | .68 | 4.14 | 7.11 | 8e-4 / 4.4e-3 / 3.2e-3 | .038 |
| star (OOD masses) | .124 | 1.54 | 8.29 | 27.6 | 63.3 | (heavy vel 3e-3, light vel .07) | |
- One-step error is flat in t (expB 3e-4..1e-3, no horizon dependence); the free-rollout curve is that seed amplified ~x1500 by tick 100. Growth ~ exp(0.6 t) for expB, 0.7-0.8 for cluster.
- By class (expB, free): nn<1 bodies err 0.036@20 vs 0.0205 for nn>=1 (1.8x); one-step error nn<1 is 4-10x nn>=1 (tf_ep@50 1.2e-3 vs 1.6e-4). Edge band 26-32 and r>=32 hold <2% of bodies before tick 75, one-step error there is not larger in position (2e-5..1.8e-3).
- Force error at truth states (model vs exact, relative): ~7-10% for interior bodies at every tick (expB), 5-14% cluster; near- vs far-dominated bodies equal (.07 vs .08). Edge band r 28-32: 14x (expB, t=100), 1-5x (cluster): the known edge artifact; r>=32: ~1.0 (zero force, as designed). Absolute force error in cluster core (r<16) is the largest (0.3).
- star (E never trained on masses 0.01-30): force error 8-11x the exact force (heavy 13x), err@1 already .0128; light bodies carry the error (heavy .077@20, light 1.8).
- Mass split only exists in star (OOD): see above. expB and cluster are unit mass.

## 2. Error-source decomposition (oracle runs; err@20 / @50 / @100, expB then cluster)
| variant | expB | cluster | star |
|---|---|---|---|
| free | .026 / .17 / 1.49 | .057 / .68 / 4.14 | 1.54 / 8.3 / 27.6 |
| reset pos+vel to truth every 5 | .0042 / .0053 / .0070 | .018 / .026 / .019 | .069 / .085 / .105 |
| reset every 10 | .011 / .012 / .018 | .032 / .064 / .048 | |
| reset every 20 | .026 / .012 / .047 | .057 / .065 / .165 | |
| zero the recurrent field every 10 | .029 / .20 / 1.64 | .063 / .77 / 4.34 | |
| exact a_prev at tick 0 only | .031 / .19 / 1.58 | .066 / .71 / 4.24 | 1.24 / 7.7 / 27.5 |
| warm field (3 force evals at IC) | .027 / .18 / 1.49 | .059 / .70 / 4.15 | 1.27 / 7.7 / 28 |
| exact force every tick (model a_prev at tick 0) | .017 / .075 / .88 | .045 / .59 / 3.84 | .26 / .67 / 1.53 |
| exact force every tick + exact a_prev | .0025 / .014 / .26 | .022 / .53 / 3.55 | .0002 / .0015 / .003 |
| exact force for one tick every 10 | .029 / .18 / 1.54 | .063 / .70 / 4.19 | |
| exact force for bodies with nn<1 | .016 / .084 / .96 | .041 / .61 / 3.85 | 1.54 / 8.3 / 27.2 |
| exact force for nn>=1 | .026 / .17 / 1.48 | .057 / .69 / 4.05 | |
| exact force for edge (|x|>=26) | .026 / .173 / 1.49 | .057 / .68 / 4.14 | |
- State is Markov: resetting pos/vel to truth bounds error at ~per-segment growth; zeroing the recurrent field changes nothing (+10%). Error lives in positions/velocities, not in the medium state. A reset every 5 ticks holds err@100 at .007 (expB), .019 (cluster) = 200x lower.
- Error is a sum of independent seeds each amplified by chaos: removing the a_prev init alone or the one-tick oracle does nothing; exact force everywhere helps (expB 12x @20) only together with exact init (.017 -> .0025). Cluster: exact force + init still .022@20 (integrator floor in dense scenes, consistent with scatter-chaos-floors.md).
- By spatial class: bodies with nn<1 account for ~60% of what exact force can remove in expB (.016 vs .017 for exact-all), ~all of it in cluster (.041 vs .045). nn>=1 and edge: zero effect on trajectory error.
- Chaos vs model drift: exact fp64 sim with white state noise equal to the measured mean one-step error (pos .00079 / vel .0081 expB; .0036/.038 cluster) gives expB .037/.20/1.54/4.58 (x1) vs model .026/.17/1.49/4.58 at ticks 20/50/100/150; cluster .17/1.22/5.18/8.5 (x1) vs model .057/.68/4.14/7.1; x0.5 brackets from below. The model is no worse than white noise of its own one-step size; no extra systematic drift is visible in position error.

## 4. Noise sensitivity (IC position noise at tick 0, |perturbed - unperturbed| trajectories, pooled; Lyapunov fit per time unit)
| family | rel noise (x rms radius) | exact lyap | model lyap | model/exact amplitude @20 | @100 |
|---|---|---|---|---|---|
| expB | 1e-6 | .65 | .58 | 10.4x | 21x |
| expB | 1e-5 | .69 | .62 | 1.9x | 2.4x |
| expB | 1e-4 | .62 | .63 | 1.05x | 1.27x |
| expB | 1e-3 | .46 | .46 | 1.0x | 1.05x |
| cluster | 1e-5 | .82 | .71 | 2.0x | 1.8x |
| star (OOD) | 1e-5 | not chaotic (~x4 growth over 150 ticks) | .41 | 17x @20 | 160x |
- For chaotic scenes the model's asymptotic growth rate equals the exact one (slightly lower). The model amplifies small perturbations ~2x (1e-5) to 10x (1e-6) more than exact in the first ~20 ticks (a local sensitivity floor from the net, visible only when the perturbation is below the model's one-step error ~1e-3), then follows the exact rate. At noise >= 1e-4 L, model == exact.
- star: E amplifies perturbations x1000 while exact orbits do not: model-specific instability, but outside E's training distribution.

## 3. Input-OOD check
Gravity, E (envelope = 8 train-seed-4738 scenes teacher-forced, 150 ticks; eval scenes 9000/9100/9200):
- expB: density, momentum, potential channel, |a| all inside envelope (frac of scene-ticks above envelope max: 0-1%); only ak_max (kernel gradient, at a fixed grid corner) 4%. |a| max 3.4-8.4 throughout, no runaway to 150 ticks.
- cluster: 1-10% of scene-ticks above envelope max in ticks 11-100 for density / force (the eval has 12 scenes vs 8 envelope scenes, so part is sampling), |a| max stays 27-45 (env 23), no runaway, field channel 52-53 (env 52).
- star (mass OOD): momentum channel exceeds envelope in 22-65% of scene-ticks after tick 10 (light bodies flung out by the force error: consequence not cause).
- In E the maximum grid acceleration is always at the grid edge (cell radius 31.8, 14.6 vs body |a| max 7 in expB, 52 vs 23-45 in cluster): the edge artifact dominates the grid field maximum but is not reaching bodies.
- The "17 -> 1150 at ticks 21-24" runaway does not occur in E (above). It occurs in C. Option-C probe (24 c_id scenes, 60 ticks, geometry-free a0 pass; envelope = ticks 0-20 of the same scenes):
  - 18/24 scenes run away (a0 > 3x the in-horizon max 41) at ticks 21-36 (median ~27); the other 6 are scenes in which the field stays ~0.
  - The recurrent potential channel (input channel 5) starts at 0 and grows ~x3.5 per 3 ticks (100 at t3, 1.4e3 at t9, 1e4 at t15, 2-5e4 at t40) in every runaway scene. For comparison E on gravity: 14-52 at all ticks. The max |grad phi| cell is NOT adjacent to a ball: it starts at the grid edge in solid/empty region (sdf -20..-36, nearest ball 20-40 units) and travels toward the balls at ~1 unit/tick (e.g. scene 0: nearest-ball 39.9 -> 22 -> 15 -> 11 (t18) -> 9.6 (t21) -> 6.3 (t24, runaway)). The a0 spike happens when the front is within ~6-12 units of the nearest ball.
  - Density and kernel inputs never leave range (rho_max 2-6 vs env 7.1, ak_max). Momentum channel exceeds its envelope only after the runaway (ticks 25-40) = consequence. Potential channel exceeds the tick 0-20 envelope after a0 in most scenes (a0 first at 21-35; field at 21-50, before a0 in only 5/18): the field is already at 1e4 magnitude within the training horizon, so "left the training range" is not a clean trigger; the growth itself is the pathology.
  - Scene vs weights: C weights on gravity expB scenes, geometry-free (job 3510): the same exponential field growth (median 285 @1, 1.6e3 @20, 1.4e4 @50, 2.5e5 @100, max 3e6), |a| up to 8e4, err@100 479 (E: ~1.5). So the instability is in the C weights, not the contact scenes.
  - Causal tests on C, 24 c_id scenes, 60 ticks (tot |a| max over ticks 21+; err@40 / @60 mean position error):
    | intervention | scenes with |a|>200 | median |a| max | err@40 | err@60 |
    |---|---|---|---|---|
    | none | 75% | 2593 | 77 | 451 |
    | zero field every 5 ticks | 0% | 36 | 12.4 | 26 |
    | zero field every 10 | 0% | 39 | 10.6 | 23 |
    | zero field every 20 | 0% | 39 | 10.5 | 25 |
    | clamp field to +-100 | 0% | 56 | 12.2 | 37 |
    | clamp field to +-1000 | 75% | 828 | 45 | 200 |
    The runaway is carried by the magnitude of the recurrent potential state. Resetting every 20 ticks (the training horizon) or clamping at ~100 (E's native range ~50) removes the blow-up with no other change. These are diagnostics, not proposals; err in contact scenes is saturated (floor 4.5-6.9) so only the runaway metric is informative.

## Caveats
- E envelope uses 8 scenes; cluster exceedance fractions are partly sampling. E zero-shot on contact scenes was not run (no geometry input). The star family is OOD for E (trained gravity, unit masses). Chaos floor: err beyond tick ~50 in cluster and ~100 in expB is near the chaos ceiling (scatter-chaos-floors.md), so late-tick differences between variants are weakly informative. bh300 / relativistic scenes not run. Contact numbers are one seed set (9100), 24 scenes.
- Exact-force override uses the exact force at the predicted position, with the model's momfix not applied; a_prev oracle uses exact force at the IC.
