# Does the simulator need dark matter? (research, 2026-10-20)

Branch `worktree-agent-afd468dd2f6e11b2f`. Script `scripts/scatter_dm_probe.py`, summary `scripts/summarize_scatter_dm_probe.py`, raw `results/scatter_dm_probe.json` (Polaris job 3772; 3770/3771 crashed on script bugs, no results). Checkpoint Exp E (`E_ms_kp_pot_v_g128.pt`), weights untouched, no training. Literature claims are tagged [read] (abstract page fetched), [snippet] (search snippet only) or [unverified] (recalled, not checked).

## 1. Physics framing

Dark matter (DM) in N-body codes is collisionless mass interacting only gravitationally, same softened Newtonian force as other matter. Adding it is an initial-condition and mass-species question, not a new force. Same law in the sim: `r2 ** -1.5` softened pair force, eps 0.5 (verified in `scripts/gravity_sim.py`, `rollout_mass_torch`). Note the sim has 2D positions with the 3D 1/r^2 force law, so 3D profile formulas (NFW, Hernquist) do not apply directly; a softened point mass IS a Plummer sphere, so a Plummer halo maps exactly (eps = scale radius).

- Cosmological halos: NFW profile, arXiv:astro-ph/9611107 [read abstract: universal shape from N-body]. Code: GADGET-2 astro-ph/0505010 [read abstract: TreePM, collisionless DM plus SPH gas; softening details not read], GIZMO 1409.7395 [snippet; gravity details not read], FIRE-2 1702.06148 [read abstract; DM treatment not in abstract]. Equilibrium live halo/disk ICs: GalIC, arXiv:1402.1623 [snippet].
- Live vs static halos: live halos exchange angular momentum with discs and change bar growth (Athanassoula astro-ph/0112077 [snippet]; Frosst et al. 2408.03375: disc barred in live halo stays unbarred in static halo [snippet]). A static halo is an accepted approximation only where the response is negligible [snippet].
- Unequal particle masses: two-body scattering transfers energy from heavy (DM) to light (stellar) particles, spurious disc heating and size growth when DM particles are heavier than stars (Ludlow et al. 1903.10110 [read abstract]; 2306.05753 [read abstract]; earlier Steinmetz and White astro-ph/9609021 [snippet]). Typical galaxy-formation runs have particle mass ratio mu ~ 5 [snippet]. Softening: Plummer softening biases force and larger softening trades noise for bias (Dehnen astro-ph/0011568 [read abstract]).
- Evidence for DM in galaxy scenes: rotation curves are flat or rising at large radius for most discs, with DM dominant at low luminosity (Persic, Salucci, Stel astro-ph/9506004 [read abstract]; history: Bertone and Hooper 1605.04909 [snippet]); dwarf spheroidal velocity dispersions stay flat to large radius, implying halos of 1e8-1e9 Msun (Walker et al. 0708.0010 [read abstract/snippet]).
- Scenes where it is not needed: Solar System DM density bounds are tiny (Khriplovich and Pitjeva astro-ph/0601422 [snippet], the number was not read); globular clusters show little or no DM (Moore astro-ph/9511147 M/L upper limit ~2.5 [snippet]; Conroy et al. 1010.5783 [snippet]; NGC 2419 caveat 1210.7787 [snippet]).

## 2. Which use cases are affected

Nothing in the repo claims galaxy-scale realism (grep for galaxy/galactic in md, py, html, js: no hits). Solar system, globular cluster, BH collapse, flyby, binary/orbit scenes: unaffected (DM is negligible or absent there per above). The 1e9-1e10 clumpy/flyby/Plummer clouds are already collisionless self-gravitating particles, which is what DM is in cosmological codes; they only lack a separate visible species. Only `nbody_ic.disk` (exponential disc, circular speed from enclosed self mass x0.7) resembles a galaxy and has no halo; its rotation curve is the baryon-only (falling) one. A dwarf/disc-galaxy scene would be wrong (no flat curve, disc unstable without halo support) without a DM component.

## 3. Representations, with the measured data (job 3772)

Setup: 100 unit-mass Plummer (a 3) visible bodies; DM halo of total mass 200 (Plummer a 6), live particles of mass m_h in {1,3,10,30} (N 300/167/120/107), or none, or static. Circular-ish ICs (0.7 v_c). 50 ticks, dt 0.1, truth exact fp64 softened all-pairs. Seeds 4738, 9000, 12000 (mean). err = mean position error per visible body (sim units); floor (1e-5-perturbed truth) is <=0.0011 at tick 20, so these are model errors. vc relerr = median |v_c,model - v_c,truth| / v_c,truth at t=0 on visible bodies, v_c^2 = -a.r.

| arm | err@5 | err@10 | err@20 | err@50 | v_c relerr |
|---|---|---|---|---|---|
| visible only (control) | .010 | .022 | .064 | .97 | .062 |
| live halo, m_h 1 (N 300) | .016 | .042 | .16 | 1.8 | .044 |
| live halo, m_h 3 | .019 | .050 | .22 | 2.9 | .055 |
| live halo, m_h 10 | .041 | .151 | .92 | 5.1 | .069 |
| live halo, m_h 30 | .298 | 1.06 | 3.75 | 16.6 | .306 |
| static analytic halo added to model force | .011 | .024 | .077 | .74 | .035 |

Predictions written before the run: control .03-.06 (got .064, slightly high), m1 .05-.15 (.16, slightly high), m3 .1-.3 (.22 ok), m10 ~1 (.92 ok), m30 >3 (3.75 ok), static within 2x of control (1.2x ok). WRONG: v_c error at m_h 10 predicted >50%, got 7%. The t=0 mean field is linear in source mass and fine to 10:1; the failure is the dynamics (heavy-body near field and encounters), so a rotation-curve check at t=0 alone would wrongly pass 10:1 mass. Consistent with scatter-ood.md (ratio >=10:1 breaks; dominant >=30 breaks 47x).

Representations:
- (a) multi-species with mass channel: works to ~3:1 (err@20 3.5x control at 3:1 with 1.7x the particles; equal mass 2.5x control from N alone). Fails at 10:1 (14x) and 30:1 (58x). Real DM/baryon particle mass ratios are 1e2-1e6 in galaxy-scale runs per the task statement (not verified in literature here); out of reach without retraining on log-uniform masses plus input normalisation (scatter-ood.md guess ~1e2:1, untested) and the larger DM softening is a separate unsupported axis (eps off by 2x costs 10-45x).
- (a') equal-mass DM particles, f x more numerous than visible (mass ratio 1): within validity now; cost is N x (1+f), pair term O(N M), mesh O(N + G^2 log G) (scatter-scaling.md: N <= ~1e3 cold, ~5e3 warm per 64-unit box).
- (b) static external halo as a known term: exact for the halo part and zero cost, error equals the control. But it is a hard-coded interaction outside the model (a closed-form acceleration added by the integrator, not learned), loses halo response (live-vs-static literature above; the live-vs-static gap was NOT measured here), is not momentum-conserving. Project-rule conflict: needs a user decision. A rule-compatible variant is an external-field INPUT channel that the model learns to respond to (needs training; untested).
- (c) mesh-only DM density deposited on the grid: not run. Mesh path sees mass as density (U-Net input, un-normalised, h^2 dependent), so a 100x denser DM channel is OOD at the mesh input; the learned kernel K is linear in source mass but only trained for r<~64 and scenes wider than 16-32 units give a wrong far field (scatter-scaling.md, sigma 32: tick-1 force error .57). Would be O(N_visible + G^2 log G), DM costing nothing per particle, and fits tiling (far field is mesh), but requires a separate DM input channel, trained.
- (d) learned halo parameterisation: untested, needs training.

## 4. Cost and constraints

Constraint check for (a'): O(N) yes (mesh) but pair term O(N M); generalization yes (no new force); tileable: unchanged from base, an extended halo is a far-field object so scenes beyond ~16-32 units still fail (known failure); interpretability unchanged; no pp=8; non-conserving ok. (b) violates "no hard-coded forces" in spirit. Compute for the probe: ~5 min on 1 GPU.

## 5. Recommendation

Not needed now (no scene claims galaxy realism; solar/globular/BH/flyby unaffected). Needed later only for a specific goal: a galaxy/dwarf scene with flat rotation curve. Cheapest route then is (a') equal-mass DM particles, no model change. PASS/KILL criterion for the smallest next experiment (set before launch): a two-component equal-mass live halo (N_h = 2-5 x N_v, N total <= 1e3), held-out seeds 9000/12000, PASS if visible err@5/10/20 <= 3x the visible-only control (.010/.022/.064) and rotation-curve median relative error <= 10% at ticks 0 and 20; KILL if err@20 > 5x control or rotation-curve error > 25% (then DM would need training with a mass channel and a wider halo, not a zero-shot add). Probe above already meets it at 1:1 (2.5x at err@20, vc 4%) at N=300; the next step is N_h = 5 N_v at a wider halo (>16 units), which probes the known far-field failure.
