# Per-body mass in the scatter field / tier 0 (design, pre-registration, 2026-10-26)

Goal: star plus disk at about 1000:1 mass ratio (planetesimal-disk test, experiment-log "star plus planetesimal disk"). Current models are unit-mass trained (G=1, softening 0.5, dt 0.1, N 100-1000, extent 64, grid 128, cell h=0.5).

## What already exists (not duplicated)
- `ScatterField` already takes per-body mass. Mass is the scatter weight (CIC deposit of [m, m vx, m vy]), the learned kernel K(r) acts on the mass density (acceleration exactly linear in source mass), the pair term is linear in m_j, `momfix` removes the mass-weighted mean acceleration. Output is acceleration per unit receiver mass (equivalence principle: gravity needs no receiver mass), `lm` mass-exchange loss and `dm` head exist only in the multilevel model and are not needed (mass is conserved in gravity).
- Tried, in `docs/debugging/scatter-review-4-6.md` (s46): fine-tunes on log-uniform masses 0.1-10 (`mass`), same with `norm="mass"` (`massnorm`, divides input by mean mass), 0.01-100 (`masswide`). All three FAIL the in-distribution gate (harness 1.2-3.9x vs 15% allowed); zero-shot mass-ratio rows improved but stayed poor (massnorm q=1000 binary err@2 216 -> 3.4, heavy-1000 in a 300-body cluster 694 -> 46). Exp C (star 8-30, lights 0.01-0.5) existed for the old v1/v2 architectures only.
- `scatter-ood.md`: failure before any fix is silent wrong dynamics (no NaNs); q>=10 binary breaks, cause assigned to U-Net input scale and the pair term only ever seeing m_j=1.
- Not tried: a hierarchical star+light scene set at controlled q, mass-range channels, a normaliser that is tile-invariant, a curriculum with unit-mass replay, orbit-element pass lines.

## How mass enters (new, all opt-in, zero-init warm start from the existing checkpoint)
1. Deposition (unchanged): mass-weighted CIC of [m, m v]. Plus one more additive cell sum, S2 = sum m^2. Cell channels added: `log1p(rho/m0)` and `log(S2/S1 / m0)` (density-weighted mean body mass, "who is in this cell": star cell vs a crowd of light bodies), zero in empty cells. Both are sums of additive moments, so they merge exactly in the tier-k 2x2 child merge (S2 is just one more moment channel) and stay O(N).
2. Input conditioning: all linear channels divided by a mass unit m0 (below), then soft-clipped (identity for |x|<=64, logarithmic beyond; a 1000-mass star cell gives about 330 instead of 4000). New input columns are appended at the tail, so `load_compat` zero-pads them and the unit-mass behaviour is reproduced exactly (unit test).
3. Force readout: still acceleration (field gathered at the receiver, scaled back by m0), not force. Receiver mass is not an input for gravity (a 1e-3 planet and a 1 planet feel the same acceleration). If a non-gravitational force is added later, the readout should output force per unit charge-like property and divide by the receiver mass, noted not implemented. `momfix` stays mass-weighted: sum m a = 0 holds exactly; the star absorbs a tiny share and the correction applied to light bodies is of order the star's field error times M/M_total.
4. Scale equivariance: acceleration scales linearly with m0 by construction (kernel and pair term linear in mass; net sees dimensionless inputs and its output is multiplied by m0). Unit test: scaling all masses by 3 scales acceleration by 3.
5. Mass unit m0 (`norm="gmean"`): geometric-mean mass of the scene, constant per scene. For tileability m0 is a scene constant passed to every tile (`model.mass_unit`), never computed from a tile's own bodies (that would give neighbouring tiles different input scaling and a seam). Equal to the existing mean-mass normaliser for unit-mass scenes. Geometric rather than arithmetic mean so one 1000x star does not set the scale of a 1000-body disk.
6. Wide range (1e-2..1e3): the kernel path carries the exact linear part for any mass; the net only corrects, from log-compressed channels. Softening is not mass dependent (eps=0.5 fixed, as in truth); no per-mass eps. Time resolution is handled in the data: every orbit has period >= 40 ticks (so a body crosses <= 2 cells per tick), which fixes a_min(M) (7.4 for M=1000).
7. Pair term: stays linear in m_j (hard structural prior, same as now). If the near-field at q=1000 fails the pre-registered line, the next arm adds log(m_j/m0) as a zero-initialised pair feature (not implemented, not run).

## Constraints check
- O(N): one extra scatter plus elementwise ops; no all-pairs; the training truth is O(N^2) but only for data.
- Tileable: new channels are additive cell sums; normaliser is a scene constant passed to tiles; no per-tile statistics.
- Model learns dynamics: nothing hard-codes a force; kernel K(r) and the pair net are learned; linearity in source mass is the existing prior (superposition), flagged. No hard-coded conservative force added.
- Compatible with the velocity-free, non-recurrent tier-0 retrain (job 4050): the new channels do not use velocity or the recurrent field (the m v channels stay as they are).
- Not a parameter sweep: one architectural change (channels) and one data change (curriculum), arms separate them.

## Implementation (committed with this note)
- `scripts/scatter_field.py`: `ScatterField(massfeat=True, norm="gmean")`, `softclip`, `mass_unit` override. Variant `ms_kp_pot_v_g128_massfeat` in `train_scatter_field.py`.
- `scripts/scatter_mass_curriculum_data.py`: star+one-planet and star+disk(+planet) generators at ratio q (stages 1, 3, 10, 30, 100, 300, 1000; disk body mass M/(q R), R log-uniform 1-10, total disk <= 0.1 M), orbit-element and energy metrics.
- `scripts/train_scatter_mass_curriculum.py`: arms `massfeat_curriculum`, `curriculum_no_massfeat`, `massfeat_flat`; iteration-deterministic, `--resume` (existing checkpoint is resumed), full-state checkpoint (model, Adam, iteration, eval results) every 50 iterations, SIGTERM saves and exits 143. Stage mix: current q 0.45, earlier q 0.25, unit-mass base replay 0.3 (the s46 failure was losing the base regime). 7 stages x 600 iterations, batch 16, k 4->20, lr 2e-4, about 1 GPU-hour estimated (not measured).
- Tests: `tests/test_scatter_mass_curriculum.py` (CPU, 10 tests: zero-init identity, mass-unit scaling, tile-constant unit, soft clip, momentum, orbit elements, generators, metrics, verdict, SIGTERM + resume).
- Limits stated up front: curriculum tops out at 1000:1 star:planet and <= 1e4:1 star:disk-particle; the real design has about 6.7e4:1 for 1000 disk particles totalling 1.5% (needs a later stage); 2D only, grid extent 64 (real design 0.3-30 AU is a 100:1 radial range, mapped by the units choice, not tested); G=1 units.

## Pre-registered pass/kill lines (written before any run; seed 4738 start, held-out eval seeds 9000+q, 16 scenes per cell, 100 ticks)
Metrics: da/a = median over scenes of the relative semi-major-axis error of the model rollout against the truth rollout at tick 100 (planet scenes: the planet; disk scenes: median over disk bodies), de = median absolute eccentricity error, dE = median relative total-energy change |E(t)-E(0)|/|E(0)| of the model's own rollout (softened energy, same definition as truth).
- Star + one planet, q in {10, 100, 1000}: PASS if da/a <= 0.02, de <= 0.03, dE <= 0.02, finite. KILL if da/a > 0.15 (planet q=1000: > 0.30) or dE > 0.2 or non-finite.
- Star + disk (30-200 bodies + planet), same q: PASS if da/a <= 0.05, de <= 0.05, dE <= 0.05. KILL if da/a > 0.25 or dE > 0.3.
- In-distribution gate (unit-mass base scenes, err@5/10/20 vs the warm-start checkpoint): PASS <= 1.15x, KILL > 1.5x.
- Arm verdict: PASS all lines; KILL any kill line; else INCONCLUSIVE. Reported for the three arms separately (features vs curriculum vs both). Zero-shot warm-start numbers are recorded as `init` in the results json.
- Next step only after a PASS on both scene types at q=1000: increase disk particle ratio to 1e4-7e4 and N to 1000, then tile.

Status: code and tests ready; not run. Polaris GPUs held by the tier-k no-velocity retrain (job 4050 and second seed); no submission made.
Run: `PYTHONPATH=. python scripts/train_scatter_mass_curriculum.py --arm massfeat_curriculum --init-ckpt <tier0 or Exp E ckpt> --device cuda`.
