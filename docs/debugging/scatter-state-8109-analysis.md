# Scatter-field live page, state export at tick 8109

File: `results/scatter_state_tick8109.json` (n=100, dt=0.1, 811 time units of simulated time). Figures: `videos/scatter_state_8109_{positions,speed_distance,history,field}.png`. Analysis scripts were run from a scratchpad (not committed). No NaN/inf anywhere. No `meta`/`exportedAt` keys are present in the file despite the spec.

## Headline

1. **The truth ghost is repulsive, not attractive. This is a bug in the web port (very likely, strong numerical evidence).** Every number below says the ghost is flying apart under repulsion, so `err`, `truthE` and the ghost overlay on that page are not a valid reference.
2. **The model also blew apart**: 99 of 100 bodies are far outside the arena, with total (true-function) energy 5.7x |E0| above start, while the true system is bound. It is not a stable long-run simulator (expected for a model trained on <=20 ticks, but the size of the failure is larger than I would have guessed).
3. **The field heatmap is healthy and sensible for what it sees**: one body in the arena, one clean potential well on it, no NaNs, no checkerboard. It shows edge brightening and faint straight bands that need a look.

## 1. The truth ghost is repulsive

Evidence (numbers):
- Code: `.claude/worktrees/web-scatter/web/scatter/js/truth.js` lines 10-13 use `dx = pos[i]-pos[j]` and `out[i] += dx*inv`. The Python truth (`scripts/gravity_sim.py:7`, `d = pos[None,:,:] - pos[:,None,:]`, then `(d*inv).sum(1)`) uses `pos[j]-pos[i]`. The JS has the opposite sign, i.e. repulsive gravity. The JS `energy()` still uses the attractive `-1/r`, so the ghost does not conserve "its" energy.
- Inferred E0 = -338.6 (solved from drift series: truth E now 497.4 gives (497.4-E0)/|E0| = 2.468; model E 1593.5 gives 5.706; both fit one E0 to 3 digits). So the start was strongly bound (initial KE is about 89 from the 0.94 speed sigma, PE about -428; untested guess for these two).
- Ghost now: KE 499, PE -2.1, E +497, all 100 bodies at r = 431-4184 (median 2377). In a bound attractive system that is impossible; under repulsion the conserved quantity is KE + sum(1/r), about 89 + 428 = 517 (predicted) vs 499 measured (consistent, the initial KE is only estimated).
- Pure free flight: 100% of ghost bodies have outward radial velocity; r/(speed x 811) has median 0.99 and r vs speed correlation 0.99996 (`speed_distance.png`). Ghost angles are isotropic (8-sector counts 11-14 each), as expected for expanding gas.
- Ghost momentum 1e-13, angular momentum -8.3 (conserved by a central force either way).

Consequences: `err` (3862 now, 3745 at the start of the window, +0.59 per tick) is just "two different explosions drift apart". It says nothing about model accuracy. Truth energy in the window is flat (relative change 3e-4) because everything is free flight. Whether the first ticks of any session show a sensible err depends on how quickly repulsion vs attraction diverges; the bug has probably been there since the page was built, so earlier err/energy readouts from this page should be treated as suspect (not checked).
Untested: that fixing the sign alone repairs the ghost. A quick check is to run the numpy truth from a random start with `+` vs `-` sign and compare energy; I started that, but it was too slow inline and I stopped it.

## 2. Model state

| quantity | model | truth ghost |
|---|---|---|
| KE / PE / E (true softened function) | 1640 / -46.7 / 1594 | 499 / -2.1 / 497 |
| (E-E0)/|E0| (from export) | 5.706 | 2.468 |
| speed min / median / p90 / max | 0.13 / 5.04 / 8.3 / 10.3 | 0.54 / 2.98 / 4.5 / 5.2 |
| r from centre: min / median / max | 27.7 / 2299 / 6762 | 431 / 2377 / 4184 |
| bodies outside the +-32 arena | 99 | 100 |
| centre of mass | (0.962, -1.402) | identical to 3e-12 |
| total momentum | 5e-13 | 8e-14 |
| angular momentum | +83.4 | -8.3 |
| nearest-neighbour distance, median | 231 | 484 |

- **Expected**: momentum is ~0 (the model re-centres momentum each tick; truth conserves it) and COM equals the ghost's, since both start at zero net momentum with the same IC. Matching to 1e-12 is a nice sanity check that the two share an initial condition.
- **Surprising**: model KE is 3.3x the (already wrong) ghost's, and the energy the model gained (about +1930 on top of an initial -339) is about 5.7|E0|. In a true gravitational system with E0 < 0, nothing can escape to r~2000; the escape energy came from the model. The window data cannot say when (see "Not known" below).
- **Surprising**: model r/(v*811) median is 0.51 vs 0.99 for the ghost, with r-v correlation 0.93 (not 1.0). The model bodies were not simply launched once from the origin at their current speed. Either they were ejected faster and then slowed or got the force in stages. Untested.
- Model angles are strongly anisotropic (8 sectors: 19, 7, 18, 3, 5, 15, 6, 27); ejection direction preference. Plausible, untested cause: the grid is not rotation-invariant and a few violent ejections dominate. 13 bodies have speed > 8.
- Angular momentum +83 vs the ghost's -8: the model does not conserve L (known, see `scatter-field-architecture.md`: dL/L 0.07-0.3 at step 100); the initial L is unknown (seed not saved), so the size of the change cannot be stated.

### Clumps and bound pairs
- Truth ghost: no pair closer than 326 units. Zero bound pairs (expected under repulsion).
- Model: 29 pairs closer than 2 units, all belonging to **two surviving clumps**: 8 bodies (centre (-233, +128), rms radius 0.33, internal rms speed 1.03, internal KE 4.3) and a tight binary (centre (-252, -79), separation about 0.3-0.4, relative speed 0.27), moving together at about 1.45 units/time in different directions. 27 of the 29 close pairs are bound by the 2-body criterion. Closest pair 0.17 (eps is 0.5, so inside the softening length, fine for the true function).
- Expected-vs-surprising: a compact bound clump surviving 8000 ticks is believable (the learned pair term handles r<2, and these pairs are far outside the arena so no grid force interferes). It is surprising that the model's clump is in a region (hundreds of units out) the model never saw. Model PE (-46.7) is almost entirely this clump. Whether the truth sim would have formed the same clump from the same start: not checkable (ghost is repulsive; seed lost).
- The remaining 90 bodies are lone escapers, which is why the 200-tick energy series is flat.

### History window (last 200 ticks, `history.png`)
- Model E drift 5.7024 to 5.7056 (about +1.3 in absolute energy over 200 ticks, +0.07% of current E). Ghost 2.4684 to 2.4687. Both are free flight plus the clump jiggling. Expected, uninformative about earlier behavior.
- err slope +0.59 per tick equals the relative speed of two flying-apart populations: consistent with 1. above.

### Not known
When the model's energy was created (cannot be seen in a 200-tick window). Likely in the first 100-500 ticks while the cold cloud collapsed (plausible, untested; a rerun with E logged every tick would show it). Whether the 99 escapers left through the +-32 edge (no grid force there) or were kicked by close encounters inside is unknown.

## 3. The potential field (`field.png`)
- Range -0.485 to -0.100, mean -0.267, no NaN/inf. Quadrant means -0.2664, -0.2674, -0.2666, -0.2675: uniform background, as expected for a nearly empty arena.
- It is consistent with the body density: exactly one body inside the arena (index 45, at (-24.6,-12.6)), and the field minimum is at (-24.75, -12.75), within a cell. Well profile: -0.474 (r<0.5), -0.41, -0.31, -0.275 (r=2-4), then flat -0.269. True softened 1/r would be -1.67, -1.09, -0.62, -0.33 at the same radii (and -0.17 at r=4-8). So the learned well is about 1/6 as deep and has no 1/r tail beyond r~4. Expected by design (the pair term handles r<2, and for 1 body the mesh potential is mostly a gauge/offset); I cannot say whether the shallow well matters at 100 bodies (this is a single-source test; untested).
- **Boundary artifact (surprising, minor)**: the potential rises toward all four edges, mean -0.17 on the outer row/column vs -0.277 interior (+0.1, 40% of the well depth). Matches zero padding in the UNet; matters little here because only 1 body is inside, but any body near the edge sees this gradient (max |grad| 0.16).
- **Faint straight bands (surprising, untested cause)**: a darker horizontal band near y=+24 spanning x~-25..+25 and a weaker vertical band near x=-22; neither is aligned with a body. Possibly a stale recurrent-state echo or UNet pooling-boundary footprint; not tested.
- Spectrum: 49% of variance at |k|<4, 38% at 4-16, 13% at 16-40, <1% at k>40 and 4e-5 at the grid scale: no checkerboard or grid-scale noise (checker score 7e-7). Laplacian rms 0.009 vs field std 0.026. Smooth.

## 4. Sanity list
- NaN/inf: none. Arrays all length 200 (or 16384 for the field).
- Physically implausible: the ghost (repulsive) and the net energy gain of the model. Everything else is consistent.
- The model's angular momentum and energy are not conserved by design (documented). The size at 8000 ticks is huge but unconstrained by training.

## Suggested checks, not done
1. Fix the sign in `truth.js` accel; confirm truth energy is conserved to a few % and that bodies stay bound.
2. Log model energy and #outside-arena per tick from tick 0 for 1000 ticks to see when and how energy is created.
3. Add a "recentre/rescale" or boundary policy for escapers (outside bodies get no force; they never come back).
4. Include the seed in the export.

## Future direction

The user wants to eventually build a general-user user interface / design style / platform out of this project, and likes the visuals it can make.

From what I looked at: the potential-field heatmap (clean, smooth, instantly shows where a body lives and boundary/band artifacts) and the side-by-side model vs truth ghost overlay (immediate, intuitive accuracy readout) look most promising. They only work as product features once the ghost is fixed (see 1) and a boundary/escaper policy exists, since a repulsive ghost and all-escaped bodies leave the page showing nothing meaningful after the first few hundred ticks.
