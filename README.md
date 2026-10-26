# Augur

**Learning particle simulators that scale, tile, and stay honest about what they get wrong.**

Augur is a research project by Adam Webb (Cal Poly Pomona, GitHub: Faraday-dot-py) on neural surrogate simulators: models that learn the rules of a particle system from examples, then run it fast enough for real-time and very large scenes. It was started as "Bounce" (a bouncing-ball model) and renamed Augur in October 2026. It is about five weeks old at the time of writing and has 537 commits.

[contact: add preferred email]

Live demos run in the browser: https://faraday-dot-py.github.io/Bounce/

## The through-line: four requirements

Every design choice is judged against four requirements, and the project reports where each one stands rather than where it hopes to be.

1. **Generalization.** One approach should work for any particle system, including ones with different rules (gravity, contact, molecular forces). Status: shown on a handful of systems, not yet on arbitrary ones.
2. **Tileable.** A model trained on a small region should simulate a much larger one by cutting it into tiles. Status: working for gravity, with a known seam and accuracy cost.
3. **O(N) cost.** Time grows linearly with the number of bodies N, never quadratically. Status: measured, see below.
4. **Interpretability.** The learned rules should be recoverable from the model. Status: a goal. It has not been tested, and this README does not claim it.

## What exists today

**Scale.** The newest model family, the scatter field, deposits each body onto a grid, applies a convolutional network to the grid, and reads forces back at each body. That keeps cost linear. On one H200 GPU it advanced a 1-billion-body gravity scene (a black-hole-style cluster) for 100 ticks (a tick is one simulation time step) at a median 1.33 s per tick and 16 GB peak memory, using 16-bit body state. An earlier token-per-ball model ran 1 billion bouncing balls for 300 ticks in 3,825 s (12.75 s per tick, 121 GB peak), with measured linear scaling from 1 million to 100 million balls (fitted exponent 0.96 to 0.99). These are speed and memory results. No accuracy claim is made at 1 billion bodies, because no exact reference exists at that size; accuracy is measured separately at small N.

![Measured cost scaling to 1 billion bodies](results/bench_scaling.png)

![All 1e9 bodies at ticks 0, 11, 40 and 100: density, learned force, far-field potential](videos/scatter_bh_1b_panels_frames.png)

**Browser.** A live site runs the models client-side with side-by-side ground truth and a view inside the network. A WebGPU port of the network passes its numerical check against the CPU path (maximum relative difference 2.4e-6 on one pass, 5.7e-6 over a 20-tick rollout). Its speed has not been measured on a real GPU, so there is no speedup claim yet.

**Interpretability tooling.** A 3D walkthrough steps through one scatter-field forward pass in the browser. This helps people see what the model does; it is not the same as recovering rules.

## Newest work: the tiered architecture

The scatter field is local: it sees a bounded neighborhood. The current design stacks tiers. A tier-0 model handles fine-scale physics in a tile; a tier-1 tile governs a 2x2 block of tier-0 tiles, and so on, with identical architecture at every level and separate weights per level. The hope is long-range physics at near-constant cost per level, and a path to scenes far larger than any training region. This design was worked out as a written theory document with ten identified weak points, each walked through and given a proposed test before any code was run.

| Milestone | State |
| --- | --- |
| Scatter field, gravity, linear cost, 1B-body speed run | done |
| Tiled inference with seams checked | done, with accuracy cost |
| Browser demos and 3D walkthrough | done |
| WebGPU network port, numerics | done |
| WebGPU network port, speed | next (needs a real-GPU browser run) |
| Tier 0 at matched training budget | done, passes every pre-registered criterion (one seed) |
| End-to-end hierarchy accuracy | not passing: 1.5 to 2x the tiled baseline's position error at tick 20 |
| Retrain removing velocity and recurrent inputs (the targeted fix) | in test, two seeds running |
| Local energy-bias loss term | code and tests done, queued |
| Per-body-mass support (planet-and-disk systems) | design, curriculum and trainer written and unit-tested, not trained |
| Interpretability (recovering the rules) | not started |

Two diagnoses made the hierarchy gap specific. First, a mid-density bias that looked architectural turned out to be mostly undertraining: the matched-budget tier 0 closed it (bias from -1.5 and -1.8 to about -0.1). Second, teacher-forced analysis located the remaining hierarchy error in the dense regions of the tier-0 model (its energy-work error there is 2.1x the baseline's), not in the upper tiers, which add only 4 to 10 percent error over an exact external field. That narrowed the fix to one retrain.

Per-body-mass support targets a star-plus-planetesimal-disk system, the kind of mixed-mass scene robotics and astrophysics both care about. The current models only handle unit-mass bodies. A direct N-body reference for the real disk design is validated (relative energy drift 2.4e-5 over 100 simulated years), a 500-frame comparison video was rendered and reviewed, and the mass-conditioned model has pre-registered pass and kill lines. It has not been trained.

## How the research is run

Work is done by a supervised team of AI agents. A supervisor agent delegates, records every decision in a running log (86 dated entries so far), and does not do experiment work itself. Each agent works in its own branch, writes pass and kill criteria before running, and logs results to a shared experiment log (93 sectioned entries across the project). When an agent hits a limit or loses its compute connection, it is resumed from saved state. After every run that produces rollouts, a fresh reviewer with no hypothesis primed describes the rendered frames, because summary statistics alone have repeatedly hidden visible artifacts. Experiment names must be descriptive, enforced by a check script.

Many results are negative, and that is part of the method. Pre-registered kill lines mean a failed arm is recorded as a result, not tuned away. A selection of what did not work and what it taught:

- **Upper tiers on a frozen tier 0** did not reduce far-field error. Several levers (higher-order moments, more data, longer training) were then tried one at a time, and the limit was the frozen base, not what the upper tiers pass up.
- **Joint fine-tuning of all tiers** cut some errors but raised energy drift roughly 6x on the earlier tier 0, and added nothing over sequential training on two later runs.
- **A molecular-dynamics scatter field** failed because its grid branch injected a small force that depends on velocity and history and does net work. Removing those inputs, with no other change, removed most of the drift. That finding is what the current retrain applies to gravity.
- **A six-law mixed-interaction retrain** regressed gravity accuracy and was killed.
- **A planetesimal-disk proxy** with the existing unit-mass model was killed, which is why per-body-mass support is being built rather than assumed.
- **A loss-function analysis** showed the error is not driven by a few outlier particles, so the tail-focused loss idea was dropped and a local energy-bias term was proposed instead.

Results are logged with the numbers that decide them, so a reader can see which claims are strong (linear cost, tier-0 gravity, the energy diagnosis) and which are one seed or untested.

## Comparison videos

Six rendered comparison videos cover gravity clusters, multi-clump scenes, successive model generations, wall contact, and liquid-argon-style molecular dynamics. A separate 500-frame disk video also exists. The video files are large and are not stored in the repository; they are available on request.

## Who this is for

- **ML research and applied science:** pre-registered experiments, held-out-seed selection, and a documented record of negative results.
- **ML engineering and infrastructure:** O(N) design, tiling, 16-bit state for fitting 1 billion bodies on one GPU, GPU-side data generation, and a WebGPU port.
- **Robotics and simulation:** learned surrogates for real-time simulation, contact dynamics, and mixed-mass systems.

Author: Adam Webb, Cal Poly Pomona (CPP).
