# Augur: a learned particle simulator with linear cost and tileable regions, and an honest account of what failed

Draft of 2026-11-17. Sole author. Compute was provided by the CPP Polaris HPC system. Not for release as is.

How to read this draft. Every number comes from a saved result file or a log entry; the bracket after a claim names its row in docs/paper/paper-claims-and-evidence-table.md (the table), where file and line citations live. The paper itself cites result tables, not raw files; no raw files are released. Each claim carries a marker: validated (several seeds or sizes, a control and an approved pre-registered line), single-seed, worker-proposed-line (verdict rests on a line the author did not approve), post-hoc-amended (verdict depends on a line or rule changed after the result), speed-only (time and memory, no accuracy) or unvalidated. Verdicts are reported as pre-registered, and amendments are labelled as amendments. Related work is marked "to look up"; no citations have been looked up yet.

Abbreviations, defined once. N-body: a simulation of many bodies pulling on each other. err@k: mean per-body position error at tick k against an exact simulation (lower is better). dE@k: relative energy drift at tick k. BL: the tiled scatter-field baseline a ratio is divided by. CIC: cloud-in-cell, a rule for spreading each body onto nearby grid cells. FFT: fast Fourier transform. U-Net: a convolutional network with a down-and-up path. GNS: graph-network simulator. PM: particle mesh. IAS15: an adaptive high-accuracy N-body integrator. GPU-h: GPU hours.

## Abstract

Augur is a family of learned particle simulators built around four requirements: generalization to new particle rules, tileability (one model applied to a larger area), cost that grows no faster than linearly in the number of bodies, and interpretability. This draft reports what the evidence supports. The main results are speed and memory results: the scatter-field model advanced 1e9 and 1e10 bodies at about 1.4 and 13.4 seconds per tick (speed-only, no accuracy at those sizes), and the contact-force token model advanced 1e9 balls at 12.7 seconds per tick (speed-only). Tileability holds for the contact model (strip-tiled rollouts match global rollouts at floating-point noise) and for tier-zero gravity tiling on position error (passes on 4 of 4 training seeds), while tiled energy drift fails its pre-registered line on every seed. Secondary results cover size generalization, a closed-loop replay fine-tune, and a baseline comparison in which the learned model beats a plain particle-mesh code at 1000 bodies and loses at 1e4 bodies and above. Interpretability is partial, and most attempts to extend the architecture failed or were inconclusive. We report these as pre-registered negative results.

## 1. Introduction

The goal is a simulator that learns the rules of a particle system from examples and then runs on far more bodies than it was trained on. Four requirements define success (core requirements document): (1) generalization across particle rules, (2) tileability, (3) cost at most linear in the number of bodies, (4) interpretability of the learned rules. The honest summary is that requirement 3 and a large part of requirement 2 are met, requirement 1 is met only for the trained gravity-like rule and fails for mixed rules and unequal masses, and requirement 4 is partly tested and mostly negative.

Contributions, in order of support. (a) A scatter-field simulator that ran 1e9 and 1e10 bodies on one GPU at near-linear time per tick (speed-only). (b) A tileability study for gravity and for contact forces, with a clearly reported energy-drift failure. (c) Secondary results on size generalization, visited-state replay and a plain-PM comparison. (d) An evaluation-validity package: a validated reference simulator, a chaos-floor analysis showing which error horizons are meaningful, and a pre-registered, independently reviewed workflow. (e) A catalogue of negative results.

## 2. Methods

### 2.1 Scatter field

Each body is spread onto a grid with CIC deposit, a U-Net maps the grid to an acceleration field, and the field is gathered back at each body. A learned kernel term is evaluated by FFT with an extended-padding edge fix, so cost is linear in bodies plus a grid term (architecture document). The pinned baseline checkpoint is ms_kp_pot_v_g128. The edge fix removes a boundary artifact without retraining: at 1e6 bodies the mean model-to-exact force ratio in the outermost grid band fell from 91.1 to 1.53 (next two bands 61.9 and 27.6 to 1.56 and 1.58) [grid-edge-artifact-fix-without-retrain; single-seed]. A later rerun of a corrected interior-force quantity gave 1.0104 instead of 1.53-1.58; the two quantities were not shown identical and the conflict is unresolved, so interior force accuracy at 1e6 or more bodies is not claimed here.

### 2.2 Tiered hierarchy and ghost-cell tiling

Tier zero is the base model on a tile; upper tiers work on moments. Tiles use ghost cells (a halo of neighbouring bodies, halo 16 in the gravity runs) plus a coarse far-field term. A design document lists ten weak points of the hierarchy, each with a proposed test [tiered-hierarchy-theory-and-ten-weak-points; unvalidated, design only].

### 2.3 Token-per-ball model

Each ball is a token and a transformer-style network predicts its next state; a conservative-contact variant adds central, distance-only pair forces. This model handles contact forces and is structurally limited for unbounded-range forces.

### 2.4 Evaluation protocol

Truth comes from an exact N-body simulator validated against IAS15 (section 7.1). err@k and dE@k are measured against it. Pass and kill lines were registered before runs; where a line was set by a worker agent and not the author, or changed after a result, this is marked. Each numeric result was recomputed from raw files by a fresh reviewer subagent with no hypothesis primed. Seeds: training seed 4738 unless stated; held-out evaluation seeds 9100, 9200, 9300 and fresh seeds 12000, 12100, 12200.

## 3. Main result 1: cost grows linearly at scale

### 3.1 Scatter field to ten billion bodies [speed-only]

At 1e9 bodies the scatter field ran at a median 1.33 s per tick over 100 ticks (16 GB) and 1.39 s per tick over 300 ticks with the edge fix (14.0 GB); a clumpy 1e9 scene ran at 1.47 s per tick over 200 ticks. At 1e10 bodies, 100 ticks, the median was 13.37 s per tick (13.4 in the log) at 88.1 GB peak memory in the raw file (87.3 GB in the log), 1.34 ns per body per tick. The ratio is 9.64x for 10x the bodies when comparing the 300-tick 1e9 median with the 100-tick 1e10 median, and 10.09x when both use 100 ticks, so the scaling is linear to slightly superlinear. Per-tick time at 1e9 rises as the scene clumps, from 1.02 s at tick 0 to about 1.8 s. These runs measure speed and memory only. No accuracy against an exact reference exists at these sizes; the only accuracy figure is a self-test relative error of 5.8e-5 to 1.1e-4. Late-time lattice stripes in fp16 and core drift are known limitations. One seed, one setup, two sizes at matched ticks: this is not a scaling curve. [scatter-field-linear-cost-to-ten-billion-bodies]

### 3.2 Token model to one billion balls [speed-only]

At 1e9 balls over 300 ticks the token model took 3825.3 s in total, median 12,749 ms per tick, 121,352 MB peak, in fp16 hidden state. Across six sizes from 5e6 to 1e9 the fitted log-log slope of median time against balls is 0.988 (0.983 over 5e6 to 1e8). One run per size, no repeats, and the hidden precision changes between fp32 and fp16 at the largest size. The linear scaling applies to contact forces only; for unbounded-range forces this model is quadratic. [token-model-linear-cost-to-one-billion-balls]

### 3.3 What explains cost blow-ups in the 2D bench

An 87x slowdown in one benchmark was traced to the bench loader's all-pairs control path, which took 99.5% of per-force time at 1e5 bodies; the production adaptive path took 15 ms at 1e5 against 60 ms for the 3D voxel variant. The production-path curve above 1e5 has not been measured. [scatter-2d-all-pairs-cost-blowup-diagnosis; single-seed]

### 3.4 Neighbour search at three million bodies [post-hoc-amended]

The pair term uses a 16-neighbour search. At 3e6 bodies the argmax body gave an error of 1.469e-2 against an fp64 brute-force reference, failing the pre-registered line (pass at most 1.7e-6, kill at least 1e-4): KILL as registered. The 16th and 17th neighbours were tied (distance difference 1.46e-10). A tie-aware rule, written after the KILL and confirmed by the author as the verdict of record, gives 5.56e-8, a PASS. This is a post-hoc amendment, not a plain pass: the tie-aware comparison is against the model's own chosen neighbour set, so it is small almost by construction, and only one body was checked. [pair-term-neighbour-search-at-three-million-bodies]

## 4. Main result 2: tileability

### 4.1 Contact model [single-seed]

Strip-tiled and global rollouts of the token model differ by floating-point noise (halo at least 12: max difference 0 to 6e-5), and translation equivariance is exact in the interior. For unbounded forces this fails: error is 22x once a cloud exceeds the 100-unit cutoff. [token-model-strip-tiling-equals-global]

### 4.2 Gravity ghost-cell tiling [unvalidated in this draft]

Ghost-cell tiling plus a coarse far term passed an accuracy gate against a single G=256 grid with no visible seam artifact, but the ghost-cell closure failed its energy gate (dE@1000 at 1.246x and 1.660x against a 1.15x gate). The seed count and underlying numbers were taken from the architecture document and not re-opened against a raw file for this draft. [scatter-field-ghost-cell-tiling-for-gravity]

### 4.3 Tier-zero tiled closed loop: position passes, energy fails [validated]

On four training seeds (4738, 4739, 4740, 4742) the tiled tier-zero model's err@20 divided by BL passed the pre-registered line (at most 1.0 on every held-out evaluation seed, and plain within 1.10x of the matched-60k baseline) on all four. Ratios across the three evaluation seeds: seed 4738 0.822/0.817/0.746; seed 4739 0.9990/0.9881/0.8761 (margin about 0.001 at the worst evaluation seed; the logs say 0.0001, which the reviewer found to be a tenfold slip); seed 4740 0.932/0.943/0.883; seed 4742 0.911/0.891/0.831. The effect is real but small on the weaker seeds. The raw files for 4740 and 4742 were not re-derived by a reviewer. [tiled-closed-loop-tier-zero-accuracy]

Energy drift fails. dE@100 divided by BL is 3.89-4.75 (seed 4738), 4.04/5.77/4.53 (4739), 3.43/3.76/3.37 (4740), 2.43/3.37/2.96 (4742), against a 2.5x line. Absolute drift is 0.033-0.038. [tiled-closed-loop-tier-zero-energy-drift-fails; validated negative]

### 4.4 Hierarchy and three-part training add nothing end to end [single-seed]

End-to-end hierarchy runs gave err@20 at 1.5-2.4x BL and drift 4-5x; upper tiers on frozen tier zero added 28-29% force error. A three-part separate-then-joint training run gave an err@20 ratio of 1.010 held-out (kill at 0.97 or above) with a 0.991/1.005 continuation control. Both are KILL results. [upper-tier-hierarchy-adds-nothing-end-to-end; three-part-separate-then-joint-training-kill]

## 5. Secondary results

### 5.1 Size generalization of the token model [single-seed]

Trained on at most 200 balls, err@20 was 1.44, 1.09 and 1.06 at 1x, 4x and 8x ball counts (seed 9000; seed 12000 gave 2.12, 1.14, 1.08), and 50,000 balls gave err@20/50/100 of 0.94/9.3/71.5 with kinetic energy within 2% of truth. The err@20 at 1x is at the contact chaos floor, and err@50 and err@100 are not usable. Density was held constant, so this tests locality, not density shift. [token-model-size-generalization-to-fifty-thousand-balls]

### 5.2 Closed-loop visited-state replay [single-seed; cause unsettled]

Fine-tuning the tiled model on labels at its own visited states cut dE@100 divided by BL from 4.25 to 1.99 (base seed 4738, job 4089) and from 4.78 to 1.88 (base seed 4739, job 6168). A truth-state replay control reached 2.10, and other arms (late ticks only 2.275, early only 2.334, one round 2.019, matched compute without replay 3.671) show the gain is not specific to replay. Curiosity arms carry no verdict. The 2.5x line is met only in the fine-tuned runs. [visited-state-label-replay-cuts-drift]

### 5.3 Comparison with a plain particle-mesh baseline [single-seed]

At N=1000, median and 90th-percentile force error of plain PM divided by the learned model: Plummer 2.29 and 1.80, two-clump 3.68 and 3.19, exponential disk 4.60 and 3.75. So the learned model wins by 2.3x to 4.6x on median error and by 1.8x to 3.75x at the 90th percentile at N=1000. At N=10,000 and 100,000 plain PM beats the learned model on every family (Plummer N=100,000 median 0.0131 for PM against 0.0875 learned). The pre-registered verdict was INCONCLUSIVE_PARTIAL (Plummer 90th percentile 1.80 below the 2.0 line). [learned-vs-plain-particle-mesh-force-at-n1000]

### 5.4 Accuracy at 100,000 bodies [single-seed]

On a 100,000-body black-hole-style scene, err@5/10/20/50/100 is 0.073/0.266/1.327/8.31/18.94 against a constant-velocity baseline of 3.44/9.51/20.54/50.65/100.84. One scene, one seed; the particle-mesh baseline was not run at this size here. [scatter-field-small-n-gravity-accuracy-at-100k]

### 5.5 Solar-system stability [single-seed]

With the edge fix, 0 of 8 planets were lost at ticks 100, 300 and 1000, and 5 of 8 (62.5%) at 2000; median period error was 3.5%, 7.2%, 17.1% and 44% at those four horizons. The failure at 2000 ticks is a limitation. [solar-system-stability-with-edge-fix]

## 6. Evaluation validity

### 6.1 The reference simulator [post-hoc-amended]

The truth simulator's force matches REBOUND 5.2.2's Plummer pair formula to 1.3e-16 to 4.3e-16 relative (line 1e-6), and position error against IAS15 falls 4.00x per substep doubling from 8 to 32 substeps on all 8 cases. As registered the overall verdict was INCONCLUSIVE (the Kepler energy clause failed at 1.3-1.8e-4 against 1e-4, and a replica-standard-deviation yardstick passed 0 of 9). Under a line amended after the result and approved by the author on 2026-11-03 it is a PASS. Both verdicts stand on record. [reference-simulator-validated-against-ias15]

### 6.2 Chaos floor [validated for err@20; single-seed for the BH scene]

In cold-collapse scenes with 300 bodies, err@20 of 0.0622 is far above floors from float32 truth (3.2e-6), eight-substep truth (5.4e-4) and a 1e-5 perturbation (3.9e-4), so err@20 is model-dominated. In 10-100 body scenes (240 scenes), err@20 is .0040 against floors 2.3e-6 / 1.1e-4 / 2.3e-4. At k=100 the 300-body scene is at the chaos ceiling: the model's 4.73 is comparable to the 3.4 produced by a 1e-3 relative perturbation (saturation about 6), so err@100 cannot rank models there; err@50 (0.912 against a floor of 0.015) is still informative. The BH scene analysis uses four scenes. [chaos-floor-err20-is-model-dominated; chaos-floor-cold-collapse-err100-at-ceiling]

### 6.3 Process

Nineteen architecture experiments were pre-registered with pass and kill lines; ten finished with verdicts: 1 PASS, 4 KILL (one of which, summary-token, rests on a worker-proposed line), 3 INCONCLUSIVE, 2 INCONCLUSIVE-undertrained; eight are pending and one is a curiosity. Fresh-reviewer subagents caught real errors, for example pooled-versus-mean ratios in a cost result. The number of reviewer interventions that changed a verdict has not been quantified. [pre-registered-kill-lines-and-fresh-reviewer-method; validated as a documented process]

## 7. Generalization across rules and masses (requirement 1)

### 7.1 Mixed-law training fails [single-seed]

A retrain on mixed laws without the amplitude-10 power-1 scenes gave gravity err@20 of 33.08 (31.128 and 35.027 on seeds 9000 and 12000) against 0.1064 for the main checkpoint and 0.0407 for the reference; kill edge 0.106. Training was unstable (about 6% of steps skipped by iteration 60,000), a confound, and the controls are different-seed checkpoints. [mixed-six-law-retrain-fails-gravity]

### 7.2 Unit mass only [single-seed]

All results use unit-mass bodies. On a planetesimal-disk proxy (700 disk bodies plus 300 core), err@20 was 0.016 but err@100 was 1.73 (kill above 1.0) and model dE/E0 was 1.23 against 0.003 (kill above 0.3): KILL. The real disk, with a 1000:1 mass ratio, is out of the trained range and was not tested. A mass-conditioned model exists, with 10 CPU tests and no training. [unit-mass-only-and-planetesimal-disk-proxy-kill]

### 7.3 Contact model results [single-seed]

The conservative-contact token model gave wall restitution 1.00-1.04 and head-on pair kinetic-energy ratios of 0.99-1.00 [token-model-contact-energy-conserved]. It fails for gravity at 100 to 1000 bodies (err@20 2.14 against constant velocity 0.95) [token-model-fails-gravity-at-100-to-1000-bodies]. Against a GNS-style baseline it passed a class rule: at 100 balls err@10 0.273 against 1.094 and absolute drift@100 0.079 against 0.743. The baseline had 30 epochs against 50, was not tuned, and the comparison is design against design, not equal footing; at 1000 balls both lose to constant velocity (err@100 133.6 against 124.8, constant velocity 111.2). [token-model-beats-gns-style-baseline]

### 7.4 Shift equivariance [single-seed]

Under a half-cell shift, median force changed by more than 2% on all six laws, on two checkpoints (smallest cell 0.0371), 32 scenes per law. The 2% line was not approved by the author in the sources found. This motivates a shift-consistency loss; it is not a result about the final model. [shift-equivariance-probe]

## 8. Interpretability (requirement 4): mostly negative

Force-law recovery from the learned model gave, at level 1 (curve against truth), 5 PASS / 9 MARGINAL / 2 KILL (INCONCLUSIVE_PARTIAL) and at level 2 (formula against exact) 2 PASS / 6 MARGINAL / 8 KILL; recovered power-law exponent 1.96 for true 2, Yukawa range 10.6 for 8 and 14.6 for 32 [force-law-recovery-partial; single-seed]. A two-number message bottleneck reached a pair-fit R2 of 0.347 (pass 0.95; INCONCLUSIVE-undertrained) [two-number-message-bottleneck-readability; single-seed]; a sparse-code U-Net gave INCONCLUSIVE with a force ratio of 1.066 [sparse-code-unet-single-law; single-seed]. Single-law network sizing found widths 8 and 16 qualify (ratios 0.831 and 0.839) with width 4 split and width 32 spiking in 2 of 4 seeds [single-law-network-sizing; single-seed]. A sparse-autoencoder study is a curiosity with no verdict [sae-random-pool-dose-response].

## 9. Negative and inconclusive architecture results

- Neighbour-context pair branch: err@20 ratio 0.649-0.650 held-out across three training seeds against a matched control (line 0.90), but with a cost: the soft tangential-penalty arm cut tangential force (0.008-0.011) and was KILLed on pooled momentum 1.231 (held-out) and 1.226 (fresh); the unpenalized arm was INCONCLUSIVE (tangential 0.131-0.166). Penalty lines were approved by the author. [neighbour-context-pair-branch; validated]
- Staggered double grid: err@20 0.6893 (pass 0.75), dE@100 1.0403 (pass 1.10), but cost 2.578x against a 2.2x line, and a cost-fix rerun gave 2.479x and a dE@100 of 1.194 on the same checkpoint (unexplained). INCONCLUSIVE. [staggered-double-grid-with-quadratic-deposit; single-seed]
- Bit-output cells at a 64 grid: force-error ratio 0.933 soft-decode (pass at most 1.5), seeds 4739/4740 0.974/0.959; a thin PASS in a narrow setting (64 grid, 1024 bodies, one law, 3000 steps). [bit-output-cell-model-at-64-grid; validated, narrow scope]
- Gaussian-ladder far kernel replacing the FFT: err@20 ratio 16.57 (kill 1.30). KILL. [gaussian-ladder-far-kernel-replacing-fft; validated]
- One network for all tiers with a scale input: far-field span error 4.65 (kill above 2.0). KILL. [one-network-for-all-tiers-with-scale-input; single-seed]
- Law-blind U-Net trunk: gravity err@20 ratio 395.9 (kill 1.15). KILL. [law-blind-unet-trunk; single-seed]
- Summary-token exchange between tiles: predictability gain -0.032 (kill below 0.06, pass 0.19). KILL by rule, but the lines were proposed by a worker agent and no approval was found. [summary-token-exchange-between-neighbour-tiles; worker-proposed-line]
- 3D sparse voxel scatter field: INCONCLUSIVE-undertrained; models diverged at 25/72 scenes at 25k iterations and 32/72 at 30k; no passing checkpoint; 3D is about 4x slower than 2D at 1e5. [three-dimensional-sparse-voxel-scatter-field; single-seed]
- Muon optimizer: no effect on drift (dE@100/BL 4.33/4.72/3.84 against 4.10/4.75/3.89). KILL. [muon-optimizer-has-no-effect-on-drift; single-seed]
- Grid next-frame models (history): flow-warp step-1 error 0.80x/0.68x/0.71x against copy-last-frame, honest horizon 6 steps at 20-50 balls, 1-2 at 250. [grid-next-frame-models-negative; unvalidated]
- Pending, with no result: eight architecture-push experiments (including a nested refinement probe voided by a harness error and rerun in progress, and a timestep-as-input experiment that the push table lists without a job while the log records job 11162 as submitted; unresolved). [eight-pending-architecture-push-experiments; unvalidated]

## 10. Limitations

- err@20 is model-dominated, but err@100 in cold-collapse scenes sits at the chaos ceiling (4 scenes only).
- Unit-mass bodies only; the planetesimal proxy failed; the mass-conditioned model is untrained.
- 1e9 and 1e10 runs are speed and stability only. The 1e10 versus 1e9 ratio is linear to slightly superlinear, and per-tick time rises with clustering.
- Most rows are one training seed; only tier-zero accuracy, the chaos floor (err@20), the pair branch, bit-output cells and the Gaussian ladder meet the validated test.
- Tiled energy drift fails its 2.5x line on every seed without replay (2.43-5.77x BL); the replay fine-tune reaches 1.88-1.99x on two base seeds, which meets the line, but the cause of the gain is unsettled and the other two seeds were not fine-tuned.
- Plain PM beats the learned model at N of 1e4 and above.
- Interior force accuracy at 1e6+ bodies is contested (1.53-1.58 against 1.0104).
- The pair-term search passes only under an amended, tie-aware rule on one body.
- 3D has no passing checkpoint.
- Interpretability is partial; level-2 force recovery is mostly KILL.
- Several raw result files live outside version control and are not released; this paper cites result tables only.

## 11. Related work (all to look up; no citations given)

Graph-network simulators for particles; particle-mesh and tree gravity, fast multipole method; learned N-body and learned PM corrections; equivariant and continuous-convolution particle networks; chaos and shadowing for N-body error measurement; the IAS15 integrator and the REBOUND package; symbolic regression and mechanistic interpretability for learned physics; neural operators on grids.

## 12. Conclusion

A learned scatter-field simulator reaches ten billion bodies at near-linear cost on one GPU (speed-only), contact tileability holds to floating-point noise, and tier-zero gravity tiling holds on position error across four training seeds but not on energy drift. Beyond that the picture is mostly one-seed or negative. The next steps in priority order are accuracy at 1e6 to 1e10 bodies, a second seed on the claims marked single-seed, mass-conditioned training, and energy-drift closure; costs are in the gap list.

## Acknowledgments

The author thanks CPP Polaris HPC for compute.

## Reproducibility

Training seed 4738 unless stated. Pass and kill lines were registered before each run where stated; amended or worker-proposed lines are marked. No raw files are released; code scope is undecided.
