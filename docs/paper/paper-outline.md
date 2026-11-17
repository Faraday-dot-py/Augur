# Paper outline (draft)

Built 2026-11-17 from docs/paper/paper-claims-and-evidence-table.md (the table; row names below are its claim names). Every number here is in a saved file cited in the table. No related-work citation has been looked up yet.

## Title options

1. Augur: a learned particle simulator whose cost grows linearly with body count
2. One model, any region: tileable learned N-body simulation with pre-registered negative results
3. What a learned gravity simulator can and cannot do at a billion bodies

## Section skeleton

1. Introduction: four requirements (generalization, tileable, O(N), interpretability); what is and is not shown.
2. Method: scatter field (deposit onto a grid, U-Net, gather; FFT kernel with edge fix); tiering and ghost cells; token-per-ball model as the contact-model comparison.
3. Evaluation protocol: exact reference validated against IAS15 (an adaptive N-body integrator); err@k and dE@k; chaos floor; pre-registered lines; fresh-reviewer process.
4. Results: cost; accuracy and chaos limits; tiling; generalization; interpretability (partial).
5. Negative and inconclusive results (own section).
6. Limitations.
7. Related work, conclusion, reproducibility and release.

## Headline structure (Pilot decision 2026-11-17)

Main headlines:
1. Linear cost at scale: scatter-field-linear-cost-to-ten-billion-bodies (speed and memory only, no accuracy at that size; marker speed-only) and token-model-linear-cost-to-one-billion-balls (speed-only).
2. Tileability: scatter-field-ghost-cell-tiling-for-gravity, token-model-strip-tiling-equals-global (fp32 noise, exact translation equivariance), tiled-closed-loop-tier-zero-accuracy (err@20 passes on 4 of 4 training seeds) with tiled-closed-loop-tier-zero-energy-drift-fails stated beside it.

Secondary:
- token-model-size-generalization-to-fifty-thousand-balls (trained at 200 balls or fewer)
- visited-state-label-replay-cuts-drift (4.25 to 1.99 and 4.78 to 1.88; truth-state control similar, cause not settled)
- learned-vs-plain-particle-mesh-force-at-n1000 (learned wins 1.8-4.6x median/p90 at N=1000, loses at 1e4 and above)

Supporting methodology: chaos-floor-err20-is-model-dominated, chaos-floor-cold-collapse-err100-at-ceiling, reference-simulator-validated-against-ias15 (as registered INCONCLUSIVE, amended PASS), pre-registered-kill-lines-and-fresh-reviewer-method, token-model-beats-gns-style-baseline (design versus design).

Every claim row carries a marker (validated, single-seed, worker-proposed-line, approved-line-2026-11-17, post-hoc-amended, speed-only, unvalidated); the draft repeats the marker beside each claim. Verdicts are reported as pre-registered, with amendments (tie-aware pair-term rule, post hoc IAS15 line) labelled as amendments, never as plain passes. Code and videos are released; no raw files, no checkpoints; the paper cites result tables. Sole author; acknowledgment that CPP Polaris HPC provided compute. Full text is in docs/paper/paper-draft.md.

## Negative-results section

tiled-closed-loop-tier-zero-energy-drift-fails; upper-tier-hierarchy-adds-nothing-end-to-end; mixed-six-law-retrain-fails-gravity; unit-mass-only-and-planetesimal-disk-proxy-kill; gaussian-ladder-far-kernel-replacing-fft; one-network-for-all-tiers-with-scale-input; law-blind-unet-trunk; summary-token-exchange-between-neighbour-tiles (lines approved 2026-11-17, after the runs); three-part-separate-then-joint-training-kill; muon-optimizer-has-no-effect-on-drift; token-model-fails-gravity-at-100-to-1000-bodies; grid-next-frame-models-negative; learned-vs-plain-particle-mesh-force-at-n1000 (loses at N of 1e4 and above); the mixed-result rows (neighbour-context-pair-branch, staggered-double-grid-with-quadratic-deposit, bit-output-cell-model-at-64-grid).

## Honest limitations (must appear)

- err@20 is model-dominated per the chaos-floor study; err@100 in cold-collapse scenes is at the chaos ceiling (4 scenes only), so err@100 cannot rank models there.
- Unit-mass bodies only; planetesimal proxy failed; mass-conditioned model untrained.
- 1e9 and 1e10 runs are speed and stability only, with no accuracy; the 1e10/1e9 time ratio is from two runs with different tick counts.
- Interpretability is partial: force-law recovery level 1 INCONCLUSIVE_PARTIAL, level 2 KILL; no passing readability design.
- Most rows are one training seed; tier 0 is the exception (4 seeds).
- Tiled energy drift fails its line on every seed even after the replay fine-tune.
- Plain particle mesh beats the learned model at N of 1e4 and above.
- Pair-term neighbour search passes only under a tie-aware rule written after the KILL.
- 3D voxel version has no passing checkpoint.
- Interior force offset: the 1.53-1.58 edge-check ratios are explained as a reference-sum defect (636,068 of 1,000,000 bodies counted); interior ratio 1.0104 at tick 0 and 0.9596 at tick 11; registered verdict INCONCLUSIVE; single scene, seed 4738.

## Related work to find (all "to look up"; no citations invented)

- Graph-network simulators for particles (the GNS family)
- Particle-mesh and tree (Barnes-Hut) gravity; fast multipole method
- Learned N-body and learned PM correction papers
- Equivariant and continuous-convolution particle networks
- Chaos and shadowing for N-body error measurement
- IAS15 integrator and the REBOUND package (versions used: REBOUND 5.2.2 per EL:3503)
- Symbolic regression and mechanistic interpretability of learned physics
- Neural operators on grids

## Figure and table inventory (existing files; anything not listed must be flagged)

| item | source file | status |
| --- | --- | --- |
| Cost scaling, token model | results/bench_scaling_big.json (plus bench_scaling_cuda.json for 1M) | exists |
| 1e9 and 1e10 scatter-field timing | results/bh_1e9_edge_free.json; docs/debugging/scatter-bh-1b.md | exists; 1e10 file not located in repo (EL:3393 only), flag |
| 100k accuracy curve | results/showcase_bh_100k_kpadext.json | exists |
| Edge artifact before and after | results/bh_edge_zero_1e6_edgecheck.json, bh_edge_ext_1e6_edgecheck.json | exists |
| Chaos floor table | results/chaos_bh.json, chaos_expb.json | exists |
| Tier-0 tiled ratios, four seeds | results/tier0_second_training_seed/; seeds 4740/4742 only on Polaris | partial |
| Pair-branch and staggered results | ~/polaris-mcp-files/ (outside git) | needs copying into repo |
| Showcase videos and grid | videos/, videos/showcase_solar_system_grid_kpadext.png | exists, untracked |
| Architecture-push summary table | docs/debugging/architecture-push-results.md | exists |
| Architecture diagram | docs/architectures.md text only | needs figure |
| Voxel divergence | results/voxel_divergence_diagnosis/scoring.json | exists |
