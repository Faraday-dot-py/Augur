# Paper gap list

Built 2026-11-17. Ordered by value to a paper. GPU-hours are from logged runs of the nearest job (derived figures marked). No job was run for this list. Cap: 2 GPUs at once.

## Missing experiments, analyses, figures

1. Seeds and error bars on the headline accuracy rows (100k scene, solar system, edge fix): many scenes, 3+ seeds. Inference only; about 0.1-1 GPU-h (analog: stage-1 probes 0.2-1 GPU-h).
2. Accuracy at 1e6 to 3e6 bodies against an exact reference, plus a 4-point scaling curve 1e6 to 1e10 on one setup. Brute force of one force at 3e6 took 224 s (EL:3841). About 0.5-2 GPU-h (estimate, untimed).
3. Resolve the interior force offset conflict (1.53-1.58 vs 1.0104) by re-running both quantities with one verified checkpoint and scene. Under 1 GPU-h (job 7249 took 7 s).
4. Second training seed for single-seed negative and mixed results that carry claims: mixed-law retrain (about 5.9 GPU-h, job 7296), staggered grid, bit-output. About 6 GPU-h each.
5. Equal-footing GNS comparison: tune the baseline, 50 epochs, second seed. About 3-6 GPU-h (analog: 5-6, flagged as analog).
6. Mass-conditioned training for unequal masses and the planetesimal disk. About 5-6 GPU-h (analog only).
7. Plain-PM and Barnes-Hut baselines at 1e4 to 1e5 on the same scenes (small CPU/GPU, under 1 GPU-h).
8. Interpretability: run the two-number bottleneck double-length extension (analog 0.7 GPU-h, job 9617); level-three force-law recovery; second SAE seed (2.4 GPU-h, job 7183).
9. Voxel 3D: extend 30k to 60k iterations, about 25 GPU-h (derived from about 1,200 iterations per hour, job 9623).
10. Tiled energy drift: close the remaining gap to the 2.5x line (replay arms reach 1.88-1.99x); tile-seam and long-run energy figure. About 5-6.4 GPU-h per run (jobs 4041, 4818).
11. Chaos study widening: more scenes, velocity perturbations, apply to the 100k scene. Under 1 GPU-h.
12. Run the 8 pending architecture-push rows and fix the nested refinement harness (job 11163 submitted).
13. Fix repo hygiene for handoff: raw files for 11101, 11098, 9248, 7415, law-blind, bit-output, and tier-0 seeds 4740/4742 live in ~/polaris-mcp-files or on Polaris, not in git; the 1e10 timing file; README supervisor-log line citations are stale; APR rows 9 and 16 lag the log; the timestep-as-input job conflict (APR vs EL:4546).
14. Figures not yet made: architecture diagram, cost-scaling plot, chaos-floor plot, tiled-ratio bar chart across seeds.
15. Related-work search (no GPU).

## Needs-user list

1. Venue and length (page limit, double blind or not).
2. Authorship and affiliation (the project has one human pilot).
3. Headline claim choice from the candidate list in paper-outline.md.
4. Whether to include unvalidated claims (one-seed rows, curiosity results, summary-token KILL on unapproved lines).
5. Open-source release scope: code, checkpoints, results files, videos, Polaris-only raw files.
6. Whether the tie-aware pair-term PASS and the post hoc IAS15 amendment are reported as passes or as as-registered verdicts with amendments.
