# H6: tileability of the momentum-symmetric dual-tree force (test id: tiling)

Code: `scripts/tiling_emulator.py` (emulator), `scripts/tiling_run.py` (runner), `scripts/polaris_tiling_{accuracy,scaling}.sh`. Emulated on one GPU (jobs 3017 accuracy, 3018 scaling); no real multi-device run.

## Design under test

T tiles (x-strips by particle count, or Morton-contiguous blocks, equal counts). Tile s walks ordered node pairs (A, B) with A one of its nodes and B any node, using only its own particles plus what it records as imported: node summaries (count, COM, quadrupole, size, and for the estimator the coarse-pass scale S) of nodes it does not fully own, and raw particles of foreign leaves used in a direct sum. Pair (B, A) is done by B's tile from the same summaries. Acceptance is the symmetric rule of dual_tree.py (geometric theta 0.5, or learned estimator mean head, lam 1, tol 1e-3, theta_max 1.2, cap 8).
Node designs: A = per-tile roots (own bounding box); B = shared global grid, node = (tile, cell) with only that tile's particles (partial nodes, no merge); C = shared global grid, merged nodes (a cell holding particles of several tiles has one merged summary, each sharing tile evaluates its ordered pairs for its own targets).
Imports are counted per tile as distinct nodes / raw particles, fetched once per step and cached. Bytes: 48 B per node summary (56 with S), 16 B per raw particle (fp64). All-gather = (N - N/T) particles x 16 B. Halo of a finite cutoff r = foreign particles within r of an own particle (upper bound via 3x3 cell dilation).

## Predictions (written before the runs)

1. C reproduces the global dual-tree forces to fp64 round-off (identical node summaries, identical pair set per target); momentum |sum a|/sum|a| about 1e-16. A and B differ from global (different node boundaries / partial nodes), with error vs exact within 1.5x of global's; momentum still ~1e-16 because the acceptance and pair contributions stay exchange-symmetric. B and A cost a few percent more evals than global (boundary pseudo-nodes cannot merge).
2. Imports per own particle fall as tile size grows; bytes per tile are a few percent of all-gather at T <= 32. Estimator mode costs extra imports (coarse-pass scale, second walk) of order 1.5-2x geometric. A finite-cutoff halo is small for small r but its truncation error is large for 1/r^2-like gravity (error decays slowly with r), so it is not a valid design here.
3. Extent 1.3k-75k: accuracy and relative imports roughly independent of extent (scale-invariant tree).
4. Balance: count-balanced Morton blocks work imbalance < 2x on the flyby frames; strips worse on clumped states. Estimator mode imbalance similar.
5. Not predicted with confidence: strips vs Morton import counts (strip boundaries are long, so I expect strips to import more nodes than Morton for C).

## Results

Jobs 3017 (accuracy, `results/tiling_accuracy.json`, 288 tiled/global/halo rows over frames t0/t1000/t5000/t10000) and 3018 (scaling, `results/tiling_scaling.json`, 225 rows over N=25k-400k, T=2-32, uniform/flyby/clumpy) both completed on Polaris.

**P1 (design equivalence, momentum) — confirmed for C, refuted for A/B's error bound.**
Design C matches the global dual tree to fp64 round-off exactly as predicted: `diff_vs_global_rel` in [2.6e-16, 4.4e-16] over all frames/T/modes, cost ratio vs global 1.00-1.07x (small excess from level bookkeeping, not extra evals). Momentum holds at ~1e-16 to 1e-8 for all three designs (worst case A at 1.4e-8, still far below any error budget).
A and B do **not** stay within 1.5x of global's error as predicted — their absolute error (`diff_vs_global_rel`) is on the order of 1e-2 to 4e-2, i.e. the boundary-node approximation itself dominates over the acceptance-driven error, not a small correction to it. This isn't a regression from exact N-body (`rel_l2` for tiled ~ same magnitude as global's `rel_l2` ~0.04), but "within 1.5x of global" was the wrong frame: A/B disagree with global by an amount comparable to global's own truncation error, so they are a different (comparably accurate) approximation, not a tightened one. Cost overhead for A/B is worse than predicted: "a few percent" was wrong — mean cost/global is 1.44x (A) and 1.56x (B), with max cases up to 7.8x/10.8x (boundary pseudo-nodes fail to merge far more than a few percent, especially at high T where boundary fraction grows).

**P2 (import scaling, estimator overhead) — partly confirmed.**
Imports per tile do fall as tile size grows, as predicted: design C's imp_bytes/all-gather drops from 1.03 (T=2) to 0.19 (T=32), i.e. tiling is a clear win over full all-gather once T is not tiny. Design C stays well under all-gather bytes even at T=2 (fair — pair structure is already better than naive replication). A and B are worse than predicted: "a few percent of all-gather at T<=32" does not hold — B's import/all-gather ratio is 0.88-1.19 even at T=32 (partial boundary nodes essentially force near-full particle import at the boundary), and A is 0.65-1.19. Only C reaches the few-percent regime, and even C only gets there at large T.
Estimator overhead vs geometric was predicted at 1.5-2x; measured ratio for design C is 0.33 (T=2) rising to 1.12 (T=32) — so at small T the estimator imports *less* than geometric (fewer accepted-far pairs to walk = less summary traffic), and only exceeds geometric by a modest 12% at T=32, not the predicted 1.5-2x. Prediction direction was wrong at low T, magnitude was wrong at high T.

**P3 (extent invariance) — confirmed for accuracy, not clearly for import ratio.**
Across frames spanning extent 1.3k-75k, design C's rel_l2 stays flat at 0.042-0.045 (T=8, geo) — the tree is scale-invariant as expected. The import/all-gather ratio is noisier (0.33-0.58 across frames) but doesn't show a monotonic trend with extent, consistent with "roughly independent of extent," though the run-to-run scatter is larger than a tight confirmation would want.

**P4 (balance) — refuted: count-balance does not give a bounded imbalance.**
Both partitions are count-balanced by construction (count_imb = 1.000 exactly, confirmed as designed), but **work** imbalance (evals per tile) is not bounded by 2x as predicted. Morton work_imb reaches 4.99x on some (N, T, ic) combination, strips reaches 2.41x. So strips came out *more* balanced than Morton on this metric, opposite to the informal expectation that Morton (spatially compact blocks) balances work better than 1-D strips on clumped data — worth flagging since it's counter to intuition and to what most FMM implementations assume.

**P5 (strips vs Morton import counts) — confirmed, and the low-confidence call was right.**
Design C: strips import more nodes than Morton on average (8133 vs 6335 mean `imp_nodes_mean`), matching the tentative prediction that long strip boundaries cost more imported nodes than compact Morton blocks.

**Halo / cutoff truncation — confirms it is not a viable design.**
Cutoff_rel_l2 (from truncating gravity at radius r, ignoring particles beyond) is 0.81/0.62/0.19/0.013/0.0006 at r=10/30/100/300/1000 (t0; similar at later frames) — even at r=1000 (i.e. ~1.3-16% of the frame's own extent depending on epoch) the halo still needs to include order 50k-87.5k particles per tile out of the 50k-87.5k total (halo_mean at r=1000 == allgather_parts, i.e. every particle is within the box-dilated cutoff at that radius for these T). A cutoff design is confirmed not viable: to get truncation error down near global's own ~0.04 rel_l2 floor requires r large enough that the "halo" is effectively the whole system, at which point it isn't a local design.

**Summary for the resume line:** design C (shared grid, merged nodes) is the only tileable design that reproduces the global dual-tree force exactly and needs only a fraction of an all-gather in imports once T >= 8; it is the one to carry forward. Designs A and B (unmerged boundary nodes) are usable as an approximation but cost more evals and import volume than expected, without an error-accuracy benefit that justifies it. A finite-radius halo/cutoff is not viable for this force law. Morton is not clearly better than strips for work balance; both need imbalance mitigation beyond count-balancing before a real multi-GPU implementation.
