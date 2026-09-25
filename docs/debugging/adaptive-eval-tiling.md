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

(pending: jobs 3017, 3018 queued behind other agents' jobs)
