# voxel-grokking-plausibility-from-existing-loss-logs: pre-registration (written 2026-11-16, no job)

Label: curiosity (exploration, no pass/kill verdict on grokking itself; main agent approves; APPROVED by Pilot 2026-11-16 12:11). Question (Pilot): could the voxel line benefit from grokking (a late sharp validation improvement after the train loss has plateaued, usually with weight decay and a small train set)?

Evidence from existing logs (read before writing; local copies /home/awebb/polaris-mcp-files/voxel-train-run.log and voxel-10098/run.log):
- Training set is 1500 fixed scenes, batch 8, N 10..200 (scripts/three_dimensional_sparse_voxel_scatter_field.py training args); optimiser is plain Adam with no weight decay (scripts/train_scatter_field.py:161). Grokking needs memorisation of a small set first; 60000 its x 8 = 480000 scene draws over 1500 scenes is about 320 epochs, so memorisation is possible in principle.
- Val err@20 curve, first run: 1.53 (it 200), 0.68 (2400), 0.23 (6000), 0.15 (8400), 0.138 (9600) (voxel-train-run.log lines 38-158 sampled). Smooth decreasing, no plateau-then-drop.
- Extension: val 0.27 (it 12000), 0.355 (14400, worse), 0.24 (16800), 0.165 (20400) (voxel-10098/run.log:210-301). The wobble after 10k coincides with the warm restart (lr back to about 0.94x of 5e-4 from about 0.05x; k reset 20 to 7), not with a plateau.
- Gap smoke on CPU, 8 scenes: val/train about 1.0 up to 25k, 1.23 at 30k. No generalisation gap means no memorisation, the grokking precondition. GPU gap mode (96 train, 48 val scenes) in job voxel-divergence-diagnosis-inference-on-checkpoints will confirm.

Verdict: grokking implausible. Honest probability that resuming from 20k yields a grokking-style late val drop: about 3% (guess). The more likely benefit of resuming is plain recovery from the lr warm-restart confound, which is not grokking.

Gate (fixed): a resume-from-20k training run is queued only if GPU gap mode shows val/train >= 3 at 20k (memorisation present) AND the divergence diagnosis hypothesis 1 identifies a cause that does not need an architecture change. Otherwise not queued. Proposed run if gated in (staged, not submitted): resume 20k, STOP_AT 40000, val-curve PASS = val err@20 below 0.5x its 20k value by it 40000 on seeds 4738 val set with no diverged rollout at N = 1000 on the 24-scene set; KILL = val flat within 10% or any N = 1000 divergence; full-state checkpoint every 500 its, SIGTERM trap, --requeue, 9 h segments; est about 20000 its x 3.7 s = 20 GPU-h (from logged 3.71 s/it).
