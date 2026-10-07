# Scatter-field optimization (agent 1)

Branch `scatter-opt` (base `scatter-regress-harness`). All changes are opt-in flags on `ScatterField` (`sf.apply_opt(model, flag)`; `--opt a,b` on `scripts/scatter_regress.py run`, `train_scatter_field.py`, `scatter_opt_bench.py`). Old checkpoints load unchanged. Tools: `scripts/scatter_opt_bench.py` (infer/train/profile), `scripts/scatter_opt_test.py` (parity). Run on Polaris from a private copy `~/bounce-opt` (shared `~/bounce` is used by other agents). H200, Exp E checkpoint, jobs 3300-3309.

## Profile (before)
- Inference B=1, N<=1000: ~420 kernels/tick, GPU time 1.1 ms vs 2.4 ms wall: launch/CPU bound (cudaLaunchKernel 32% of CPU). Convs 37% of GPU.
- N>=3000: `pp_acc` all-pairs (B,N,N) dense + topk dominates (N=1e4 7.7 ms/2.3 GB, 3e4 53 ms/20 GB, 1e5 OOM). Mesh part is N-independent (~0.7 ms graphed).
- Training B32 N100 k20: 277 ms/it, GPU 239 ms: GPU-bound. 16% NCHW<->NHWC layout kernels around cudnn convs, wgrad 17%, elementwise (gelu/avgpool bwd/add) ~20%.

## Changes
| flag | change | predicted | measured | gate |
|---|---|---|---|---|
| kcache | cache FFT of the learned kernel K (inference only, keyed on kmlp param versions) | -0.1 ms/tick | 2.46->2.29 (N100) | PASS (identical) |
| fastio | one index_add / one gather over 4 CIC corners | -0.5 ms | 2.29->1.69 | PASS |
| cl | channels_last UNet (removes layout kernels) | ~0 B1 inference, -15% training | train 277->230 ms/it; infer neutral | PASS (t=1 diff 3e-5) |
| graph | CUDA graph of whole force eval per shape; inference only | 2.2x (GPU 1.1 ms) | 1.73->0.67 ms/tick (N<=1000) | PASS |
| graphtrain | per-unroll-step CUDA graphs (fwd+bwd), `sf.make_step_fns`, `--opt graphtrain` in train script | 1.1-1.45x | 277->246 alone; cl+graphtrain 218 ms/it (1.27x); real recipe 400 s: 3480 -> 4805 iters (1.38x) | retrain err equal or better (below) |
| cell | sorted cell-list kNN for pair term (cell=pp=2.0, 3x3 search, chunked, same top-16 semantics, N>512; host sync so not graphable) | O(N) | force identical to stock (max diff 0.0 at N=1e3/3e3/1e4); N=1e4 7.7->4.0 ms, 3e4 53->15 ms (20 GB->1.5 GB), 1e5 OOM->253 ms (dense clipped-gaussian scene, worst case) | force parity |
| half16 / half / bf16 | fp16 UNet (persistent half copy / autocast) | 1.3x on convs | infer B48: 3.26->2.66 ms (half16), B1 neutral; regress PASS (half) but two_body err@5 +1.5%; TRAINING fp16 autocast NaN (no scaler), grad rel diff 0.23: REJECTED for training | inference optional |

## Gate (scripts/scatter_regress.py compare vs results/scatter_baseline.json)
Stack `kcache,fastio,cl,graph`: OVERALL PASS (results/scatter_opt_stack.json). err@5/10/20 within +-0.2% of baseline in all regimes; tick two_body 4.72->3.26, expB 4.95->3.41, bh300 2.48->0.71, expB_batch1_n100 2.45->0.67 ms. Peak memory in the harness is not comparable: CUDA-graph pools are allocated during warm-up (outside the measured window); real extra = one force's activations (<50 MB).
Rollout parity: single-tick diff ~1e-5 (t=1 max|dpos| 3e-5 for cl; 0 for kcache/graph). 100-tick chaotic scenes (N>=100) amplify any perturbation; eager-vs-eager (atomic index_add order) already differs 5e-5 at t=10 (N=100) and 5e-4 (N=300), so only t<=10 diffs and the accuracy gate are meaningful.

## ms/tick, B=1 (before -> after)
| N | eager | graph stack | cell stack (no graph) |
|---|---|---|---|
| 2 | 2.39 | 0.61 | |
| 100 | 2.46 | 0.67 | |
| 300 | 2.48 | 0.68 | |
| 1e3 | 2.53 | 0.82 | 2.28 |
| 3e3 | 2.81 | 1.45 | 2.47 |
| 1e4 | 7.74 | 6.8 | 4.0 |
| 3e4 | 52.9 | - | 15.4 |
| 1e5 | OOM | - | 253 |

Best choice: graph stack for N<=~4000, cell stack above. B=48 N<=100: 4.7-4.95 -> 3.3-3.4 (half16: 2.7-2.8).

## Training (B=32, N=100, k=20, steps30)
277 ms/it (3.6 it/s) -> 218 ms/it (4.6 it/s) with `--opt cl,graphtrain`; B16 5.1 it/s eager. Real recipe (E warm start, batch 16, steps 30, k 4->20, 400 s budget): eager 3480 it, err@5/10/20 (seed 9000) .0037/.0105/.0341; cl+graphtrain 4805 it, .0037/.0099/.0319 (seed 12000: .0040/.0111/.0368 vs .0039/.0106/.0343).

## Open
- Remaining inference B=1 cost at N<=1000 is GPU (UNet ~0.4 ms); more needs fused kernels (no nvcc/Triton on Polaris).
- cell + graph needs a static candidate cap (avoid host sync); dense cores (BH collapse) make 9*maxocc candidates large.
- bf16/fp16 training needs loss scaling and a retrain-based accuracy check; not done.

## Decision #3 apply (branch worktree-agent-a8d2b3989212e2233)
Defaults: `--opt default` = infer `kcache,fastio,cl,graph,auto`, train `cl,graphtrain`; `none` = eager, `-flag` opts out (sf.resolve_opt/apply_opts); old checkpoints load unchanged (flags are runtime-only). `auto` pair term: N<=512 dense, 513..49999 cell, >=5e4 adapt (threshold sf.adapt_n, from job 3326). `cell` under `graph` uses a static candidate cap (pow2 >= 2x max occupancy at capture, device-side cell_max_occ for overflow check).
Jobs (one GPU at a time): 3352 default-stack gate (done: regress OVERALL PASS vs baseline, bh300 2.48->0.71 ms, train B1 134->61 ms/it, auto infer 1e5 193 ms vs cell 254), 3375 cell-graph parity + auto/cell bench (predict: cap parity <1e-6 when no overflow, N=1e4 graph-cell ~<=2.4 ms), 3376 torch.compile/Triton probe (predict: stock fails on missing Python.h; header-path variants may work; speedup over CUDA graph unlikely, <1.3x).

### Results 3375 (cell-graph) and 3376 (compile probe)
Cell-graph (3375, COMPLETED): no overflow in any case (scene N 1e3..3e4: cap 64/64/128/256 vs max_occ 17/18/45/109; densecore cap 512..8192 vs max_occ 140/436/1368/4073). Pair-term parity eager vs cell (dynamic) 4.8e-7 at N 1e3/3e3/1e4. Whole-force graph-cell vs dynamic-cell max|da| 6.5e-3/1.9e-2/4.4e-2/1.2e-1 (scene), 1.1e-2/5.5e-2/2.4e-1/5.7e-1 (densecore), i.e. 4e-4..1e-4 of max|a|, same as eager-vs-graph (6.2e-3 at N 1e3) so it is likely the fastio/cl conv path, not the cap (unconfirmed: job 3398 prints PPCAP pair-term-only cap-vs-dynamic parity and a compile probe at N 1e3..1e4 with an own-home header copy; was still queued behind other GPU jobs, scripts/polaris_d3_followup.sh). Train default stack: loss 21.984545 vs 21.984537, max grad rel diff 2.9e-3, 60/60 params with grad. Bench B1 (default stack, auto): N 1e3 0.98 ms, 1e4 2.43, 3e4 8.47, 1e5 192.9 (adapt, peak 1.17 GB); 1e5 with graph+cell (static cap) 90.6 ms, peak 50 MB, final_pos_sum 7868.573 vs 7868.518 (auto). Previously cell at 1e5 was 254 ms without graph, so graph+cell is 2.1x faster than auto and 2.8x faster than old cell at 1e5; auto threshold (adapt at >=5e4) should be revisited.
Compile probe (3376, COMPLETED): stock fails (no /usr/include/python3.12/Python.h; Triton's driver.c build fails, so the earlier blocker was real for stock). With C_INCLUDE_PATH pointing at a full Python 3.12 include dir (carla pyenv or conda pkgs copy), torch 2.14.0+cu130 / triton 3.8.0 compile works (PROBE_SIMPLE 2.4e-7). Force ms eager(kcache,fastio,cl)/graph/compiled: N 100 1.62/0.63/0.65, N 1000 1.69/0.91/0.72 (compile ~40 s). Compiled vs graph stack max|da| 4-9e-3 (1e-3 relative). Python.org Include-only headers fail (no pyconfig for 3.12.14); not usable. Compile gives no gain at N 100 and ~1.25x at N 1000 over the CUDA graph; not adopted as default (item 4 fused kernels stay parked).

### Follow-up 3398 (items 3 and 5), COMPLETED
PPCAP (pair term only, cell static cap vs cell dynamic): max|da| 0 for N 1e3/3e3 (scene and densecore) and <=9.5e-7 at 1e4/3e4. The static cap is not the cause of the ~1e-3 whole-force mismatch. CELLGRAPH rerun: graph-cell vs dynamic-cell max|da| 5.6e-3/1.8e-2/3.9e-2/1.1e-1 (scene N 1e3..3e4), 1.7e-2/1.1e-1/1.6e-1/5.2e-1 (densecore); eager vs graph-cell is the same size (6.5e-3, 1.8e-2, 3.8e-2 scene; 1.5e-2, 1.1e-1, 2.5e-1 densecore), so dynamic-cell and eager agree and the residual belongs to the graph-stack/conv path (fastio/cl conv algorithm or graph replay), not the cell cap. Relative to max|a| it is 5e-4..1.6e-3 (scene) and 7e-5..1.5e-4 (densecore). Not isolated further (no run toggling fastio/cl alone under graph); no overflow in any case (max_occ well under cap).
torch.compile re-probe (force ms eager(kcache,fastio,cl) / CUDA graph / compiled; compiled-vs-graph max|da|):
- default: N 1e3 1.689/0.935/0.695 (compile 43 s, 7.1e-3); N 3e3 1.683/1.117/0.711 (13.5 s, 1.1e-2); N 1e4 6.347/2.174/2.334 (2.6e-2).
- max-autotune-no-cudagraphs: N 3e3 1.665/1.122/0.692 (compile 150 s, 7.9e-3); N 1e4 6.352/2.179/2.316 (2.7e-2). N 1e3 result lost (tail -30 truncation in the job script); autotune picked cuBLAS mm over triton for the 64x64 matmuls.
Conclusion: compile is ~1.35x (1e3) and ~1.6x (3e3) faster than the CUDA graph, no gain at 1e4 (2.3 vs 2.2 ms); max-autotune adds nothing over default. Do not make compile a default: gain is small and vanishes at the sizes where cost matters, compile costs 14-150 s, and it requires the header workaround below. Graph+cell (static cap, with the overflow check) is worth keeping as default for inference rollouts (1.8x/1.5x/2.9x over eager at N 1e3/3e3/1e4; 2.1x over auto at 1e5); the ~1e-3 relative graph-stack deviation is below chaos floors but the regress gate should stay on eager for exact comparisons.
Fragile: the probe copied Python 3.12 headers from another user's conda_pkgs (/data/data03/home/davidmalone/conda_caches/conda_pkgs/python-3.12.14-h5f976f7_3_cpython/include/python3.12) into ~/py312hdr_conda. This depends on another user's files and can vanish or change. Before any use of compile, vendor or recreate the headers from an official source (a matching full python 3.12.14 build/install include dir with pyconfig.h), not another user's directory.
