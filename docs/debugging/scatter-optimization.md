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
