# GPU utilization audit (2026-10-14)

GPU: NVIDIA H200 NVL, 143771 MiB (~141 GB), MIG disabled, compute mode Default, power limit 600 W. Node gpu01 has 8 GPUs; QOS gives one GPU per user.

| job type | GPU-hours (sacct, Sep 21-Oct 14, 474 GPU jobs, 136.3 total) | avg util | p95 util | memory used / 141 GB | bottleneck |
|---|---|---|---|---|---|
| training (scatter field, B=16, 300 it) | 94.2 | 71.9% | 100% | 24.8 GB peak (17.6%; includes data gen + eval) | many tiny kernels, 1 CPU thread; power only 181 W avg of 600 W |
| BH / gravity scaling | 31.3 | not measured | not measured | not measured (logs: 121 GB at 1B bodies) | memory-sized, not util-sized; skip |
| eval / regress | 5.5 | 54.5% | 100% | 6.6 GB (4.7%) | batch 1 and 48 at 0.7-3.4 ms/tick, launch/sync bound |
| conservation (1000 ticks, B=4) | incl. above | 44.2% | 100% | 1.3 GB (0.9%) | too short to matter (11 s), launch bound |
| profiler window (train, k=20) | - | 23.6% (profiler-inflated startup) | 100% | 16.5 GB | see below |
| rendering / video (GPU part) | 0.6 | not measured | not measured | not measured | mostly CPU/local |
| other + smoke | 4.7 | not measured | | | |

Util = nvidia-smi `utilization.gpu`, sampled every 500 ms (job 3617, `util/*.csv` in `~/polaris-mcp-files/gpuaudit/res/util`). It is the fraction of time any kernel was running, not SM occupancy.

## Summary

- Slurm accounting cannot give utilization: `gres/gpuutil` and `gres/gpumem` are in AccountingStorageTRES, but TRESUsageInAve/Max are empty for all 590 job rows. Job logs print nvidia-smi only at start. Only the direct measurement above exists.
- Per-job host CPU/elapsed from sacct is 0.93-1.06 cores for all types. One Python thread is saturated, consistent with launch/sync bound work (not proof; spin-waiting on syncs looks the same).
- Training (real trainer, ms_kp_pot_v_g128, batch 16, default `cl,graphtrain`): 300 iters in 29 s, avg 72% "busy" but only 181 W average (30% of the 600 W limit), memory-bandwidth utilization 7.5% avg. Peak memory 24.8 GB of 141 (17.6%).
- torch.profiler on 30 iterations at k=20 (profiler overhead inflates CPU cost, so treat as upper bound for GPU-busy): 22,614 kernels per iteration, mean kernel 6.5 us, GPU-busy 87% of a 166 ms/iter window, CPU time in sync/item calls 3.4 s of 5.0 s window (the `aten::item`/sync entries overlap with kernel time, so this is not additive). The largest kernel is tf32 conv wgrad (0.75 s of 4.35 s). Kernels this small cannot fill 132 SMs; "busy" overstates real SM use.
- Eval/regress: ms-per-tick numbers (0.67 ms at batch 1, N=100-300; 3.3 ms at batch 48) are launch-latency scale, not throughput scale.
- Verdict: the GPU is underused by SM/power/bandwidth measures (30% power, 7.5% memory util, 17% memory capacity for the heaviest measured job) even though nvidia-smi util reads 50-70%. Flag as an anomaly per standing rule.
- Sharing: MIG is disabled (and enabling needs admin). `nvidia-cuda-mps-control`, `dcgmi` and `nsys` exist on gpu01; `ncu` does not. MPS is untested and needs a daemon started inside the job; not tried.
- Untested: whether two concurrent jobs would slow each other. Because per-job kernels are tiny and launch-bound with one saturated host thread each, concurrency plausibly works (cpu=4 per job is the QOS limit, so two jobs cannot be one job each on 4 cores), but the QOS (`QOSMaxGRESPerUser`) forbids a second GPU job anyway, and the one-job rule stands. No concurrent test was run.

## What would raise utilization (untested, within one job)

1. Run 2-4 independent arms/seeds in one process on separate CUDA streams, or as a stacked batch (vmap/`torch.func`) over arms: loss tests, error-correction options, seed sweeps are all small-model jobs with 260k parameters and 25 GB at most. Memory fits 4-5 arms easily.
2. Raise batch from 16 (e.g. 64) and compare it/s; kernels per iteration stay at ~22k while work per kernel grows, so it/s should scale well sublinearly in cost. Changes the optimization trajectory, so only for throughput tests.
3. Whole-iteration CUDA graph / `torch.compile(mode="reduce-overhead")` for the k-step unroll (current `graphtrain` is per step function; 22k kernels per iteration remain).
4. Fuse the many small elementwise/reduce kernels (reduce_kernel 255 ms, elementwise 184 ms, GeluBackward 164 ms, avg_pool backward 145 ms in the profile).

## Measurement design for sharing (needs user approval; not run)

Needs the one-job rule relaxed or a single job that launches two processes. Inside one 1-GPU job: run the 300-iter training alone twice (control, A/A), then two copies concurrently (plain, then under MPS), compare it/s and the sidecar CSV. Accept if concurrent it/s >= 0.8x solo for each copy. Cost ~5 GPU-min. Alternative without any policy change: item 1 above (multi-arm in one process).

## Caveats

- The training window includes data generation and eval, 300 iterations only, one seed (4738). Profiler run used k=20 (late-curriculum worst case), batch 16.
- GPU-hours by type come from job-name regex classification (scripts/gpu_audit_sacct.py); counts include cancelled and failed jobs; BH, render and "other" bins are rough.
- BH scaling jobs (31 GPU-h) were not measured; their cost is memory/force-eval bound and likely higher util, unknown.

Files: scripts/gpu_audit_sacct.py, scripts/gpu_audit_profile.py, scripts/gpu_audit_util_summary.py, job 3617 (`~/augur-gpuaudit-3617.log`), job 3616 (sacct dump). Nothing committed.

## Concurrency test (user-approved; jobs 3619 + 3625, one 1-GPU job each, 2 processes inside)

Workload: scatter trainer, ms_kp_pot_v_g128, batch 16, 400 iters, seed 4738, `cl,graphtrain`; wall = trainer's "trained 400 iters in Ns". Job gets cpus-per-task=4.

| arm | wall per process (s) | slowdown vs solo (39 s) | aggregate throughput vs solo |
|---|---|---|---|
| solo (warm, solo1, solo2) | 39, 39, 39 | 1.00 | 1.00 |
| 2 concurrent, plain | 81, 81 | 2.08x each | 0.96x (no gain) |
| 2 concurrent, MPS | 62, 62 | 1.59x each (vs 39) | 1.26x |
| solo under MPS daemon | 41 | 1.05 | - |

- Plain sharing gives nothing: time-slicing serializes the two processes (and the kernels are too small to overlap). nvidia-smi util during solo 82.7% avg / 206 W; MPS pair 92.6% / 259 W, memory 41 GB total.
- MPS helps modestly (about 1.26x aggregate throughput), not 2x. The GPU is limited by per-process serial kernel-launch work, but two processes still contend for the same SMs once overlapped.
- Job 3619 (first attempt, plain then MPS in one script) hung ~17 min after the plain phase at MPS start and was cancelled at 22:53; the plain numbers above come from it. Rerun as 3625 (MPS only) completed in 4 min. Cause of the hang untested (likely MPS daemon start after processes had already used the GPU in the same job).
- Plain-phase util CSV is contaminated by the hang; not reported.
- Caveat: 400 iters, single run per arm, no repeats beyond solo x3.

Recommendation: concurrent separate jobs/processes are not worth it (plain 0.96x, MPS 1.26x with setup risk). Better lever is multi-arm in one process (stacked batch or streams) or larger batch / fewer launches; untested.
