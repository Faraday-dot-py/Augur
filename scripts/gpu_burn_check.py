import argparse
import json
import subprocess
import time

import torch


def smi(fields):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().split("\n")[0]
    return [x.strip() for x in out.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=1200)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--mem-frac", type=float, default=0.85)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=4738)
    a = p.parse_args()

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.manual_seed(a.seed)
    n = a.n
    x16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    y16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    x32 = torch.randn(n // 2, n // 2, device=dev, dtype=torch.float32)
    y32 = torch.randn(n // 2, n // 2, device=dev, dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = (x16[:512].float() @ y16.float()).double()

    ballast = []
    chunk = 1 << 30
    while torch.cuda.memory_allocated() + chunk < total * a.mem_frac:
        ballast.append(torch.empty(chunk // 4, device=dev, dtype=torch.float32).normal_())
    bw_src = ballast[0] if ballast else torch.empty(1 << 28, device=dev)
    bw_dst = torch.empty_like(bw_src)

    def timed(fn, iters):
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return time.time() - t

    timed(lambda: x16 @ y16, 3)
    timed(lambda: x32 @ y32, 3)

    start = time.time()
    samples = []
    errors = 0
    while time.time() - start < a.seconds:
        dt16 = timed(lambda: torch.mm(x16, y16), 20)
        tf16 = 20 * 2 * n ** 3 / dt16 / 1e12
        dt32 = timed(lambda: torch.mm(x32, y32), 10)
        tf32 = 10 * 2 * (n // 2) ** 3 / dt32 / 1e12
        dtbw = timed(lambda: bw_dst.copy_(bw_src), 20)
        gbs = 20 * 2 * bw_src.numel() * bw_src.element_size() / dtbw / 1e9
        chk = (x16[:512].float() @ y16.float()).double()
        if not torch.equal(chk, ref):
            errors += 1
        t, w, sm_clk, mem_used = smi("temperature.gpu,power.draw,clocks.sm,memory.used")
        s = dict(t=time.time() - start, bf16_tflops=tf16, fp32_tflops=tf32, copy_gbs=gbs,
                 temp_c=float(t), power_w=float(w), sm_mhz=float(sm_clk), mem_used_mib=float(mem_used),
                 mismatches=errors)
        samples.append(s)
        print(json.dumps(s), flush=True)

    def med(k):
        v = sorted(s[k] for s in samples)
        return v[len(v) // 2]

    summary = dict(
        gpu=name, uuid=smi("uuid")[0], bus=smi("pci.bus_id")[0], samples=len(samples),
        bf16_tflops_median=med("bf16_tflops"), fp32_tflops_median=med("fp32_tflops"),
        copy_gbs_median=med("copy_gbs"), temp_c_max=max(s["temp_c"] for s in samples),
        power_w_median=med("power_w"), sm_mhz_median=med("sm_mhz"),
        mem_used_mib_max=max(s["mem_used_mib"] for s in samples), mismatches=errors,
    )
    print("SUMMARY " + json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(summary=summary, samples=samples), f)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
