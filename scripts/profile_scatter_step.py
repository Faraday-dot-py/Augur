"""Time the pieces of one ScatterField training iteration (forward+backward, k=20, batch 32, 2 bodies) on GPU."""
import time

import torch

from scripts import scatter_field as sf
from scripts.train_scatter_field import EXPS, build


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    dev = torch.device("cuda")
    m = build(EXPS["A"], "ms_kp_pot", 0.1).to(dev)
    B, N, k = 32, 2, 20
    pos = (torch.rand(B, N, 2, device=dev) - 0.5) * 10
    vel = torch.randn(B, N, 2, device=dev) * 0.5
    mass = torch.ones(B, N, device=dev)
    mask = torch.ones(B, N, device=dev)
    field = m.init_field(B, dev)

    def parts():
        x = m.scatter(pos, vel, mass, mask)
        return x

    x = parts()
    rho = x[:, :1]
    ak = m.kernel_acc(rho)
    xin = torch.cat([x, ak, field], 1)
    out = {}
    out["scatter"] = timeit(parts)
    out["kernel_acc"] = timeit(lambda: m.kernel_acc(rho))
    out["unet_fwd"] = timeit(lambda: m.net(xin))
    out["pp_acc"] = timeit(lambda: m.pp_acc(pos, mass, mask))
    out["gather"] = timeit(lambda: m.gather(ak, pos))

    def full(amp=None):
        m.zero_grad()
        p, v, f = pos, vel, field
        loss = 0
        ctx = torch.autocast("cuda", dtype=amp) if amp else torch.autocast("cuda", enabled=False)
        with ctx:
            for _ in range(k):
                p, v, _, f, _ = sf._step_m(m, p, v, mass, mask, f)
                loss = loss + (p ** 2).mean() + (v ** 2).mean()
        loss.backward()

    out["full_fwd_bwd_fp32"] = timeit(full, 5)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out["full_fwd_bwd_tf32"] = timeit(full, 5)
    try:
        out["full_fwd_bwd_bf16"] = timeit(lambda: full(torch.bfloat16), 5)
    except Exception as e:
        out["full_fwd_bwd_bf16"] = str(e)[:100]
    for k_, v_ in out.items():
        print(k_, v_, flush=True)


if __name__ == "__main__":
    main()
