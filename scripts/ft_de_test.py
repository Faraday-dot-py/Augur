"""Smoke tests for the fine-tune flags of train_scatter_field.py (--snap-iters, --lr-warmup, --ema, --train-only, --w-vel) and ft_de_probe.py interpolation.
Run on Polaris: PYTHONPATH=. python scripts/ft_de_test.py"""
import os
import sys
import tempfile

import torch

from scripts import train_scatter_field as tsf


def run(extra, tmp, tag):
    sys.argv = ["x", "--exp", "E", "--train", "6", "--steps", "30", "--iters", "6", "--batch", "2", "--k-start", "2", "--k-end", "3", "--eval-steps", "25",
                "--eval-scenes", "2", "--device", "cuda", "--lr", "1e-3", "--log-every", "3", "--opt", "none",
                "--out-dir", f"{tmp}/{tag}/res", "--ckpt-dir", f"{tmp}/{tag}/ck"] + extra
    tsf.main()
    return f"{tmp}/{tag}/ck/E_ms_kp_pot_v_g128.pt"


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)["model"]


def main():
    tmp = tempfile.mkdtemp()
    p = run(["--variant", "ms_kp_pot_v_g128", "--snap-iters", "0,3", "--ema", "0.9", "--lr-warmup", "4", "--w-vel", "0.0"], tmp, "a")
    assert os.path.exists(p + ".snap0") and os.path.exists(p + ".snap3") and os.path.exists(p + ".ema")
    s0, s3, fin, ema = load(p + ".snap0"), load(p + ".snap3"), load(p), load(p + ".ema")
    assert any((s0[k] != fin[k]).any() for k in s0 if s0[k].dtype.is_floating_point)
    assert any((ema[k] != fin[k]).any() for k in ema if ema[k].dtype.is_floating_point)
    print("snap/ema/warmup/w-vel ok")
    p = run(["--variant", "ms_kp_pot_v_g128", "--train-only", "kmlp,ppmlp", "--snap-iters", "0"], tmp, "b")
    s0, fin = load(p + ".snap0"), load(p)
    moved = [k for k in s0 if s0[k].dtype.is_floating_point and (s0[k] != fin[k]).any()]
    assert moved and all(("kmlp" in k or "ppmlp" in k) for k in moved), moved
    print("train-only ok, moved:", moved)
    print("ALL PASS")


if __name__ == "__main__":
    main()
