"""Tests for the fed-back-field softcap flag (ScatterField.softcap; run on Polaris).

  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/scatter_softcap_test.py dump <out.pt> [E ckpt]   E rollout, flag untouched (run once on the pre-change commit f0e6238, once on this branch)
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/scatter_softcap_test.py test <ref.pt> [E ckpt]   E identity vs the pre-change dump + flag semantics + ckpt/auto resolution
The scatter uses index_add_ (atomics), so E is only bit-reproducible with deterministic algorithms on; the identity check also reports the run-to-run noise floor.
"""
import os
import sys
import tempfile

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scenes as ss

torch.use_deterministic_algorithms(True, warn_only=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CK = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
OK = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)
    OK.append(bool(cond))


def scene(n, seed=4738):
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.uniform(-6, 6, (2, n, 2)), dtype=torch.float32, device=dev)
    vel = torch.tensor(rng.uniform(-2, 2, (2, n, 2)), dtype=torch.float32, device=dev)
    return pos, vel, torch.ones(2, n, device=dev), torch.ones(2, n, device=dev)


def rollout(model, n=64, steps=50):
    with torch.no_grad():
        return sf.rollout(model, *scene(n), steps)


def main():
    mode, path = sys.argv[1], sys.argv[2]
    model = ss.build_model("E", ss.BOX_DT, dev, CK)
    P, V = rollout(model)
    if mode == "dump":
        Pn, Vn = rollout(model)
        torch.save((P.cpu(), V.cpu(), Pn.cpu()), path)
        print("dumped", path, tuple(P.shape), "self-repeat max diff", float((P - Pn).abs().max()))
        return
    Pr, Vr, Prn = torch.load(path)
    P_b, V_b = rollout(model)
    old_noise = float((Pr - Prn).abs().max())
    new_noise = float((P.cpu() - P_b.cpu()).abs().max())
    d = float((P.cpu() - Pr).abs().max())
    check("E flag off: positions bitwise == pre-change commit", torch.equal(P.cpu(), Pr), f"max diff {d:.1e}; run-to-run noise old {old_noise:.1e} new {new_noise:.1e}")
    check("E flag off: velocities bitwise == pre-change commit", torch.equal(V.cpu(), Vr))
    check("default softcap is 0", model.softcap == 0.0)
    model.softcap = 100.0
    P2, V2 = rollout(model)
    check("softcap 100 on E: finite, differs from off only slightly (field max ~52)", bool(torch.isfinite(P2).all()) and 0 < float((P2 - P).abs().max()) < 1.0, f"max {float((P2 - P).abs().max()):.2e}")
    model.softcap = 0.0
    P3a, _ = rollout(model, steps=3)
    model.softcap = 1e9
    P3, _ = rollout(model, steps=3)
    check("softcap 1e9 ~ identity (3 ticks; float32 rounding is chaos-amplified over 50)", float((P3 - P3a).abs().max()) < 1e-4, f"max {float((P3 - P3a).abs().max()):.1e}")
    model.softcap = 0.0
    P4, V4 = rollout(model)
    check("flag back to 0 restores bitwise (no state leak)", torch.equal(P4, P) and torch.equal(V4, V), f"max {float((P4 - P).abs().max()):.1e}")
    from scripts import scatter_contact_scenes as cs

    sd = torch.load(CK, map_location="cpu", weights_only=False)["model"]
    tmp = tempfile.mkdtemp()
    plain, meta = os.path.join(tmp, "plain.pt"), os.path.join(tmp, "meta.pt")
    torch.save({"model": sd}, plain)
    torch.save({"model": sd, "softcap": 50.0}, meta)
    check("auto: E ckpt without metadata -> off", cs.resolve_softcap("E", plain, "auto") == 0.0)
    check("auto: option-C ckpt without metadata -> 100", cs.resolve_softcap("C", plain, "auto") == cs.C_SOFTCAP == 100.0)
    check("auto: B/A ckpt without metadata -> off", cs.resolve_softcap("B", plain, "auto") == 0.0 and cs.resolve_softcap("A", plain, "auto") == 0.0)
    check("auto: ckpt metadata wins", cs.resolve_softcap("C", meta, "auto") == 50.0 and cs.resolve_softcap("E", meta, "auto") == 50.0)
    check("explicit value/0 overrides", cs.resolve_softcap("C", meta, "0") == 0.0 and cs.resolve_softcap("E", plain, "200") == 200.0)
    check("default build() is off for every model", cs.build("C", dev, plain).softcap == 0.0 and cs.build("E", dev, plain).softcap == 0.0)
    check("build(softcap='auto') C on, E off", cs.build("C", dev, plain, "auto").softcap == 100.0 and cs.build("E", dev, plain, "auto").softcap == 0.0)
    print("ALL PASS" if all(OK) else f"FAILURES {OK.count(False)}")
    sys.exit(0 if all(OK) else 1)


main()
