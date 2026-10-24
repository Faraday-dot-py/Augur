"""Scene builders for the tiered-architecture generators and eval harness. Tier-0 scenes live in a window of
WINDOW = TILE + 2*HALO (centred coordinates, [-32, 32)^2). `build_batch` pads scenes of one kind into tensors plus a Physics.
All randomness is a numpy Generator, so (kind, seed) reproduces a scene. Seed 4738 for training, 9100/9200/9300 held out."""
import math

import numpy as np
import torch

from scripts import tiered_truth as tt

TILE, HALO = 32.0, 16.0
WINDOW = TILE + 2 * HALO
GC = 32
TIER0_KINDS = ["grav", "lj_gas", "lj_liquid", "lj_solid", "contact", "bonded", "collision", "boundary", "edge"]
LJ_PHASE = dict(lj_gas=(0.05, 1.0, 0.15), lj_liquid=(0.70, 0.60, 0.05), lj_solid=(0.95, 0.30, 0.02))
BALL_R, BALL_K, BALL_G, BALL_V = 0.75, 400.0, 9.0, 2.3
LJ_TIMING = dict(dt=0.1, sub=20)
BOUND_KINDS = ("wall", "open", "periodic")


def hex_lattice(nx, ny, a):
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    p = np.stack([(ix + 0.5 * (iy % 2)).ravel() * a, iy.ravel() * a * math.sqrt(3) / 2], 1)
    return p - p.mean(0)


def lj_spacing(rho):
    return math.sqrt(2.0 / (math.sqrt(3) * rho))


def maxwell(n, temp, rng):
    v = rng.normal(0.0, 1.0, (n, 2))
    v -= v.mean(0)
    return v * math.sqrt(temp * 2 * n / max((v ** 2).sum(), 1e-9))


def irregular_blob(rng, a=1.0, r_range=(2.0, 6.0), n_discs=(1, 3)):
    """Hex-lattice points inside a union of random discs: an irregular body, centred on its centre of mass."""
    k = int(rng.integers(n_discs[0], n_discs[1] + 1))
    centres = np.concatenate([[[0.0, 0.0]], rng.uniform(-1, 1, (k - 1, 2)) * r_range[1] * 0.7])
    radii = rng.uniform(r_range[0], r_range[1], k)
    n = int(2 * r_range[1] * 2.5 / a) + 4
    p = hex_lattice(n, n, a)
    ok = np.zeros(len(p), bool)
    for c, r in zip(centres, radii):
        ok |= ((p - c) ** 2).sum(1) < r * r
    p = p[ok]
    return p - p.mean(0)


def bonds_from(p, cutoff):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    i, j = np.nonzero(np.triu(d < cutoff, 1))
    return np.stack([i, j], 1), d[i, j]


def spin_vel(p, omega):
    return omega * np.stack([-p[:, 1], p[:, 0]], 1)


def _empty(kind, dt, sub):
    return dict(kind=kind, dt=dt, sub=sub, bonds=np.zeros((0, 2), int), brest=np.zeros(0), bk=np.zeros(0), bd=np.zeros(0),
                bound=dict(kind="none"), law={}, tags={})


def grav_scene(rng, n_max=256, kind="grav"):
    n = int(rng.integers(8, n_max + 1))
    nc = int(rng.integers(1, 4))
    cen = rng.uniform(-14, 14, (nc, 2))
    sg = rng.uniform(2.0, 10.0, nc)
    lab = rng.integers(0, nc, n)
    pos = np.clip(cen[lab] + rng.normal(0, 1, (n, 2)) * sg[lab, None], -30, 30)
    mass = np.exp(rng.uniform(np.log(0.01), np.log(5.0), n))
    if rng.random() < 0.3:
        mass[0] = rng.uniform(10, 40)
    vf = rng.uniform(0.1, 0.8)
    vel = rng.normal(0, 1, (n, 2)) * vf * math.sqrt(mass.sum() / (4 * sg.mean()))
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    s = _empty(kind, 0.1, 4)
    s.update(pos=pos, vel=vel, mass=mass, kmat=np.ones(n), law=dict(grav=dict(eps=tt.EPS, g=1.0)))
    return s


def lj_scene(rng, phase, n_max=1024, kind=None):
    rho, T, jit = LJ_PHASE[phase]
    a = lj_spacing(rho)
    w, h = rng.uniform(14, 44, 2)
    while True:
        nx, ny = max(2, int(w / a)), max(2, int(h / (a * math.sqrt(3) / 2)))
        if nx * ny <= n_max:
            break
        w, h = w * 0.9, h * 0.9
    p = hex_lattice(nx, ny, a)
    p = p + rng.normal(0, jit * a, p.shape) + rng.uniform(-3, 3, 2)
    n = len(p)
    s = _empty(kind or phase, LJ_TIMING["dt"], LJ_TIMING["sub"])
    s.update(pos=p, vel=maxwell(n, T, rng), mass=np.ones(n), kmat=np.ones(n), law=dict(lj=dict(rc=tt.LJ_RC)),
             tags=dict(rho=rho, T=T, equil=(T, 200)))
    return s


def poisson_pts(rng, n, lo, hi, dmin):
    pos = []
    for _ in range(n * 60):
        q = rng.uniform(lo, hi)
        if all(np.linalg.norm(q - r) > dmin for r in pos):
            pos.append(q)
        if len(pos) == n:
            break
    return np.array(pos)


def contact_scene(rng, n_max=40, kind="contact", box=None):
    n = int(rng.integers(2, min(n_max, 40) + 1))
    L = rng.uniform(8, 20)
    lo, hi = box if box is not None else (np.full(2, -L), np.full(2, L))
    pos = poisson_pts(rng, n, lo, hi, 2 * BALL_R * 1.05)
    n = len(pos)
    g = np.zeros(2)
    if rng.random() < 0.5:
        g[rng.integers(0, 2)] = BALL_G * rng.choice([-1, 1])
    s = _empty(kind, 0.15, 8)
    s.update(pos=pos, vel=rng.uniform(-BALL_V, BALL_V, (n, 2)), mass=np.ones(n), kmat=np.ones(n),
             law=dict(contact=dict(radius=BALL_R, k=BALL_K), g_vec=g))
    return s


def _bonded_body(rng, mat_k=None, mat_d=None):
    p = irregular_blob(rng)
    idx, rest = bonds_from(p, 1.2)
    k = mat_k if mat_k is not None else math.exp(rng.uniform(math.log(30), math.log(400)))
    d = mat_d if mat_d is not None else rng.uniform(0.0, 1.0)
    return p, idx, rest, k, d


def _assemble_bodies(bodies, dt=0.05, sub=10, kind="bonded"):
    pos, vel, bonds, rest, bk, bd, km = [], [], [], [], [], [], []
    off = 0
    for p, v, idx, r, k, d in bodies:
        pos.append(p), vel.append(v)
        bonds.append(idx + off), rest.append(r), bk.append(np.full(len(r), k)), bd.append(np.full(len(r), d))
        km.append(np.full(len(p), k / 100.0))
        off += len(p)
    s = _empty(kind, dt, sub)
    s.update(pos=np.concatenate(pos), vel=np.concatenate(vel), mass=np.ones(off), kmat=np.concatenate(km),
             bonds=np.concatenate(bonds), brest=np.concatenate(rest), bk=np.concatenate(bk), bd=np.concatenate(bd),
             law=dict(contact=dict(radius=0.45, k=200.0), bonded=True))
    return s


def bonded_scene(rng, n_max=256, kind="bonded"):
    nb = int(rng.integers(1, 3))
    bodies = []
    for i in range(nb):
        p, idx, r, k, d = _bonded_body(rng)
        p = p + np.array([(-1) ** i * 9.0, 0.0]) * (nb - 1) + rng.uniform(-3, 3, 2)
        v = rng.normal(0, 0.5, 2) + spin_vel(p - p.mean(0), rng.normal(0, 0.15))
        bodies.append((p, v, idx, r, k, d))
    s = _assemble_bodies(bodies, kind=kind)
    if len(s["pos"]) > n_max:
        return bonded_scene(rng, n_max, kind)
    return s


def collision_scene(rng, n_max=512, kind="collision"):
    typ = rng.choice(["bonded", "lj_droplet", "grav"])
    b_frac, v_rel = rng.uniform(0, 1), rng.uniform(0.5, 4.0)
    om = rng.normal(0, 0.15, 2)
    if typ == "bonded":
        ps = [_bonded_body(rng) for _ in range(2)]
        R = [np.linalg.norm(p[0], axis=1).max() for p in ps]
        b = b_frac * (R[0] + R[1])
        sep = R[0] + R[1] + 8
        bodies = []
        for i, (p, idx, r, k, d) in enumerate(ps):
            sg = 1 - 2 * i
            q = p + np.array([-sg * sep / 2, sg * b / 2])
            v = np.array([sg * v_rel / 2, 0.0]) + spin_vel(p, om[i])
            bodies.append((q, v, idx, r, k, d))
        s = _assemble_bodies(bodies, kind=kind)
        s["tags"] = dict(b=float(b), v=float(v_rel), omega=om.tolist(), material=[float(ps[0][3]), float(ps[1][3])], body="bonded")
    elif typ == "lj_droplet":
        a = lj_spacing(0.8)
        bodies = []
        pr = []
        for i in range(2):
            rad = rng.uniform(3, 7)
            p = hex_lattice(int(2 * rad / a) + 3, int(2 * rad / a) + 3, a)
            p = p[np.linalg.norm(p, axis=1) < rad]
            pr.append((p - p.mean(0), rad))
        b = b_frac * (pr[0][1] + pr[1][1])
        sep = pr[0][1] + pr[1][1] + 6
        pos, vel = [], []
        for i, (p, rad) in enumerate(pr):
            sg = 1 - 2 * i
            pos.append(p + np.array([-sg * sep / 2, sg * b / 2]))
            vel.append(np.array([sg * v_rel / 2, 0.0]) + spin_vel(p, om[i]) + maxwell(len(p), 0.1, rng))
        n = sum(len(p) for p in pos)
        s = _empty(kind, LJ_TIMING["dt"], LJ_TIMING["sub"])
        s.update(pos=np.concatenate(pos), vel=np.concatenate(vel), mass=np.ones(n), kmat=np.ones(n), law=dict(lj=dict(rc=tt.LJ_RC)))
        s["tags"] = dict(b=float(b), v=float(v_rel), omega=om.tolist(), material="lj", body="lj_droplet")
    else:
        pos, vel, mass = [], [], []
        sgs = []
        for i in range(2):
            n = int(rng.integers(20, max(21, n_max // 2)))
            sg = rng.uniform(1.5, 4.0)
            p = rng.normal(0, sg, (n, 2))
            m = np.exp(rng.uniform(np.log(0.05), np.log(1.0), n))
            vv = rng.normal(0, 1, (n, 2)) * 0.3 * math.sqrt(m.sum() / (4 * sg)) + spin_vel(p, om[i])
            pos.append(p), vel.append(vv), mass.append(m), sgs.append(sg)
        b = b_frac * 2 * (sgs[0] + sgs[1])
        sep = 4 * (sgs[0] + sgs[1]) + 6
        for i in range(2):
            sg = 1 - 2 * i
            pos[i] = pos[i] + np.array([-sg * sep / 2, sg * b / 2])
            vel[i] = vel[i] + np.array([sg * v_rel / 2, 0.0])
        m = np.concatenate(mass)
        s = _empty(kind, 0.1, 4)
        s.update(pos=np.concatenate(pos), vel=np.concatenate(vel), mass=m, kmat=np.ones(len(m)), law=dict(grav=dict(eps=tt.EPS, g=1.0)))
        s["tags"] = dict(b=float(b), v=float(v_rel), omega=om.tolist(), material="grav", body="grav")
    s["pos"] = np.clip(s["pos"], -30, 30)
    return s


def occupancy(bound, G=GC, extent=WINDOW, ss=4):
    """(3, G, G): [wall fraction, open-void fraction, periodic flag] on the window grid, supersampled ss x ss per cell."""
    occ = np.zeros((3, G, G))
    k = bound["kind"]
    if k == "periodic":
        occ[2] = 1.0
        return occ
    if k == "none":
        return occ
    h = extent / G
    c = (np.arange(G * ss) + 0.5) * (h / ss) - extent / 2
    X, Y = np.meshgrid(c, c, indexing="xy")
    lo, hi = np.asarray(bound["lo"]), np.asarray(bound["hi"])
    out = ((X < lo[0]) | (X > hi[0]) | (Y < lo[1]) | (Y > hi[1])).astype(float)
    f = out.reshape(G, ss, G, ss).mean((1, 3))
    occ[0 if k == "wall" else 1] = f
    return occ


def boundary_scene(rng, n_max=256, bkind=None, base=None, kind="boundary"):
    bkind = bkind or str(rng.choice(BOUND_KINDS))
    base = base or str(rng.choice(["lj_gas", "contact", "grav"]))
    hb = rng.uniform(8, 30)
    ctr = rng.uniform(-3, 3, 2)
    P = float(rng.choice([32.0, 48.0, 64.0]))
    lo, hi = (np.full(2, -P / 2), np.full(2, P / 2)) if bkind == "periodic" else (ctr - hb, ctr + hb)
    if base == "contact":
        s = contact_scene(rng, n_max, box=(lo + 1.0, hi - 1.0))
    elif base == "lj_gas":
        s = lj_scene(rng, "lj_gas", n_max)
        n = min(len(s["pos"]), max(4, int(0.08 * np.prod(hi - lo))))
        s["pos"] = poisson_pts(rng, n, lo + 1.0, hi - 1.0, 1.1)
        n = len(s["pos"])
        s.update(vel=maxwell(n, 1.0, rng), mass=np.ones(n), kmat=np.ones(n))
    else:
        s = grav_scene(rng, n_max)
        s["pos"] = np.clip(s["pos"] * (hb / 30.0 + 0.2) * (1.0 if bkind != "periodic" else 0.8), lo + 1.0, hi - 1.0)
    if bkind == "periodic":
        s["bound"] = dict(kind="periodic", period=np.array([P, P]))
    else:
        s["bound"] = dict(kind=bkind, lo=lo, hi=hi)
        if bkind == "wall":
            s["law"]["walls"] = dict(radius=BALL_R, k=BALL_K)
    s["kind"] = kind
    s["tags"] = dict(s.get("tags", {}), base=base, bkind=bkind)
    s["tags"].pop("equil", None)
    return s


def edge_scene(rng, n_max=256, kind="edge"):
    inward = bool(rng.random() < 0.5)
    sgn = float(rng.choice([-1.0, 1.0]))
    v = rng.uniform(1.5, 4.0)
    if rng.random() < 0.5:
        s = lj_scene(rng, "lj_gas", n_max, kind=kind)
        n = min(len(s["pos"]), 40)
        s["pos"] = poisson_pts(rng, n, np.array([-6.0, -12.0]), np.array([6.0, 12.0]), 1.1)
        n = len(s["pos"])
        s.update(vel=maxwell(n, 0.3, rng), mass=np.ones(n), kmat=np.ones(n))
    else:
        s = grav_scene(rng, min(n_max, 128), kind=kind)
        s["pos"] = s["pos"] * 0.3
    x0 = sgn * (HALO + 2 if inward else TILE / 2 - 6)
    s["pos"] = s["pos"] * np.array([0.5, 1.0]) + np.array([x0, rng.uniform(-8, 8)])
    s["vel"] = s["vel"] + np.array([-sgn * v, 0.0])
    s["pos"] = np.clip(s["pos"], -30, 30)
    s["tags"] = dict(s.get("tags", {}), inward=inward, v=float(v))
    s["tags"].pop("equil", None)
    return s


def make_scene(kind, rng, n_max=256):
    if kind == "grav":
        return grav_scene(rng, n_max)
    if kind in LJ_PHASE:
        return lj_scene(rng, kind, max(n_max, 256))
    return dict(contact=contact_scene, bonded=bonded_scene, collision=collision_scene, boundary=boundary_scene, edge=edge_scene)[kind](rng, n_max)


def build_batch(scenes, device, dtype=torch.float64):
    """Pad same-kind scenes into tensors and a Physics. Returns dict(pos, vel, mass, mask, phys, occ, dt, sub, tags, kind)."""
    B = len(scenes)
    N = max(len(s["pos"]) for s in scenes)
    z = lambda *sh: torch.zeros(*sh, device=device, dtype=dtype)
    pos, vel, mass, mask, kmat = z(B, N, 2), z(B, N, 2), z(B, N), z(B, N), z(B, N)
    E = max(1, max(len(s["bk"]) for s in scenes))
    bidx = torch.zeros(B, E, 2, device=device, dtype=torch.long)
    bk, brest, bd, bm = z(B, E), z(B, E), z(B, E), z(B, E)
    for b, s in enumerate(scenes):
        n, e = len(s["pos"]), len(s["bk"])
        pos[b, :n] = torch.as_tensor(s["pos"], device=device, dtype=dtype)
        vel[b, :n] = torch.as_tensor(s["vel"], device=device, dtype=dtype)
        mass[b, :n] = torch.as_tensor(s["mass"], device=device, dtype=dtype)
        kmat[b, :n] = torch.as_tensor(s["kmat"], device=device, dtype=dtype)
        mask[b, :n] = 1.0
        if e:
            bidx[b, :e] = torch.as_tensor(s["bonds"], device=device)
            bk[b, :e] = torch.as_tensor(s["bk"], device=device, dtype=dtype)
            brest[b, :e] = torch.as_tensor(s["brest"], device=device, dtype=dtype)
            bd[b, :e] = torch.as_tensor(s["bd"], device=device, dtype=dtype)
            bm[b, :e] = 1.0
    s0 = scenes[0]
    law = dict(s0["law"])
    kw = dict(kmat=kmat)
    if "grav" in law:
        kw["grav"] = law["grav"]
    if "lj" in law:
        kw["lj"] = law["lj"]
    if "contact" in law:
        kw["contact"] = law["contact"]
    if law.get("bonded"):
        kw["bonds"] = dict(idx=bidx, k=bk, rest=brest, damp=bd, bmask=bm)
    if "g_vec" in law:
        kw["g_vec"] = np.stack([s["law"].get("g_vec", np.zeros(2)) for s in scenes])[:, None, :]
    bk0 = s0["bound"]["kind"]
    if bk0 == "wall":
        lo = torch.as_tensor(np.stack([s["bound"]["lo"] for s in scenes]), device=device, dtype=dtype)
        hi = torch.as_tensor(np.stack([s["bound"]["hi"] for s in scenes]), device=device, dtype=dtype)
        kw["walls"] = dict(lo=lo, hi=hi, **law["walls"])
    if bk0 == "periodic":
        kw["period"] = torch.as_tensor(np.stack([s["bound"]["period"] for s in scenes]), device=device, dtype=dtype)
    occ = torch.as_tensor(np.stack([occupancy(s["bound"]) for s in scenes]), device=device, dtype=torch.float32)
    return dict(pos=pos, vel=vel, mass=mass, mask=mask, phys=tt.Physics(**kw), occ=occ, dt=s0["dt"], sub=s0["sub"],
                kind=s0["kind"], tags=[s.get("tags", {}) for s in scenes], bound=[s["bound"] for s in scenes])


def equilibrate(batch, steps=200, h=0.005):
    """Berendsen pre-equilibration for LJ scenes (per-scene target temperature from tags['equil'])."""
    t = batch["tags"][0].get("equil")
    if not t or steps <= 0:
        return batch
    p, v = tt.thermostat_equil(batch["phys"], batch["pos"], batch["vel"], batch["mass"], batch["mask"], steps, h, t[0])
    batch["pos"], batch["vel"] = p, v
    return batch


def circular_velocities(pos, mass, dev, eps=tt.EPS, sign=None):
    p = torch.as_tensor(pos, device=dev, dtype=torch.float64)[None]
    m = torch.as_tensor(mass, device=dev, dtype=torch.float64)[None]
    a = tt.grav_acc_large(p, m, torch.ones_like(m), eps)[0].cpu().numpy()
    com = (mass[:, None] * pos).sum(0) / mass.sum()
    r = pos - com
    rr = np.linalg.norm(r, axis=1).clip(1e-6)
    ar = -(a * r).sum(1) / rr
    vc = np.sqrt(np.clip(ar, 0, None) * rr)
    t = np.stack([-r[:, 1], r[:, 0]], 1) / rr[:, None]
    return t * vc[:, None]


def big_scene(name, n, span, rng, dev):
    """Large-scene ICs for tier>=1 data and the gravity-systems eval group: plummer, cold_collapse, binary, disc, bh, uniform, clumpy."""
    if name == "plummer":
        a = span / 12
        u = rng.uniform(0, 0.99, n)
        r = a / np.sqrt(u ** (-2 / 3) - 1)
        th = rng.uniform(0, 2 * np.pi, n)
        pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        mass = np.full(n, 0.05)
        vel = circular_velocities(pos, mass, dev) * rng.uniform(0.6, 1.0, (n, 1)) * rng.choice([-1, 1], (n, 1)) \
            + rng.normal(0, 0.2, (n, 2)) * math.sqrt(mass.sum() / (4 * a))
    elif name == "cold_collapse":
        sg = span / 10
        pos = rng.normal(0, sg, (n, 2))
        mass = np.full(n, 0.05)
        vel = rng.normal(0, 0.05, (n, 2))
    elif name == "binary":
        m1, m2 = rng.uniform(5, 20, 2)
        sep = span / 10
        mu = m1 + m2
        vrel = math.sqrt(mu * sep ** 2 / (sep ** 2 + tt.EPS ** 2) ** 1.5) * 0.95
        pos = np.array([[-sep * m2 / mu, 0.0], [sep * m1 / mu, 0.0]])
        vel = np.array([[0.0, -vrel * m2 / mu], [0.0, vrel * m1 / mu]])
        mass = np.array([m1, m2])
        if n > 2:
            r = rng.uniform(span / 8, span / 3, n - 2)
            th = rng.uniform(0, 2 * np.pi, n - 2)
            pl = np.stack([r * np.cos(th), r * np.sin(th)], 1)
            ml = np.full(n - 2, 0.02)
            vc = np.sqrt(mu * r ** 2 / (r ** 2 + tt.EPS ** 2) ** 1.5)
            pos = np.concatenate([pos, pl])
            vel = np.concatenate([vel, vc[:, None] * np.stack([-np.sin(th), np.cos(th)], 1)])
            mass = np.concatenate([mass, ml])
    elif name in ("disc", "bh"):
        M = 30.0 if name == "bh" else 15.0
        r = rng.exponential(span / 10, n - 1).clip(1.0, span / 2.2)
        th = rng.uniform(0, 2 * np.pi, n - 1)
        pos = np.concatenate([[[0.0, 0.0]], np.stack([r * np.cos(th), r * np.sin(th)], 1)])
        mass = np.concatenate([[M], np.full(n - 1, 0.05)])
        vel = circular_velocities(pos, mass, dev) * rng.uniform(0.9, 1.0, (n, 1))
        vel[0] = 0.0
    elif name == "uniform":
        pos = rng.uniform(-span / 2, span / 2, (n, 2)) * 0.9
        mass = np.exp(rng.uniform(np.log(0.01), np.log(0.5), n))
        vel = rng.normal(0, 0.3, (n, 2))
    elif name == "clumpy":
        k = 8
        cen = rng.uniform(-span / 3, span / 3, (k, 2))
        lab = rng.integers(0, k, n)
        pos = cen[lab] + rng.normal(0, span / 25, (n, 2))
        mass = np.exp(rng.uniform(np.log(0.01), np.log(0.5), n))
        vel = rng.normal(0, 0.3, (n, 2))
    else:
        raise ValueError(name)
    vel = vel - (mass[:, None] * vel).sum(0) / mass.sum()
    return pos.astype(np.float64), vel.astype(np.float64), mass.astype(np.float64)
