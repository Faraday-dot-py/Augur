"""Randomised obstacle scenes for option C (see docs/debugging/scatter-contact-c.md): 0-4 solid obstacles of random shape/size/position inside a random
container, plus held-out families that each leave one training range (size, position, count, shape class, container class).
Coordinates are CENTERED. Ball radius .75, 2-15 balls, speed <= 2.3 per axis, as scatter-contact.md. Obstacle size s = circumradius.

  c_train   count 0-4, s in [.8,3.2], obstacle centres y <= 3.5, convex shapes, containers box / disc / open (floor + no walls)
  c_id      same law as c_train (fresh seeds)
  c_big     s in [3.8,5.2]            c_small  s in [.3,.6]          c_pos    obstacle centres y in [4.5,10]
  c_count   6-8 obstacles             c_nonconvex  L / star / crescent  c_cont  triangle / hexagon / rotated-rectangle container
"""
import math
from collections import deque

import numpy as np
import torch

from scripts import scatter_contact as sc

R0, V0 = 0.75, 2.3
WIN, CELL = 14.0, 0.5
NONCONVEX = ("L", "star", "crescent")
CONVEX = ("circle", "rect", "reg", "irr")
BASE = dict(count=(0, 4), size=(0.8, 3.2), ymax=3.5, ymin=None, cont=("box", "disc", "open"), cbox=(4.5, 8.0), cdisc=(5.0, 8.0), shapes=CONVEX, need_nc=False, coff=(-4.0, 4.0))
SPECS = {
    "c_train": BASE,
    "c_id": BASE,
    "c_big": dict(BASE, count=(1, 2), size=(3.8, 5.2), cont=("box", "disc"), cbox=(7.0, 9.0), cdisc=(7.5, 9.5)),
    "c_small": dict(BASE, count=(2, 4), size=(0.3, 0.6)),
    "c_pos": dict(BASE, count=(1, 3), ymax=None, ymin=4.5, cont=("box", "disc"), cbox=(6.0, 8.0), cdisc=(6.5, 8.0), coff=(-4.0, 4.0), cy=(1.0, 4.0)),
    "c_count": dict(BASE, count=(6, 8), size=(0.8, 1.8), cont=("box", "disc"), cbox=(7.0, 9.0), cdisc=(7.5, 9.5)),
    "c_nonconvex": dict(BASE, count=(1, 3), shapes=CONVEX + NONCONVEX, need_nc=True, cont=("box", "disc")),
    "c_cont": dict(BASE, cont=("tri", "hex", "rrect"), cbox=(4.5, 8.0)),
}
FAMS_NEW = tuple(SPECS)[1:]
HOLD_AXES = FAMS_NEW[1:]


def hull(p):
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def _rot(pts, rot, c):
    cs, sn = math.cos(rot), math.sin(rot)
    return np.stack([c[0] + cs * pts[:, 0] - sn * pts[:, 1], c[1] + sn * pts[:, 0] + cs * pts[:, 1]], 1)


def make_obstacle(rng, kind, c, s):
    rot = rng.uniform(0, 2 * math.pi)
    if kind == "circle":
        pr = sc.circle(c, s)
    elif kind == "rect":
        u = rng.uniform(0.4, 0.95)
        pr = sc.rect_poly(c, 2 * s * u, 2 * s * math.sqrt(1 - u * u), rot)
    elif kind == "reg":
        pr = sc.regular_poly(c, s, int(rng.integers(3, 9)), rot)
    elif kind == "irr":
        a = rng.uniform(0, 2 * math.pi, int(rng.integers(5, 10)))
        pts = hull(np.stack([np.cos(a), np.sin(a)], 1) * rng.uniform(0.5, 1.0, len(a))[:, None])
        pts = pts / np.linalg.norm(pts, axis=1).max() * s
        pr = sc.poly(_rot(pts, rot, c))
    elif kind == "L":
        t = rng.uniform(0.5, 0.9) * s
        pr = sc.poly(_rot(np.array([(-s, -s), (s, -s), (s, -s + t), (-s + t, -s + t), (-s + t, s), (-s, s)]) / math.sqrt(2), rot, c))
    elif kind == "star":
        n = int(rng.integers(4, 7))
        a = np.pi * np.arange(2 * n) / n
        r = np.where(np.arange(2 * n) % 2 == 0, s, rng.uniform(0.35, 0.55) * s)
        pr = sc.poly(_rot(np.stack([r * np.cos(a), r * np.sin(a)], 1), rot, c))
    else:
        t = np.linspace(0, np.pi, 17)
        k = rng.uniform(0.25, 0.5)
        pts = np.concatenate([np.stack([s * np.cos(t), s * np.sin(t)], 1), np.stack([s * np.cos(t[-2:0:-1]), k * s * np.sin(t[-2:0:-1])], 1)])
        pr = sc.poly(_rot(pts, rot, c))
    pr["size"], pr["shape"], pr["convex"], pr["ctr"] = float(s), kind, kind not in NONCONVEX, (float(c[0]), float(c[1]))
    return pr


def make_container(rng, name, sp):
    c = rng.uniform(*sp["coff"], 2)
    if "cy" in sp:
        c[1] = rng.uniform(*sp["cy"])
    if name == "box":
        h = rng.uniform(*sp["cbox"], 2)
        return sc.hbox(c - h, c + h), c - h, c + h
    if name == "disc":
        r = rng.uniform(*sp["cdisc"])
        return sc.hdisc(c, r), c - r, c + r
    if name == "rrect":
        h = rng.uniform(*sp["cbox"], 2)
        return sc.hpoly(sc.rect_poly(c, 2 * h[0], 2 * h[1], rng.uniform(0, math.pi))["pts"]), c - h.max(), c + h.max()
    r = rng.uniform(6.5, 9.0) if name == "tri" else rng.uniform(5.5, 8.0)
    return sc.hpoly(sc.regular_poly(c, r, 3 if name == "tri" else 6, rng.uniform(0, 2 * math.pi))["pts"]), c - r, c + r


def grid_pts():
    xs = np.arange(-WIN + CELL / 2, WIN, CELL)
    return xs, np.stack(np.meshgrid(xs, xs, indexing="xy"), -1)


def free_cells(prims, margin):
    xs, p = grid_pts()
    sd = sc.scene_sdf(prims, torch.tensor(p, dtype=torch.float64)).numpy()
    return p, sd > R0 + margin


def largest_component(m):
    lab = np.zeros(m.shape, dtype=np.int32)
    best, nlab = (0, 0), 0
    for i, j in zip(*np.nonzero(m)):
        if lab[i, j]:
            continue
        nlab += 1
        lab[i, j] = nlab
        q, cnt = deque([(i, j)]), 0
        while q:
            a, b = q.popleft()
            cnt += 1
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = a + da, b + db
                if 0 <= x < m.shape[0] and 0 <= y < m.shape[1] and m[x, y] and not lab[x, y]:
                    lab[x, y] = nlab
                    q.append((x, y))
        if cnt > best[0]:
            best = (cnt, nlab)
    return lab == best[1]


def place_balls(rng, prims, cells, n):
    P = []
    for _ in range(n):
        for _ in range(300):
            p = cells[rng.integers(len(cells))] + rng.uniform(-CELL / 2, CELL / 2, 2)
            if float(sc.scene_sdf(prims, torch.tensor(p, dtype=torch.float64))) < R0 + 0.1:
                continue
            if all(np.linalg.norm(p - q) > 2 * R0 + 0.1 for q in P):
                P.append(p)
                break
    return np.array(P)


def gen_scene(rng, fam):
    sp = SPECS[fam]
    for _ in range(200):
        cont = sp["cont"][int(rng.integers(len(sp["cont"])))]
        nobs = int(rng.integers(sp["count"][0], sp["count"][1] + 1))
        if cont == "open":
            fx, fy, th, wd = rng.uniform(2, 6), rng.uniform(-3, 3), rng.uniform(1.5, 3.0), rng.uniform(14, 22)
            rot = rng.uniform(-0.25, 0.25)
            floor = sc.rect_poly((fx, fy), th, wd, rot)
            floor["size"], floor["shape"], floor["convex"] = float(math.hypot(th, wd) / 2), "floor", True
            xmax = fx - th / 2 - 0.5 * wd * abs(math.sin(rot)) - 0.8
            lo, hi = np.array([max(xmax - 12, -11.0), fy - 8.0]), np.array([xmax, fy + 8.0])
            prims, base = [floor], [floor]
        else:
            c0, lo, hi = make_container(rng, cont, sp)
            prims, base = [c0], [c0]
        kinds = [sp["shapes"][int(rng.integers(len(sp["shapes"])))] for _ in range(nobs)]
        if sp["need_nc"] and not any(k in NONCONVEX for k in kinds):
            kinds = [NONCONVEX[int(rng.integers(3))]] + kinds[1:] if kinds else [NONCONVEX[int(rng.integers(3))]]
        ok = True
        for k in kinds:
            s = rng.uniform(*sp["size"])
            for _ in range(60):
                c = rng.uniform(lo, hi)
                if sp["ymax"] is not None and c[1] > sp["ymax"]:
                    continue
                if sp["ymin"] is not None and not (sp["ymin"] <= c[1] <= 10.0):
                    continue
                if float(sc.scene_sdf(base, torch.tensor(c, dtype=torch.float64))) > 0.2:
                    prims.append(make_obstacle(rng, k, c, s))
                    break
            else:
                ok = False
        if not ok:
            continue
        p, adm = free_cells(prims, 0.05)
        comp = largest_component(adm)
        if cont == "open":
            sel = comp & (p[..., 0] >= lo[0]) & (p[..., 0] <= hi[0]) & (p[..., 1] >= lo[1]) & (p[..., 1] <= hi[1])
            need = 60
        else:
            sel = comp
            need = max(40, 0.35 * free_cells(base, 0.05)[1].sum())
        if comp.sum() < need or sel.sum() < 40:
            continue
        pos = place_balls(rng, prims, p[sel], int(rng.integers(2, 16)))
        if len(pos) < 2:
            continue
        return dict(pos=pos, vel=rng.uniform(-V0, V0, (len(pos), 2)), mass=np.ones(len(pos)), rad=np.full(len(pos), R0), prims=prims, fam=fam, cont=cont)
    raise RuntimeError(f"gen_scene {fam}: no valid scene in 200 tries")
