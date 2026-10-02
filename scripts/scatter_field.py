import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CENTER = 500.0


class LocalNet(nn.Module):
    def __init__(self, cin, cout, width=32, layers=8):
        super().__init__()
        self.inp = nn.Conv2d(cin, width, 3, padding=1)
        self.mid = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(layers - 1))
        self.out = nn.Conv2d(width, cout, 3, padding=1)

    def forward(self, x):
        h = F.gelu(self.inp(x))
        for c in self.mid:
            h = h + F.gelu(c(h))
        return self.out(h)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    def __init__(self, cin, cout, width=32, levels=4):
        super().__init__()
        self.levels = levels
        self.inp = nn.Conv2d(cin, width, 3, padding=1)
        self.enc = nn.ModuleList(_block(width, width) for _ in range(levels + 1))
        self.dec = nn.ModuleList(_block(2 * width, width) for _ in range(levels))
        self.out = nn.Conv2d(width, cout, 3, padding=1)

    def forward(self, x):
        h = self.enc[0](F.gelu(self.inp(x)))
        skips = [h]
        for l in range(1, self.levels + 1):
            h = self.enc[l](F.avg_pool2d(h, 2))
            skips.append(h)
        for l in reversed(range(self.levels)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.dec[l](torch.cat([h, skips[l]], 1))
        return self.out(h)


class ScatterField(nn.Module):
    """Tokens (pos, vel, mass) -> CIC scatter of [m, m*vx, m*vy] onto a grid
    (x,y consumed by the scatter) -> grid net -> output diff field
    [ddx, ddy, hidden...] -> bilinear gather at each token -> vel += dt*gathered,
    pos += dt*(vel_old + vel_new)/2. Mass is conserved in the data, so dm is a
    hard 0 (no channel). With recurrent=True the previous output field is a
    network input (the medium state). Positions are relative to CENTER; the
    grid covers [-extent/2, extent/2)^2 and tokens outside it get zero force."""

    def __init__(self, grid=64, extent=64.0, net="local", recurrent=True, hidden_ch=0, dt=0.1, width=32, layers=8,
                 levels=5, momfix=True, in_scale=1.0, potential=False, kernel=False, pp=0.0, split=False, nonet=False):
        super().__init__()
        self.potential, self.kernel, self.pp, self.split, self.nonet = potential, kernel, pp, split, nonet
        if pp:
            self.ppmlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.ppmlp[-1].weight)
            nn.init.zeros_(self.ppmlp[-1].bias)
        self.grid, self.extent, self.h = grid, extent, extent / grid
        self.recurrent, self.dt, self.momfix, self.in_scale = recurrent, dt, momfix, in_scale
        self.cf = (1 if potential else 2) + hidden_ch
        cin = 3 + (self.cf if recurrent else 0) + (2 if kernel else 0)
        if kernel:
            self.kmlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.kmlp[-1].weight)
            nn.init.zeros_(self.kmlp[-1].bias)
        if net == "local":
            self.net = LocalNet(cin, self.cf, width, layers)
        else:
            self.net = UNet(cin, self.cf, width, min(levels, int(math.log2(grid)) - 2))
        nn.init.zeros_(self.net.out.weight)
        nn.init.zeros_(self.net.out.bias)

    def _corners(self, pos):
        u = (pos + self.extent / 2) / self.h - 0.5
        i0 = torch.floor(u)
        return i0.long(), u - i0

    def _corner_iter(self, pos):
        G = self.grid
        i0, w = self._corners(pos)
        B, N = pos.shape[:2]
        bidx = torch.arange(B, device=pos.device)[:, None].expand(B, N)
        for ox in (0, 1):
            for oy in (0, 1):
                ix, iy = i0[..., 0] + ox, i0[..., 1] + oy
                wt = (w[..., 0] if ox else 1 - w[..., 0]) * (w[..., 1] if oy else 1 - w[..., 1])
                ok = ((ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)).to(pos.dtype)
                flat = bidx * G * G + iy.clamp(0, G - 1) * G + ix.clamp(0, G - 1)
                yield flat, wt * ok

    def scatter(self, pos, vel, mass, mask):
        B, N = pos.shape[:2]
        G = self.grid
        mm = mass * mask
        val = torch.stack([mm, mm * vel[..., 0], mm * vel[..., 1]], -1) * (self.in_scale / self.h ** 2)
        out = torch.zeros(B * G * G, 3, device=pos.device, dtype=pos.dtype)
        for flat, w in self._corner_iter(pos):
            out = out.index_add(0, flat.reshape(-1), (val * w[..., None]).reshape(-1, 3))
        return out.view(B, G, G, 3).permute(0, 3, 1, 2)

    def gather(self, field, pos):
        B, C, G, _ = field.shape
        fl = field.permute(0, 2, 3, 1).reshape(B * G * G, C)
        out = 0
        for flat, w in self._corner_iter(pos):
            out = out + fl[flat] * w[..., None]
        return out

    def kernel_acc(self, rho):
        G = self.grid
        idx = torch.arange(2 * G, device=rho.device, dtype=rho.dtype)
        d = torch.minimum(idx, 2 * G - idx) * self.h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        K = self.kmlp(torch.stack([r, torch.log(r + self.h)], -1))[..., 0]
        if self.split:
            K = K * (1 - (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2)
        Kf = torch.fft.rfft2(K)
        rf = torch.fft.rfft2(rho, s=(2 * G, 2 * G))
        phi = torch.fft.irfft2(rf * Kf, s=(2 * G, 2 * G))[..., :G, :G] * self.h ** 2
        return self.neg_grad(phi)

    def pp_acc(self, pos, mass, mask):
        d = pos[:, None, :, :] - pos[:, :, None, :]
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        g = self.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
        win = (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
        n = pos.shape[1]
        w = g * win * (mass * mask)[:, None, :] * mask[:, :, None] * (1 - torch.eye(n, device=pos.device, dtype=pos.dtype))
        return (w[..., None] * d / r[..., None]).sum(2)

    def neg_grad(self, phi):
        p = F.pad(phi, (1, 1, 1, 1))
        gx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / (2 * self.h)
        gy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / (2 * self.h)
        return -torch.cat([gx, gy], 1)

    def init_field(self, B, device, dtype=torch.float32):
        return torch.zeros(B, self.cf, self.grid, self.grid, device=device, dtype=dtype)

    def step(self, pos, vel, mass, mask, field):
        x = self.scatter(pos, vel, mass, mask)
        if self.kernel:
            ak = self.kernel_acc(x[:, :1])
            x = torch.cat([x, ak], 1)
        if self.recurrent:
            x = torch.cat([x, field], 1)
        if self.nonet:
            field = x.new_zeros(x.shape[0], self.cf, self.grid, self.grid)
        else:
            field = self.net(x)
        acc = self.neg_grad(field[:, :1]) if self.potential else field[:, :2]
        if self.kernel:
            acc = acc + ak
        a_tok = self.gather(acc, pos)
        if self.pp:
            a_tok = a_tok + self.pp_acc(pos, mass, mask)
        dv = a_tok * self.dt
        if self.momfix:
            mm = (mass * mask)[..., None]
            dv = dv - (mm * dv).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
        dv = dv * mask[..., None]
        return pos + self.dt * (vel + 0.5 * dv), vel + dv, field, dv


def receptive_field(model):
    """Per-step receptive radius in cells of the grid net (impulse via autograd
    on a copy with randomised output layer), and in sim units."""
    import copy

    net = copy.deepcopy(model.net).float().cpu()
    nn.init.normal_(net.out.weight, std=0.1)
    G = model.grid
    cin = net.inp.in_channels
    x = (0.1 * torch.randn(1, cin, G, G)).requires_grad_(True)
    y = net(x)
    y[0, 0, G // 2, G // 2].backward()
    g = x.grad.abs().amax(1)[0]
    ok = (g > 1e-9 * g.max()).nonzero()
    r = int((ok - G // 2).abs().max())
    return r, r * model.h, bool(r >= G // 2 - 1)


@torch.no_grad()
def rollout(model, pos, vel, mass, mask, steps):
    field = model.init_field(pos.shape[0], pos.device, pos.dtype)
    ps, vs, dms = [pos], [vel], []
    m0 = mass
    for _ in range(steps):
        pos, vel, mass, field, _ = _step_m(model, pos, vel, mass, mask, field)
        ps.append(pos)
        vs.append(vel)
        dms.append(((mass - m0).abs() * mask).sum(1) / mask.sum(1))
    rollout.last_dm = torch.stack(dms, 1).mean(0)
    return torch.stack(ps, 1), torch.stack(vs, 1)


def energy(pos, vel, mass, mask, eps):
    mm = mass * mask
    d = pos[..., None, :, :] - pos[..., :, None, :]
    r = torch.sqrt((d ** 2).sum(-1) + eps ** 2)
    n = pos.shape[-2]
    pair = mm[..., :, None] * mm[..., None, :] / r * (1 - torch.eye(n, device=pos.device, dtype=pos.dtype))
    return 0.5 * (mm * (vel ** 2).sum(-1)).sum(-1) - 0.5 * pair.sum((-1, -2))


def momentum(vel, mass, mask):
    return ((mass * mask)[..., None] * vel).sum(-2)


def ang_momentum(pos, vel, mass, mask):
    mm = mass * mask
    com = (mm[..., None] * pos).sum(-2) / mm.sum(-1, keepdim=True)
    r = pos - com[..., None, :]
    return (mm * (r[..., 0] * vel[..., 1] - r[..., 1] * vel[..., 0])).sum(-1)


def to_tensors(data, device):
    """list of (P, V) or (P, V, m) numpy trajectories (T,N,2) in absolute coords -> padded relative tensors."""
    S = len(data)
    T = data[0][0].shape[0]
    N = max(d[0].shape[1] for d in data)
    P = torch.zeros(S, T, N, 2, dtype=torch.float64)
    V = torch.zeros(S, T, N, 2, dtype=torch.float64)
    M = torch.zeros(S, N, dtype=torch.float64)
    mask = torch.zeros(S, N, dtype=torch.float64)
    for s, d in enumerate(data):
        n = d[0].shape[1]
        P[s, :, :n] = torch.tensor(d[0] - CENTER)
        V[s, :, :n] = torch.tensor(d[1])
        M[s, :n] = torch.tensor(d[2]) if len(d) > 2 else 1.0
        mask[s, :n] = 1.0
    return P.to(device), V.to(device), M.to(device), mask.to(device)


def rollout_mass_torch(pos, vel, m, steps, device, dt=0.1, substeps=4, eps=0.5, g=1.0):
    h = dt / substeps
    pos, vel, m = torch.tensor(pos, device=device), torch.tensor(vel, device=device), torch.tensor(m, device=device)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv.fill_diagonal_(0.0)
        return g * (d * inv[..., None] * m[None, :, None]).sum(1)

    ps, vs = [pos.cpu().numpy()], [vel.cpu().numpy()]
    a = acc(pos)
    for _ in range(steps):
        for _ in range(substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = acc(pos)
            vel = vel + 0.5 * h * a
        ps.append(pos.cpu().numpy())
        vs.append(vel.cpu().numpy())
    return np.stack(ps), np.stack(vs)


def make_star_dataset(num, steps, seed, device, dt=0.1, eps=0.5, n_light=(3, 11)):
    """One heavy star near the centre, light bodies of mass 0.01-0.5 on rough circular orbits."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        nl = int(rng.integers(n_light[0], n_light[1] + 1))
        M = rng.uniform(8.0, 30.0)
        r = rng.uniform(3.0, 28.0, nl)
        th = rng.uniform(0, 2 * np.pi, nl)
        sgn = rng.choice([-1.0, 1.0], nl)
        vc = np.sqrt(M * r ** 2 / (r ** 2 + eps ** 2) ** 1.5) * rng.uniform(0.7, 1.1, nl)
        off = rng.uniform(-4, 4, 2)
        pos = np.concatenate([[off], off + np.stack([r * np.cos(th), r * np.sin(th)], 1)]) + CENTER
        vel = np.concatenate([[[0.0, 0.0]], (sgn * vc)[:, None] * np.stack([-np.sin(th), np.cos(th)], 1)])
        m = np.concatenate([[M], np.exp(rng.uniform(np.log(0.01), np.log(0.5), nl))])
        vel -= (m[:, None] * vel).sum(0) / m.sum()
        P, V = rollout_mass_torch(pos, vel, m, steps, device, dt=dt, eps=eps)
        out.append((P, V, m))
    return out


def traj_metrics(Pm, Vm, Pt, Vt, M, mask, eps):
    """Per-step means over scenes. Pm/Vm model, Pt/Vt truth, (S,T,N,2); M, mask (S,N)."""
    cnt = mask.sum(1)
    err = (((Pm - Pt) ** 2).sum(-1).sqrt() * mask[:, None]).sum(2) / cnt[:, None]
    T = Pm.shape[1]
    Mx, Kx = M[:, None].expand(-1, T, -1), mask[:, None].expand(-1, T, -1)
    out = {"err": err[:, 1:].mean(0).tolist()}
    for tag, P, V in (("model", Pm, Vm), ("true", Pt, Vt)):
        E = energy(P, V, Mx, Kx, eps)
        L = ang_momentum(P, V, Mx, Kx)
        mom = momentum(V, Mx, Kx).norm(dim=-1)
        out["energy_" + tag] = E.mean(0).tolist()
        out["energy_drift_rel_" + tag] = ((E - E[:, :1]).abs() / E[:, :1].abs().clamp_min(1e-9)).mean(0).tolist()
        out["angmom_drift_rel_" + tag] = ((L - L[:, :1]).abs() / L[:, :1].abs().clamp_min(1e-3)).mean(0).tolist()
        out["mom_drift_" + tag] = (mom - mom[:, :1]).abs().mean(0).tolist()
    return out


def const_vel_err(Pt, Vt, mask, dt):
    T = Pt.shape[1]
    t = torch.arange(1, T, device=Pt.device, dtype=Pt.dtype)[None, :, None, None]
    pred = Pt[:, :1] + Vt[:, :1] * dt * t
    cnt = mask.sum(1)
    e = (((pred - Pt[:, 1:]) ** 2).sum(-1).sqrt() * mask[:, None]).sum(2) / cnt[:, None]
    return e.mean(0).tolist()


def self_force_probe(model, device, n=512, seed=4738):
    """One isolated token at rest at random sub-cell offsets; true force is 0. Returns mean |dv|/dt with momfix off."""
    g = torch.Generator().manual_seed(seed)
    pos = ((torch.rand(n, 1, 2, generator=g) - 0.5) * 8.0).to(device)
    vel = torch.zeros_like(pos)
    mass = torch.ones(n, 1, device=device)
    mask = torch.ones(n, 1, device=device)
    old = model.momfix
    model.momfix = False
    with torch.no_grad():
        dv = model.step(pos, vel, mass, mask, model.init_field(n, device))[3]
    model.momfix = old
    return float((dv.norm(dim=-1) / model.dt).mean())


def delete_star_probe(model, device, dists=(3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 28.0), warm=12, after=12, mstar=20.0,
                      mtracer=0.01, eps=0.5):
    """Static star (mass mstar) at the centre with 4 fixed test-mass tracers per distance. After `warm` steps the
    star token is deleted (mass 0, masked out). Tracers/star are held in place so only the field is probed.
    Returns, per distance, inward acceleration trace (model) over warm+after steps for delete and control runs."""
    D = len(dists)
    res = {}
    for delete in (True, False):
        pos = torch.zeros(D, 5, 2, device=device)
        for i, d in enumerate(dists):
            for k in range(4):
                a = k * math.pi / 2
                pos[i, 1 + k] = torch.tensor([d * math.cos(a), d * math.sin(a)])
        vel = torch.zeros_like(pos)
        mass = torch.full((D, 5), mtracer, device=device)
        mass[:, 0] = mstar
        mask = torch.ones(D, 5, device=device)
        pos0 = pos.clone()
        field = model.init_field(D, device)
        inward = -pos0[:, 1:] / pos0[:, 1:].norm(dim=-1, keepdim=True)
        trace = []
        with torch.no_grad():
            for t in range(warm + after):
                if delete and t == warm:
                    mass[:, 0] = 0.0
                    mask[:, 0] = 0.0
                pos, vel, _, field, dv = _step_m(model, pos, vel, mass, mask, field)
                a_r = ((dv[:, 1:] / model.dt) * inward).sum(-1).mean(1)
                trace.append(a_r.cpu().tolist())
                pos, vel = pos0.clone(), torch.zeros_like(vel)
        res["delete" if delete else "control"] = np.array(trace).T.tolist()
    truth = [mstar * d / (d ** 2 + eps ** 2) ** 1.5 for d in dists]
    out = {"dists": list(dists), "warm": warm, "after": after, "a_truth_before": truth, "delete": res["delete"],
           "control": res["control"], "lag_steps": [], "ratio_before": []}
    for i in range(D):
        tr = np.array(res["delete"][i])
        before = tr[warm - 3:warm].mean()
        out["ratio_before"].append(float(before / truth[i]))
        post = tr[warm:]
        below = np.nonzero(post < 0.5 * before)[0]
        out["lag_steps"].append(int(below[0]) if len(below) else None)
    return out


TILE = 8


def _tile_conv(conv, x, nb):
    T, C = x.shape[:2]
    xp = torch.cat([x, x.new_zeros(1, C, TILE, TILE)])
    t = xp[torch.where(nb < 0, torch.full_like(nb, T), nb)].view(T, 3, 3, C, TILE, TILE)
    mosaic = t.permute(0, 3, 1, 4, 2, 5).reshape(T, C, 3 * TILE, 3 * TILE)
    return conv(mosaic[:, :, TILE - 1:2 * TILE + 1, TILE - 1:2 * TILE + 1])


class LevelNet(nn.Module):
    def __init__(self, cin, cout, width=32):
        super().__init__()
        self.c1 = nn.Conv2d(cin, width, 3)
        self.c2 = nn.Conv2d(width, width, 3)
        self.c3 = nn.Conv2d(width, cout, 3)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, x, nb):
        h = F.gelu(_tile_conv(self.c1, x, nb))
        h = h + F.gelu(_tile_conv(self.c2, h, nb))
        return _tile_conv(self.c3, h, nb)


class MultiLevelScatterField(nn.Module):
    """Block-sparse multilevel scatter field. Levels l=0..L-1 have cell size h_leaf*2^l; each level is stored as 8x8 tiles
    allocated only where a token whose leaf level <= l lives (top level covers the domain with one tile per scene).
    A token's leaf level is the coarsest level whose cell holds <= k_leaf tokens (level 0 = depth cap). Each token
    deposits moments [m, m*o (dipole), m*o o (quad, optional), m*v, n] (o = offset from cell centre in cell units) into
    its cell at every level >= leaf (restriction = direct sum). Top-down: F_l = prolong(F_{l+1}) + net_l([moments_l,
    prolonged parent, previous F_l]); F has channels [a0(2), Jh(4) = h*dA/dx, dm(1), hidden...]. Prolongation uses the
    Taylor expansion a0 + J.(delta). A token reads its leaf cell: a = a0 + Jh.o. The previous field is stored sparsely per
    level (tile map + data); tiles that did not exist last step are initialised by prolongation from the coarser level.
    mass += dm (heads zero-initialised, dm trained to 0 when mass is conserved)."""

    def __init__(self, extent=64.0, h_leaf=0.5, k_leaf=1, quad=True, recurrent=True, hidden_ch=0, dt=0.1, width=32,
                 momfix=True, in_scale=1.0, dm_apply=True):
        super().__init__()
        self.dm_apply = dm_apply
        self.extent, self.h0, self.k, self.quad = extent, h_leaf, k_leaf, quad
        self.G0 = int(round(extent / h_leaf))
        self.L = int(math.log2(self.G0)) - 2
        assert 2 ** (self.L + 2) == self.G0
        self.recurrent, self.dt, self.momfix, self.in_scale = recurrent, dt, momfix, in_scale
        self.cm = 1 + 2 + (3 if quad else 0) + 2 + 1
        self.cf = 7 + hidden_ch
        cin = self.cm + self.cf * (2 if recurrent else 1)
        self.nets = nn.ModuleList(LevelNet(cin, self.cf, width) for _ in range(self.L))
        self.extent_cells = [self.G0 >> l for l in range(self.L)]

    def init_field(self, B, device, dtype=torch.float32):
        return None

    def _structure(self, pos, mask):
        B, N = mask.shape
        dev = pos.device
        u0 = (pos + self.extent / 2) / self.h0
        inside = ((u0 >= 0) & (u0 < self.G0)).all(-1) & (mask > 0)
        u0 = u0.clamp(0, self.G0 - 1e-3)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
        ic, cnt = [], []
        for l in range(self.L):
            n = self.extent_cells[l]
            i = torch.floor(u0 / 2 ** l).long().clamp(0, n - 1)
            key = bidx * n * n + i[..., 0] * n + i[..., 1]
            c = torch.bincount(key[inside], minlength=B * n * n)
            ic.append(i)
            cnt.append(c[key])
        leaf = torch.zeros(B, N, dtype=torch.long, device=dev)
        for l in range(self.L):
            leaf = torch.where(cnt[l] <= self.k, torch.full_like(leaf, l), leaf)
        levels = []
        for l in range(self.L):
            nt = self.extent_cells[l] // TILE
            alloc = inside & (leaf <= l)
            tx, ty = ic[l][..., 0] // TILE, ic[l][..., 1] // TILE
            occ = torch.zeros(B, nt, nt, dtype=torch.bool, device=dev)
            occ[bidx[alloc], tx[alloc], ty[alloc]] = True
            ids = torch.cumsum(occ.reshape(-1).long(), 0) - 1
            tmap = torch.where(occ.reshape(-1), ids, torch.full_like(ids, -1)).view(B, nt, nt)
            tb, ttx, tty = occ.nonzero(as_tuple=True)
            nbs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = ttx + dx, tty + dy
                    ok = (nx >= 0) & (nx < nt) & (ny >= 0) & (ny < nt)
                    nbs.append(torch.where(ok, tmap[tb, nx.clamp(0, nt - 1), ny.clamp(0, nt - 1)], torch.full_like(nx, -1)))
            levels.append(dict(map=tmap, T=int(tb.numel()), tb=tb, tx=ttx, ty=tty, nb=torch.stack(nbs, 1),
                               tid=torch.where(alloc, tmap[bidx, tx, ty], torch.full_like(tx, -1)), alloc=alloc))
        return dict(u0=u0, ic=ic, leaf=leaf, inside=inside, levels=levels)

    def _moments(self, S, vel, mass):
        out = []
        for l, lv in enumerate(S["levels"]):
            T = lv["T"]
            if T == 0:
                out.append(mass.new_zeros(0, self.cm, TILE, TILE))
                continue
            u = S["u0"] / 2 ** l
            ic = S["ic"][l]
            o = u - (ic + 0.5)
            w = (lv["alloc"]).to(mass.dtype)
            m = mass * w * (self.in_scale / (self.h0 * 2 ** l))
            parts = [m, m * o[..., 0], m * o[..., 1]]
            if self.quad:
                parts += [m * o[..., 0] ** 2, m * o[..., 0] * o[..., 1], m * o[..., 1] ** 2]
            parts += [m * vel[..., 0], m * vel[..., 1], w]
            val = torch.stack(parts, -1)
            loc = (ic[..., 0] % TILE) * TILE + ic[..., 1] % TILE
            flat = (lv["tid"].clamp(min=0) * TILE * TILE + loc).reshape(-1)
            data = torch.zeros(T * TILE * TILE, self.cm, device=mass.device, dtype=mass.dtype)
            data = data.index_add(0, flat, val.reshape(-1, self.cm))
            out.append(data.view(T, TILE, TILE, self.cm).permute(0, 3, 1, 2))
        return out

    def _prolong(self, Fp, lp, lc):
        tmapp = lp["map"]
        pid = tmapp[lc["tb"], lc["tx"] // 2, lc["ty"] // 2]
        sx, sy = lc["tx"] % 2, lc["ty"] % 2
        T = lc["T"]
        C = Fp.shape[1]
        ar = torch.arange(TILE, device=Fp.device)
        rows = (sx[:, None] * 4 + ar[None] // 2)[:, None, :, None].expand(T, C, TILE, TILE)
        cols = (sy[:, None] * 4 + ar[None] // 2)[:, None, None, :].expand(T, C, TILE, TILE)
        P = Fp[pid]
        P = torch.gather(torch.gather(P, 2, rows), 3, cols)
        offx = (((ar % 2) * 2 - 1) * 0.25).to(Fp.dtype)[None, :, None]
        offy = (((ar % 2) * 2 - 1) * 0.25).to(Fp.dtype)[None, None, :]
        a0x = P[:, 0] + P[:, 2] * offx + P[:, 3] * offy
        a0y = P[:, 1] + P[:, 4] * offx + P[:, 5] * offy
        return torch.cat([a0x[:, None], a0y[:, None], 0.5 * P[:, 2:6], P[:, 6:]], 1)

    def step_m(self, pos, vel, mass, mask, state):
        S = self._structure(pos, mask)
        mom = self._moments(S, vel, mass)
        B = pos.shape[0]
        Fs = [None] * self.L
        for l in reversed(range(self.L)):
            lv = S["levels"][l]
            T = lv["T"]
            if T == 0:
                Fs[l] = mom[l].new_zeros(0, self.cf, TILE, TILE)
                continue
            if l == self.L - 1:
                par = mom[l].new_zeros(T, self.cf, TILE, TILE)
            else:
                par = self._prolong(Fs[l + 1], S["levels"][l + 1], lv)
            x = [mom[l], par]
            if self.recurrent:
                pv = par
                if state is not None:
                    omap, odata = state[l]
                    oid = omap[lv["tb"], lv["tx"], lv["ty"]]
                    odp = torch.cat([odata, odata.new_zeros(1, *odata.shape[1:])])
                    old = odp[torch.where(oid < 0, torch.full_like(oid, odata.shape[0]), oid)]
                    pv = torch.where((oid >= 0)[:, None, None, None], old, par)
                x.append(pv)
            Fs[l] = par + self.nets[l](torch.cat(x, 1), lv["nb"])
        sel_f = mom[0].new_zeros(pos.shape[0], pos.shape[1], self.cf)
        sel_o = mom[0].new_zeros(pos.shape[0], pos.shape[1], 2)
        for l, lv in enumerate(S["levels"]):
            if lv["T"] == 0:
                continue
            is_l = (S["leaf"] == l) & S["inside"]
            tid = lv["tid"].clamp(min=0)
            ic = S["ic"][l]
            vals = Fs[l][tid, :, ic[..., 0] % TILE, ic[..., 1] % TILE]
            o = S["u0"] / 2 ** l - (ic + 0.5)
            sel_f = sel_f + vals * is_l[..., None].to(vals.dtype)
            sel_o = sel_o + o * is_l[..., None].to(o.dtype)
        a = sel_f[..., 0:2] + torch.stack([sel_f[..., 2] * sel_o[..., 0] + sel_f[..., 3] * sel_o[..., 1],
                                          sel_f[..., 4] * sel_o[..., 0] + sel_f[..., 5] * sel_o[..., 1]], -1)
        dv = a * self.dt
        if self.momfix:
            mm = (mass * mask)[..., None]
            dv = dv - (mm * dv).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
        dv = dv * mask[..., None]
        dm = sel_f[..., 6] * mask
        new_state = [(lv["map"], Fs[l]) for l, lv in enumerate(S["levels"])]
        return pos + self.dt * (vel + 0.5 * dv), vel + dv, (mass + dm if self.dm_apply else mass), new_state, dv

    def step(self, pos, vel, mass, mask, state):
        p, v, _, st, dv = self.step_m(pos, vel, mass, mask, state)
        return p, v, st, dv


def _step_m(model, pos, vel, mass, mask, field):
    if hasattr(model, "step_m"):
        return model.step_m(pos, vel, mass, mask, field)
    p, v, f, dv = model.step(pos, vel, mass, mask, field)
    return p, v, mass, f, dv


def make_orbit_dataset(num, steps, seed, device, ratios=(0.25, 16.0), h_leaf=0.5, dt=0.1, eps=0.5, ecc=(0.6, 1.1), fixed_ratio=None):
    """Two unit-mass bodies on a bound orbit, separation d = ratio * h_leaf, ratio log-uniform in `ratios`
    (or fixed), speed = circular * U(ecc), random orientation/position."""
    from scripts import gravity_sim as gs

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        r = fixed_ratio if fixed_ratio is not None else float(np.exp(rng.uniform(np.log(ratios[0]), np.log(ratios[1]))))
        d = r * h_leaf
        th, ph = rng.uniform(0, 2 * np.pi, 2)
        a = d / (d ** 2 + eps ** 2) ** 1.5
        v = np.sqrt(a * d / 2) * (1.0 if fixed_ratio is not None else rng.uniform(*ecc))
        e = np.array([np.cos(th), np.sin(th)])
        t = np.array([-np.sin(th), np.cos(th)])
        c = CENTER + rng.uniform(-4, 4, 2)
        pos = np.stack([c + 0.5 * d * e, c - 0.5 * d * e])
        vel = np.stack([v * t, -v * t])
        P, V = gs.rollout_torch(pos, vel, steps, device, dt=dt, eps=eps)
        out.append((P, V))
    return out


def curl_probe(model, device, mstar=5.0, extent=8.0, n=21, hold=3):
    """Force field of a static star sampled with a light test token at each lattice point (one scene per point).
    Returns rms(curl)/rms(|grad a|) over r>2 plus the maps. Truth is exactly curl-free."""
    xs = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    S = len(pts)
    pos0 = torch.zeros(S, 2, 2, device=device)
    pos0[:, 1] = torch.tensor(pts, dtype=torch.float32, device=device)
    mass = torch.tensor([mstar, 1e-3], device=device)[None].expand(S, 2).contiguous()
    mask = torch.ones(S, 2, device=device)
    field = model.init_field(S, device)
    pos, vel = pos0.clone(), torch.zeros_like(pos0)
    with torch.no_grad():
        for _ in range(hold):
            pos, vel, _, field, dv = _step_m(model, pos, vel, mass, mask, field)
            pos, vel = pos0.clone(), torch.zeros_like(pos0)
    a = (dv[:, 1] / model.dt).cpu().numpy().reshape(n, n, 2)
    dx = xs[1] - xs[0]
    dax = np.gradient(a[..., 0], dx, axis=0), np.gradient(a[..., 0], dx, axis=1)
    day = np.gradient(a[..., 1], dx, axis=0), np.gradient(a[..., 1], dx, axis=1)
    curl = day[0] - dax[1]
    div = dax[0] + day[1]
    r = np.sqrt(gx ** 2 + gy ** 2)
    ok = r > 2.0
    tot = np.sqrt((curl[ok] ** 2).mean() + (div[ok] ** 2).mean())
    return {"curl_over_total": float(np.sqrt((curl[ok] ** 2).mean()) / tot), "rms_curl": float(np.sqrt((curl[ok] ** 2).mean())),
            "rms_div": float(np.sqrt((div[ok] ** 2).mean())), "a_map": a.tolist()}
