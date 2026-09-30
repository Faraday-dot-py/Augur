import numpy as np

CENTER = 500.0


def accel(pos, eps, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d ** 2).sum(-1) + eps ** 2
    inv = r2 ** -1.5
    np.fill_diagonal(inv, 0.0)
    return g * (d * inv[..., None]).sum(1)


def energy(pos, vel, eps, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt((d ** 2).sum(-1) + eps ** 2)
    iu = np.triu_indices(len(pos), 1)
    return 0.5 * (vel ** 2).sum() - g * (1.0 / r[iu]).sum()


def init_bodies(n_bodies, rng, spread=5.0, speed=0.5, scale=False, dim=2):
    if scale:
        f = n_bodies / 8
        spread, speed = spread * f ** 0.5, speed * f ** 0.25
    pos = CENTER + rng.uniform(-spread, spread, (n_bodies, dim))
    vel = rng.normal(0.0, speed, (n_bodies, dim))
    vel -= vel.mean(0)
    return pos, vel


def rollout(pos, vel, steps, dt=0.1, substeps=4, eps=0.5, g=1.0):
    h = dt / substeps
    ps, vs = [pos.copy()], [vel.copy()]
    pos, vel = pos.copy(), vel.copy()
    a = accel(pos, eps, g)
    for _ in range(steps):
        for _ in range(substeps):
            vel += 0.5 * h * a
            pos += h * vel
            a = accel(pos, eps, g)
            vel += 0.5 * h * a
        ps.append(pos.copy())
        vs.append(vel.copy())
    return np.stack(ps), np.stack(vs)


def rollout_torch(pos, vel, steps, device, dt=0.1, substeps=4, eps=0.5, g=1.0, relativistic=False, c=1.0):
    import torch

    h = dt / substeps
    pos, vel = torch.tensor(pos, device=device), torch.tensor(vel, device=device)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv.fill_diagonal_(0.0)
        return g * (d * inv[..., None]).sum(1)

    if relativistic:
        # momentum state p, dp/dt = F, coordinate velocity v = p / sqrt(1 + |p|^2/c^2),
        # so |v| < c by construction. Matches scripts/orbit_bh.py --relativistic.
        vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()

        def speed(p):
            return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()

        ps, vs = [pos.cpu().numpy()], [speed(mom).cpu().numpy()]
        a = acc(pos)
        for _ in range(steps):
            for _ in range(substeps):
                mom = mom + 0.5 * h * a
                pos = pos + h * speed(mom)
                a = acc(pos)
                mom = mom + 0.5 * h * a
            ps.append(pos.cpu().numpy())
            vs.append(speed(mom).cpu().numpy())
        return np.stack(ps), np.stack(vs)

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


def energy_rel(pos, vel, eps, c, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt((d ** 2).sum(-1) + eps ** 2)
    iu = np.triu_indices(len(pos), 1)
    vmag2 = np.clip((vel ** 2).sum(-1), 0.0, 0.9801 * c ** 2)
    gamma = 1.0 / np.sqrt(1 - vmag2 / c ** 2)
    return (c ** 2 * (gamma - 1)).sum() - g * (1.0 / r[iu]).sum()


def make_dataset(num, ball_range, steps, seed, scale=False, device=None, dim=2, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        n = int(rng.integers(ball_range[0], ball_range[1] + 1))
        p, v = init_bodies(n, rng, scale=scale, dim=dim)
        out.append(rollout(p, v, steps, **kw) if device is None else rollout_torch(p, v, steps, device, **kw))
    return out
