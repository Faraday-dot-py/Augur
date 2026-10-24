import torch

M, PX, PY, SX, SY, QXX, QXY, QYY, WXX, WXY, WYY, XXX, XXY, XYX, XYY, OCC = range(16)
TXXX, TXXY, TXYY, TYYY = 16, 17, 18, 19
NMOM = 20


def deposit_moments(pos, vel, mass, mask, n, h, origin, periodic=False, occ=None):
    """NGP deposit of raw mass-weighted moments about each cell centre: m, sum m v, sum m d, sum m d d^T,
    sum m v v^T, sum m d v^T (d = offset from cell centre), plus the occupancy fraction channel."""
    B, N = mask.shape
    s = (pos - origin) / h
    if periodic:
        s = s % n
    i = torch.floor(s)
    inside = ((i >= 0) & (i < n)).all(-1)
    d = (s - i - 0.5) * h
    i = i.long().clamp(0, n - 1)
    w = mass * mask * inside.to(pos.dtype)
    vx, vy, dx, dy = vel[..., 0], vel[..., 1], d[..., 0], d[..., 1]
    val = torch.stack([torch.ones_like(vx), vx, vy, dx, dy, dx * dx, dx * dy, dy * dy, vx * vx, vx * vy, vy * vy,
                       dx * vx, dx * vy, dy * vx, dy * vy, dx ** 3, dx * dx * dy, dx * dy * dy, dy ** 3], -1) * w[..., None]
    bidx = torch.arange(B, device=pos.device)[:, None].expand(B, N)
    flat = bidx * n * n + i[..., 1] * n + i[..., 0]
    out = torch.zeros(B * n * n, NMOM - 1, device=pos.device, dtype=pos.dtype).index_add(0, flat.reshape(-1), val.reshape(-1, NMOM - 1))
    out = torch.cat([out[:, :OCC], torch.zeros_like(out[:, :1]), out[:, OCC:]], 1).view(B, n, n, NMOM).permute(0, 3, 1, 2).contiguous()
    if occ is not None:
        out[:, OCC] = occ[:, 0]
    return out


def _shift(c, dx, dy):
    m, px, py, sx, sy = c[:, M], c[:, PX], c[:, PY], c[:, SX], c[:, SY]
    return torch.stack([
        m, px, py, sx + m * dx, sy + m * dy,
        c[:, QXX] + 2 * dx * sx + m * dx * dx,
        c[:, QXY] + dx * sy + dy * sx + m * dx * dy,
        c[:, QYY] + 2 * dy * sy + m * dy * dy,
        c[:, WXX], c[:, WXY], c[:, WYY],
        c[:, XXX] + dx * px, c[:, XXY] + dx * py, c[:, XYX] + dy * px, c[:, XYY] + dy * py,
        c[:, OCC],
        c[:, TXXX] + 3 * dx * c[:, QXX] + 3 * dx * dx * sx + dx ** 3 * m,
        c[:, TXXY] + 2 * dx * c[:, QXY] + dy * c[:, QXX] + dx * dx * sy + 2 * dx * dy * sx + dx * dx * dy * m,
        c[:, TXYY] + 2 * dy * c[:, QXY] + dx * c[:, QYY] + dy * dy * sx + 2 * dx * dy * sy + dx * dy * dy * m,
        c[:, TYYY] + 3 * dy * c[:, QYY] + 3 * dy * dy * sy + dy ** 3 * m,
    ], 1)


def merge2x2(mom, hc):
    """Exact parent moments about the parent centre from 2x2 children of size hc (parallel-axis translation)."""
    B, C, H, W = mom.shape
    v = mom.view(B, C, H // 2, 2, W // 2, 2)
    out = 0
    for by in (0, 1):
        for bx in (0, 1):
            out = out + _shift(v[:, :, :, by, :, bx], (bx - 0.5) * hc, (by - 0.5) * hc)
    occ = v[:, OCC].mean((2, 4))
    return torch.cat([out[:, :OCC], occ[:, None], out[:, OCC + 1:]], 1)


def nfeat(disp=True, cross=False, order3=False):
    return 8 + 3 * disp + 4 * cross + 4 * order3 + 1


def moment_features(mom, h, disp=True, cross=False, eps=1e-9, order3=False):
    m = mom[:, M:M + 1]
    ok = (m > eps).to(mom.dtype)
    ms = m.clamp_min(eps)
    v = mom[:, PX:PY + 1] / ms
    c = mom[:, SX:SY + 1] / ms
    q = mom[:, QXX:QYY + 1] / ms - torch.cat([c[:, :1] ** 2, c[:, :1] * c[:, 1:], c[:, 1:] ** 2], 1)
    ch = [m / h ** 2, v * ok, c / h * ok, q / h ** 2 * ok]
    if disp:
        w = mom[:, WXX:WYY + 1] / ms - torch.cat([v[:, :1] ** 2, v[:, :1] * v[:, 1:], v[:, 1:] ** 2], 1)
        ch.append(w * ok)
    if cross:
        x = mom[:, XXX:XYY + 1] / ms
        cv = torch.stack([c[:, 0] * v[:, 0], c[:, 0] * v[:, 1], c[:, 1] * v[:, 0], c[:, 1] * v[:, 1]], 1)
        ch.append((x - cv) / h * ok)
    if order3:
        e = mom[:, TXXX:TYYY + 1] / ms
        qq = mom[:, QXX:QYY + 1] / ms
        cx, cy = c[:, 0:1], c[:, 1:2]
        exx, exy, eyy = qq[:, 0:1], qq[:, 1:2], qq[:, 2:3]
        t = torch.cat([e[:, 0:1] - 3 * cx * exx + 2 * cx ** 3,
                       e[:, 1:2] - 2 * cx * exy - cy * exx + 2 * cx ** 2 * cy,
                       e[:, 2:3] - 2 * cy * exy - cx * eyy + 2 * cx * cy ** 2,
                       e[:, 3:4] - 3 * cy * eyy + 2 * cy ** 3], 1)
        ch.append(t / h ** 3 * ok)
    ch.append(mom[:, OCC:OCC + 1])
    return torch.cat(ch, 1)
