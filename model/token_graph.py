import torch


def build_radius_graph(positions, neighbor_radius):
    """Undirected edges (as directed pairs, both directions) between
    tokens within `neighbor_radius`, in absolute position space. Has no
    notion of tiles -- passing positions from two adjacent tiles together
    produces edges across the tile boundary exactly as if the tokens had
    come from one untiled scene. Building the graph from one tile's
    tokens in isolation loses any cross-boundary edge, so callers must
    always pass the full combined position set (design spec's tiling
    requirement)."""
    n = positions.shape[0]
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    within = dist <= neighbor_radius
    within.fill_diagonal_(False)
    src, dst = torch.nonzero(within, as_tuple=True)
    return torch.stack([src, dst], dim=0)


def build_radius_graph_cells(positions, neighbor_radius):
    """Same edge set as `build_radius_graph` in O(N log N + N*k) time and
    O(N*k) memory (k = neighbours per token), with no N x N tensor: tokens
    are sorted by cell (side = neighbor_radius) and each token only looks at
    the occupants of its 3x3 cell block, found by searchsorted on the sorted
    cell keys, so nothing scales with the grid size either."""
    n = positions.shape[0]
    device = positions.device
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    cell = torch.floor(positions.detach() / neighbor_radius).long()
    cell = cell - cell.min(dim=0).values + 1
    stride = int(cell[:, 1].max()) + 2
    key = cell[:, 0] * stride + cell[:, 1]
    sorted_key, order = torch.sort(key)
    offsets = torch.tensor([dx * stride + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=device)
    probe = key.unsqueeze(1) + offsets.unsqueeze(0)
    lo = torch.searchsorted(sorted_key, probe, right=False)
    hi = torch.searchsorted(sorted_key, probe, right=True)
    counts = (hi - lo).reshape(-1)
    total = int(counts.sum())
    src = torch.arange(n, device=device).repeat_interleave(9).repeat_interleave(counts)
    starts = lo.reshape(-1).repeat_interleave(counts)
    within_block = torch.arange(total, device=device) - (counts.cumsum(0) - counts).repeat_interleave(counts)
    dst = order[starts + within_block]
    diff = positions[dst] - positions[src]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    keep = (dist <= neighbor_radius) & (src != dst)
    return torch.stack([src[keep], dst[keep]], dim=0)


def build_radius_graph_periodic(positions, box, rc):
    """Same as `build_radius_graph_cells` but for one periodic rectangular box (min-image distance,
    cell index wrapped mod the per-axis cell count). Cell side = box/floor(box/rc) >= rc, so a pair
    within rc is always in the same or an axis-wrapped-adjacent cell; no N x N tensor at any point.
    Falls back to periodic all-pairs when the box holds fewer than 3 cells on an axis."""
    n = positions.shape[0]
    device = positions.device
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    ncell = torch.clamp((box / rc).floor().long(), min=1)
    if int(ncell.min()) < 3:
        diff = positions[None, :, :] - positions[:, None, :]
        diff = diff - box * torch.round(diff / box)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
        within = dist <= rc
        within.fill_diagonal_(False)
        src, dst = torch.nonzero(within, as_tuple=True)
        return torch.stack([src, dst], dim=0)
    cellsize = box / ncell.to(positions.dtype)
    cell = torch.floor(positions / cellsize).long() % ncell
    key = cell[:, 0] * ncell[1] + cell[:, 1]
    sorted_key, order = torch.sort(key)
    idx = torch.arange(n, device=device)
    src_parts, dst_parts = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nkey = ((cell[:, 0] + dx) % ncell[0]) * ncell[1] + (cell[:, 1] + dy) % ncell[1]
            lo = torch.searchsorted(sorted_key, nkey, right=False)
            hi = torch.searchsorted(sorted_key, nkey, right=True)
            counts = hi - lo
            total = int(counts.sum())
            if total == 0:
                continue
            starts = lo.repeat_interleave(counts)
            within_block = torch.arange(total, device=device) - (counts.cumsum(0) - counts).repeat_interleave(counts)
            src_parts.append(idx.repeat_interleave(counts))
            dst_parts.append(order[starts + within_block])
    src, dst = torch.cat(src_parts), torch.cat(dst_parts)
    diff = positions[dst] - positions[src]
    diff = diff - box * torch.round(diff / box)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
    keep = (dist <= rc) & (src != dst)
    return torch.stack([src[keep], dst[keep]], dim=0)
