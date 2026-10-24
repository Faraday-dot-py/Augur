import torch

from scripts import poly_cell_potential as pc

G, H_, E = 8, 1.0, 8.0


def _coef(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 6, G, G, generator=g, dtype=torch.float64)


def test_force_matches_finite_difference_of_potential():
    c = _coef()
    pos = torch.tensor([[[0.3, -1.1], [2.2, 1.7]]], dtype=torch.float64)
    f = pc.force(c, pos, H_, E)
    eps = 1e-6
    for d in (0, 1):
        e = torch.zeros(2, dtype=torch.float64)
        e[d] = eps
        fd = -(pc.potential(c, pos + e, H_, E) - pc.potential(c, pos - e, H_, E)) / (2 * eps)
        assert torch.allclose(f[..., d], fd, atol=1e-6)


def test_gradient_form_has_no_loop_work_and_naive_blend_does():
    c = _coef(1)
    ctr = torch.tensor([0.4, -0.3], dtype=torch.float64)
    w_exact = pc.loop_work(lambda p: pc.force(c, p, H_, E), ctr, 1.3, H_, E)
    w_naive = pc.loop_work(lambda p: pc.naive_force(c, p, H_, E), ctr, 1.3, H_, E)
    assert abs(w_exact) < 5e-3
    assert abs(w_naive) > 5 * abs(w_exact), (w_exact, w_naive)


def test_single_cell_polynomial_is_exact_inside_cell_when_neighbours_agree():
    c = torch.zeros(1, 6, G, G, dtype=torch.float64)
    pos = torch.tensor([[[0.2, 0.1]]], dtype=torch.float64)
    c[0, 0] = 1.0
    f = pc.force(c, pos, H_, E)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-12)
