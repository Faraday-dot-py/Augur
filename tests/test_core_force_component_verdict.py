from scripts import core_force_component_verdict as cv


def doc(e, u, k, p, x, n=500):
    probes = [{"t": t, "cls": {"core15": {"n": n, "bias": e}}, "parts": {"core15": {"unet_mean": u, "kernel_mean": k, "pair_mean": p, "exact_mean": x}}} for t in cv.TICKS]
    return {"probes": probes}


STOCK = doc([0.0, 10.0], [0.0, 11.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0])


def run(e, u, k, p):
    return cv.verdict(STOCK, doc(e, u, k, p, [0.0, 1.0]))["verdict"]


def test_each_class_reachable():
    assert run([0, 2.0], [0, 2.0], [0, 0.1], [0, 0.0]) == "UNET-CARRIES"
    assert run([0, 2.0], [0, 0.1], [0, 2.0], [0, 0.0]) == "KERNEL-CARRIES"
    assert run([0, 2.0], [0, 0.1], [0, 0.0], [0, 2.0]) == "PAIR-CARRIES"
    assert run([0, 2.0], [0, 1.2], [0, 1.2], [0, 0.0]) == "SHARED"
    assert run([0, 2.0], [0, 0.2], [0, 0.2], [0, 0.2]) == "CANCELLATION"
    assert run([0, 0.5], [0, 0.5], [0, 0.0], [0, 0.0]) == "NO-NET-FORCE"


def test_stock_not_reproduced_and_mixed():
    weak = doc([0, 2.0], [0, 2.0], [0, 0.0], [0, 0.0], [0, 1.0])
    assert cv.verdict(weak, weak)["verdict"].startswith("INCONCLUSIVE-stock")
    mixed = doc([0, 2.0], [0, 2.0], [0, 0.0], [0, 0.0], [0, 1.0])
    mixed["probes"][1]["parts"]["core15"].update(unet_mean=[0, 0.1], kernel_mean=[0, 2.0])
    mixed["probes"][2]["parts"]["core15"].update(unet_mean=[0, 0.1], pair_mean=[0, 2.0])
    assert cv.verdict(STOCK, mixed)["verdict"] == "INCONCLUSIVE-mixed-over-ticks"


def test_small_core_skipped():
    assert cv.verdict(STOCK, doc([0, 2.0], [0, 2.0], [0, 0.0], [0, 0.0], [0, 1.0], n=10))["verdict"].startswith("INCONCLUSIVE")
