import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Serve web/ on port 8765 first. Usage: python3 ui_precomputed.py <outdir> [ticks_to_compute=20]
out = sys.argv[1] if len(sys.argv) > 1 else "."
MORE = sys.argv[2] if len(sys.argv) > 2 else "20"
base = (sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8765") + "/scatter-tiled/index.html"
ref = json.loads((Path(__file__).parent / "ref_continue.json").read_text())
errors = []


def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)


APP = "window.__app()"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1500, "height": 880}).new_page()
    logs, reqs = [], []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.on("request", lambda r: reqs.append(r.url.split("/scatter-tiled/")[-1]))
    t0 = time.time()
    pg.goto(base)
    pg.wait_for_function("window.__app && window.__app().tick >= 0", timeout=120000)
    load_s = time.time() - t0
    print(f"first shipped tick on screen {load_s:.2f} s after navigation")
    pg.wait_for_function("window.__app().tick >= 30", timeout=60000)
    pg.screenshot(path=out + "/pre_tick30.png")
    pg.wait_for_function("window.__app().tick >= 100", timeout=120000)
    a = pg.evaluate(APP)
    check(a["fromPre"] and not a["producerStarted"] and not a["simStarted"] and a["produced"] == 0, f"ticks 0-100 played with no model worker, no sim worker, 0 frames produced ({a['tick']} shown)")
    heavy = [r for r in reqs if any(s in r for s in ("weights.bin", "weights.json", "tiled_worker", "tile_worker", "sim_worker", "state100"))]
    check(not heavy, f"no weights, worker or tick-100 state requests while playing 0-100 {heavy}")
    sizes = pg.evaluate("performance.getEntriesByType('resource').filter(r => r.name.includes('precomputed')).map(r => [r.name.split('/').pop(), r.transferSize, r.decodedBodySize])")
    print("   shipped files:", sizes)
    shown = pg.evaluate("window.__app().shownRate")
    check(pg.evaluate("document.getElementById('more').hidden") is False, "compute-more panel shown at tick 100")
    pg.wait_for_timeout(500)
    pg.screenshot(path=out + "/pre_tick100.png")
    f100 = pg.evaluate("Array.from(window.__frames.get(100).pos)")
    d = max(abs(x - y) for x, y in zip(f100, ref["pos"][0]))
    check(d < 1e-5, f"displayed tick 100 equals the reference run (max diff {d:.1e})")
    check(not any("rror" in l for l in logs), f"no console errors {logs[:3]}")

    # compute more
    pg.select_option("#sel-more", MORE)
    t1 = time.time()
    pg.click("#b-more")
    pg.wait_for_function("window.__app().producerStarted", timeout=30000)
    pg.wait_for_timeout(3000)
    pg.screenshot(path=out + "/more_computing.png")
    target = 100 + int(MORE)
    pg.wait_for_function(f"window.__app().newest >= {target}", timeout=900000)
    secs = time.time() - t1
    a = pg.evaluate(APP)
    print(f"computed {MORE} ticks in {secs:.1f} s including worker start ({int(MORE) / secs:.2f} ticks/s end to end), workers {a['workers']}")
    check(a["produced"] == int(MORE) and not a["computing"], f"{a['produced']} frames produced by the model, computing finished")
    worst = 0.0
    for k in range(101, min(target, ref["from"] + len(ref["ticks"]) - 1) + 1):
        pos = pg.evaluate(f"Array.from(window.__frames.get({k}).pos)")
        r = ref["pos"][k - ref["from"]]
        worst = max(worst, max(abs(x - y) for x, y in zip(pos, r)))
        if k == 101:
            print(f"   tick 100 -> 101 max diff vs reference {worst:.2e}")
    check(worst <= 1e-3, f"browser-computed ticks 101-{min(target, 120)} vs the continuous reference: max diff {worst:.2e} (tolerance 1e-3)")
    pg.wait_for_function(f"window.__app().tick >= {min(target, 112)}", timeout=60000)
    pg.screenshot(path=out + "/more_tick112.png")
    check(pg.evaluate("window.__app().inspect >= 0") and pg.evaluate("window.__frames.get(window.__app().newest).unet !== null"), "computed ticks carry the U-Net activations")
    check(pg.evaluate("document.getElementById('p-more').hidden"), "progress hidden after the batch")

    # stop
    pg.select_option("#sel-more", "100")
    pg.click("#b-more")
    pg.wait_for_function(f"window.__app().newest >= {target + 3}", timeout=300000)
    pg.click("#b-more-stop")
    pg.wait_for_timeout(300)
    n1 = pg.evaluate("window.__app().newest")
    pg.wait_for_timeout(8000)
    n2 = pg.evaluate("window.__app().newest")
    a = pg.evaluate(APP)
    check(not a["computing"] and n2 - n1 <= 1 and n2 < target + 100, f"Stop halts the computation (newest {n1} -> {n2} over 8 s)")
    pg.click("#b-more")
    pg.wait_for_function(f"window.__app().newest >= {n2 + 3}", timeout=300000)
    check(True, f"computation resumes after Stop (newest {pg.evaluate('window.__app().newest')})")
    pg.click("#b-more-stop")

    # back to the shipped run
    pg.keyboard.press("r")
    pg.wait_for_function("window.__app().tick >= 1 && window.__app().fromPre", timeout=30000)
    check(pg.evaluate("window.__app().newest") == 100, "Reset returns to the shipped 0-100 ticks")
    # leaving the default scene starts live computation from tick 0
    pg.select_option("#sel-scene", "warm")
    pg.wait_for_function("window.__app().tick >= 1 && !window.__app().fromPre", timeout=300000)
    check(pg.evaluate("document.getElementById('more').hidden") and pg.evaluate("window.__app().produced") > 0, "another scene is computed live from tick 0")
    print("console errors:", logs)
    check(not any("rror" in l for l in logs), "no console errors")
    pg.close()

    # phone layout
    ctx = b.new_context(viewport={"width": 390, "height": 800}, has_touch=True, is_mobile=True)
    m = ctx.new_page()
    m.goto(base)
    m.wait_for_function("window.__app && window.__app().tick >= 100", timeout=180000)
    m.wait_for_timeout(800)
    m.screenshot(path=out + "/mobile_tick100.png")
    ov = m.evaluate("document.documentElement.scrollWidth > innerWidth")
    check(not ov, "no horizontal page overflow at phone width")
    b.close()

print("FAILED" if errors else "ALL OK", errors)
sys.exit(1 if errors else 0)
