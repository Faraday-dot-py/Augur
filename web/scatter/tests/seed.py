import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter/index.html"
out = sys.argv[1] if len(sys.argv) > 1 else "."
errors = []

def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)

SET_N = "(n) => { const r = document.getElementById('r-bodies'); r.value = n; r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); }"
FILE = """async (n) => {
  const idx = await (await fetch('precomputed/index.json')).json();
  const buf = await (await fetch('precomputed/' + idx.runs[n])).arrayBuffer();
  return Array.from(new Float32Array(buf, 32, 4 * n * 12));
}"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}).new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(base)
    pg.wait_for_function("window.__sim && window.__sim().tick >= 0", timeout=90000)
    check(pg.get_attribute("#b-mode", "aria-pressed") == "true" and pg.inner_text("#b-mode") == "Fixed seed", "defaults to the fixed seed")

    for n in range(10, 101, 10):
        pg.evaluate(f"({SET_N})({n})")
        pg.wait_for_function(f"window.__sim().tick >= 0 && window.__model.stage.nBodies === {n} && window.__sim().epoch > 0", timeout=60000)
        pg.wait_for_timeout(300)
        f0 = pg.evaluate("Array.from(window.__frames.get(0).pos)")
        ref = pg.evaluate(FILE, n)
        err = max(abs(a - c) for a, c in zip(f0, ref[:2 * n]))
        check(len(f0) == 2 * n and err < 1e-5, f"N={n}: tick 0 equals the seed-4738 run (max diff {err:.1e})")
        if n in (10, 100):
            pg.screenshot(path=f"{out}/seed_n{n}.png")

    pg.evaluate(f"({SET_N})(10)")
    pg.wait_for_function("window.__model.stage.nBodies === 10", timeout=30000)
    pg.keyboard.press("Space")
    pg.wait_for_function("window.__sim().ahead >= 6", timeout=60000)
    ref = pg.evaluate(FILE, 10)
    worst = 0
    for t in range(0, 7):
        fp = pg.evaluate(f"Array.from(window.__frames.get({t}).pos)")
        fv = pg.evaluate(f"Array.from(window.__frames.get({t}).vel)")
        o = t * 40
        worst = max(worst, max(abs(a - c) for a, c in zip(fp, ref[o:o + 20])), max(abs(a - c) for a, c in zip(fv, ref[o + 20:o + 40])))
    check(worst < 1e-5, f"browser chain reproduces the stored run for ticks 0-6 (max diff {worst:.1e})")

    pg.click("#b-reset")
    pg.wait_for_function("window.__sim().tick === 0 && window.__sim().frames >= 1", timeout=30000)
    pg.keyboard.press("Space")
    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Random'")
    check(pg.get_attribute("#r-bodies", "step") == "1", "random mode allows any body count")
    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Fixed seed'")
    pg.set_viewport_size({"width": 390, "height": 800})
    pg.wait_for_timeout(400)
    check(pg.evaluate("document.documentElement.scrollWidth <= innerWidth"), "390px: no horizontal scroll")
    check(not logs, f"no console errors {logs[:3]}")
    b.close()

print("FAIL: " + "; ".join(errors) if errors else "ALL OK")
sys.exit(1 if errors else 0)
