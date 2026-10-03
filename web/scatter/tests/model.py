import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter/index.html"
out = sys.argv[1] if len(sys.argv) > 1 else "."
errors = []

def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}).new_page()
    logs, reqs = [], []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.on("request", lambda r: reqs.append(r.url))
    pg.goto(base)
    pg.wait_for_function("document.getElementById('s-ckpt').textContent !== '-'", timeout=60000)
    pg.wait_for_timeout(500)
    check(not any("three" in u for u in reqs), "three.js not loaded before the Model view is opened")
    check(pg.locator("#stage3d").is_hidden() and pg.locator("#b-field").is_visible(), "2D view shown by default")

    pg.keyboard.press("m")
    pg.wait_for_function("document.getElementById('mstat').textContent.startsWith('live')", timeout=60000)
    check(any("three" in u for u in reqs), "three.js loaded on first open")
    check(pg.locator("#stage3d").is_visible() and pg.locator("#stage").is_hidden(), "3D view replaces the 2D canvas")
    check(pg.locator("#b-field").is_hidden() and pg.locator("#r-ch").is_visible(), "2D-only controls hidden, channel slider shown")
    check(pg.get_attribute("#b-model", "aria-pressed") == "true", "Model button pressed")
    check(pg.locator("#mview a[href*=walkthrough]").is_visible(), "walkthrough link present")
    check("260,419" in pg.inner_text("#mcap"), "caption shows parameter count from weights")
    check(pg.locator(".lab:visible").count() > 5, "block name labels visible")

    ph = lambda: pg.evaluate("(() => { const s = window.__model.stage; return [s.camera.position.distanceTo(s.controls.target), s.blocks.get('phi').planes[0].data.slice(0, 4096).reduce((a, v) => a + v, 0), window.__model.tick()]; })()")
    d0, sum0, tick0 = ph()
    pg.mouse.move(600, 450)
    for _ in range(5):
        pg.mouse.wheel(0, -300)
        pg.wait_for_timeout(60)
    d1 = ph()[0]
    check(d1 < d0 - 1e-3, f"wheel zooms ({d0:.1f} -> {d1:.1f})")
    for x in range(200, 1000, 40):
        for y in (260, 330, 400, 480):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(25)
    pg.wait_for_timeout(300)
    check(abs(ph()[0] - d1) < 1e-6, "hover does not reset zoom")

    shown = False
    for x in range(100, 1300, 20):
        for y in range(230, 600, 20):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(15)
            if pg.locator("#tip").is_visible():
                shown = True
                break
        if shown:
            break
    check(shown, "hover tooltip appears")
    check("value" in pg.inner_text("#tip") or "body" in pg.inner_text("#tip"), "tooltip has content: " + pg.inner_text("#tip").replace("\n", " | ")[:70])

    pg.mouse.move(5, 5)
    pg.keyboard.press("Space")
    pg.wait_for_timeout(800)
    t1 = pg.evaluate("window.__model.tick()")
    pg.keyboard.press(".")
    pg.wait_for_function(f"window.__model.tick() > {t1}", timeout=60000)
    pg.wait_for_timeout(500)
    t2 = pg.evaluate("window.__model.tick()")
    check(t2 > t1, f"slabs update after a sim step (traced tick {t1} -> {t2})")

    pg.keyboard.press("m")
    pg.wait_for_timeout(300)
    check(pg.locator("#stage").is_visible() and pg.locator("#stage3d").is_hidden(), "2D view back after toggling")
    k0 = pg.evaluate("document.getElementById('s-tick').textContent")
    pg.keyboard.press(".")
    pg.wait_for_timeout(1500)
    pg.keyboard.press("d")
    pg.wait_for_timeout(400)
    k1 = pg.evaluate("document.getElementById('s-tick').textContent")
    check(k1 != k0, f"2D sim still steps after toggling back ({k0} -> {k1})")
    pg.keyboard.press("m")
    pg.wait_for_timeout(500)
    check(pg.locator("#stage3d").is_visible(), "reopens")
    pg.screenshot(path=out + "/model_desktop.png")
    check(not logs, f"no console errors {logs[:3]}")

    ph2 = b.new_context(viewport={"width": 390, "height": 800}, has_touch=True).new_page()
    logs2 = []
    ph2.on("console", lambda m: logs2.append(m.text) if m.type == "error" else None)
    ph2.on("pageerror", lambda e: logs2.append(str(e)))
    ph2.goto(base)
    ph2.wait_for_function("document.getElementById('s-ckpt').textContent !== '-'", timeout=60000)
    ph2.click("#b-model")
    ph2.wait_for_function("document.getElementById('mstat').textContent.startsWith('live')", timeout=60000)
    ph2.wait_for_timeout(800)
    check(ph2.evaluate("document.documentElement.scrollWidth <= innerWidth"), "390px: no horizontal scroll")
    ph2.screenshot(path=out + "/model_phone.png")
    check(not logs2, f"390px: no console errors {logs2[:3]}")
    b.close()

print("FAIL: " + "; ".join(errors) if errors else "ALL OK")
sys.exit(1 if errors else 0)
