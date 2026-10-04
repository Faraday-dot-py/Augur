import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter/index.html"
out = sys.argv[1] if len(sys.argv) > 1 else "."
errors = []

def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)

SAMPLE = "(() => { const s = window.__model.stage; return { tick: +document.getElementById('o-tick').textContent, x: s.bodyPos[0], phi: s.blocks.get('phi').planes[0].data.slice(0, 4096).reduce((a, v) => a + v, 0), stat: document.getElementById('mstat').textContent }; })()"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}).new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(base)
    pg.wait_for_function("window.__model && document.getElementById('mstat').textContent.includes('slabs')", timeout=90000)
    check(pg.locator("#stage").count() == 0 and pg.locator("#b-model").count() == 0 and pg.locator("#b-field").count() == 0, "2D canvas and Model/Field/Fit controls are gone")
    check(pg.locator("#stage3d").is_visible() and pg.locator("#r-ch").is_visible(), "3D view is the page, channel slider shown")
    check(pg.locator("#mview a[href*=walkthrough]").is_visible(), "walkthrough link present")
    check("260,419" in pg.inner_text("#mcap"), "caption shows parameter count from weights")
    check(pg.locator(".lab:visible").count() > 5, "block name labels visible")

    pg.evaluate("(() => { const r = document.getElementById('r-bodies'); r.value = 100; r.dispatchEvent(new Event('change')); })()")
    pg.wait_for_function("window.__model.stage.nBodies === 100", timeout=20000)
    check(pg.evaluate("window.__model.stage.bodies.instanceMatrix.array.length >= 100 * 16 && window.__model.stage.bodyHl.length >= 100"), "body instance buffers hold 100 bodies")
    pg.wait_for_timeout(300)
    if pg.inner_text("#b-pause").startswith("Resume"):
        pg.keyboard.press("Space")
    xs, phis, ticks = [], [], []
    for _ in range(19):
        s = pg.evaluate(SAMPLE)
        xs.append(round(s["x"], 4)); phis.append(round(s["phi"], 2)); ticks.append(s["tick"])
        pg.wait_for_timeout(500)
    check(len(set(xs)) >= 8, f"bodies move during playback ({len(set(xs))} distinct positions over ticks {ticks[0]}..{ticks[-1]})")
    check(len(set(phis)) >= 3, f"slabs refresh during playback ({len(set(phis))} distinct potential checksums)")
    pg.wait_for_function("document.getElementById('mstat').textContent.includes('slabs: tick 100')", timeout=60000)
    check(True, "slabs catch up to the last tick after playback ends")
    pg.screenshot(path=out + "/model_desktop.png")

    d0 = pg.evaluate("window.__model.stage.camera.position.distanceTo(window.__model.stage.controls.target)")
    pg.mouse.move(600, 450)
    for _ in range(5):
        pg.mouse.wheel(0, -300)
        pg.wait_for_timeout(60)
    d1 = pg.evaluate("window.__model.stage.camera.position.distanceTo(window.__model.stage.controls.target)")
    check(d1 < d0 - 1e-3, f"wheel zooms ({d0:.1f} -> {d1:.1f})")
    pg.keyboard.press("r")
    pg.wait_for_timeout(300)
    for x in range(200, 1000, 40):
        for y in (260, 330, 400, 480):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(25)
    pg.wait_for_timeout(300)
    d2 = pg.evaluate("window.__model.stage.camera.position.distanceTo(window.__model.stage.controls.target)")
    check(abs(d2 - d1) < 1e-6, "hover and playback do not reset zoom")

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

    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Live'")
    if pg.inner_text("#b-pause").startswith("Resume"):
        pg.keyboard.press("Space")
    pg.keyboard.press("d")
    xs, phis, ticks = [], [], []
    for _ in range(24):
        s = pg.evaluate(SAMPLE)
        xs.append(round(s["x"], 4)); phis.append(round(s["phi"], 2))
        ticks.append(int(pg.evaluate("document.getElementById('s-tick').textContent")))
        pg.wait_for_timeout(1000)
    check(len(set(xs)) >= 4, f"live: bodies move ({len(set(xs))} distinct positions, ticks {ticks[0]}..{ticks[-1]})")
    check(len(set(phis)) >= 3, f"live: slabs refresh ({len(set(phis))} distinct potential checksums)")
    pg.keyboard.press("d")
    pg.screenshot(path=out + "/model_live.png")
    check(not logs, f"no console errors {logs[:3]}")

    ph2 = b.new_context(viewport={"width": 390, "height": 800}, has_touch=True).new_page()
    logs2 = []
    ph2.on("console", lambda m: logs2.append(m.text) if m.type == "error" else None)
    ph2.on("pageerror", lambda e: logs2.append(str(e)))
    ph2.goto(base)
    ph2.wait_for_function("window.__model && document.getElementById('mstat').textContent.includes('slabs')", timeout=90000)
    ph2.wait_for_timeout(800)
    check(ph2.evaluate("document.documentElement.scrollWidth <= innerWidth"), "390px: no horizontal scroll")
    ph2.screenshot(path=out + "/model_phone.png")
    check(not logs2, f"390px: no console errors {logs2[:3]}")
    b.close()

print("FAIL: " + "; ".join(errors) if errors else "ALL OK")
sys.exit(1 if errors else 0)
