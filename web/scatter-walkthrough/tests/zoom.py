import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter-walkthrough/index.html"
fails = []

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}).new_page()
    pg.goto(base)
    pg.wait_for_function("window.__walk && window.__walk.S.trace", timeout=60000)
    pg.wait_for_function("!window.__walk.walk.playing", timeout=30000)
    dist = lambda: pg.evaluate("window.__walk.stage.camera.position.distanceTo(window.__walk.stage.controls.target)")
    d0 = dist()
    pg.mouse.move(700, 450)
    for _ in range(6):
        pg.mouse.wheel(0, -300)
        pg.wait_for_timeout(60)
    d1 = dist()
    for x in range(300, 1100, 40):
        for y in (300, 450, 600):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(30)
    pg.wait_for_timeout(300)
    d2 = dist()
    print("dist", d0, d1, d2)
    if not d1 < d0 - 1e-3:
        fails.append("wheel did not zoom")
    if abs(d2 - d1) > 1e-6:
        fails.append("hover reset the view")
    b.close()
print("FAIL " + "; ".join(fails) if fails else "ok zoom survives hover")
sys.exit(1 if fails else 0)
