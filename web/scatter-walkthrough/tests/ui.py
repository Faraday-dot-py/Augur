import os
import sys
from playwright.sync_api import sync_playwright

out = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8765/scatter-walkthrough/index.html"
shots = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(9))
w, h = (int(x) for x in sys.argv[4].split("x")) if len(sys.argv) > 4 else (1400, 850)
errors = []

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(base)
    pg.wait_for_function("window.__walk && window.__walk.S.trace", timeout=60000)
    for i in range(9):
        frac = float(os.environ.get("FRAC", "1"))
        pg.evaluate(f"(() => {{ const w = window.__walk.walk; w.go({i}, 0); w.time = w.length * {frac}; w.playing = false; }})()")
        pg.wait_for_timeout(700)
        n = pg.evaluate("document.querySelectorAll('#text p').length")
        if i in shots:
            pg.screenshot(path=f"{out}/walk_{w}x{h}_p{i}_f{int(frac * 100)}.png")
        print("phase", i, "paragraphs", n, "title", pg.inner_text("#ph-t"))
    b.close()
print("console errors:", errors)
sys.exit(1 if errors else 0)
