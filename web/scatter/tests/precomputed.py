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
    logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(base)
    st = lambda: pg.evaluate("document.getElementById('o-tick').textContent")
    pg.wait_for_function("+document.getElementById('o-tick').textContent > 0", timeout=20000)
    check(pg.get_attribute("#b-mode", "aria-pressed") == "true", "defaults to precomputed")

    for n in range(10, 101, 10):
        pg.evaluate(f"(() => {{ const r = document.getElementById('r-bodies'); r.value = {n}; r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); }})()")
        pg.wait_for_function(f"document.getElementById('o-bodies').textContent === '{n}'")
        pg.wait_for_function("+document.getElementById('o-tick').textContent > 2", timeout=20000)
        t1 = int(st())
        pg.wait_for_timeout(500)
        t2 = int(st())
        check(t2 > t1, f"N={n} loads and plays ({t1} -> {t2})")
        if n in (10, 100):
            pg.screenshot(path=f"{out}/pre_n{n}.png")

    pg.click("#b-pause")
    t = int(st())
    pg.wait_for_timeout(400)
    check(int(st()) == t, "pause holds the tick")
    pg.click("#b-step")
    check(int(st()) == t + 1, "step advances one tick")
    pg.evaluate("(() => { const r = document.getElementById('r-tick'); r.value = 5; r.dispatchEvent(new Event('input')); })()")
    check(int(st()) == 5, "scrubbing back to tick 5")
    pg.click("#b-reset")
    pg.wait_for_function("document.getElementById('o-tick').textContent === '0' || +document.getElementById('o-tick').textContent < 3")
    check(int(st()) < 3, "reset returns to tick 0")
    pg.evaluate("(() => { const r = document.getElementById('r-tick'); r.value = 100; r.dispatchEvent(new Event('input')); })()")
    pg.click("#b-pause")
    check(pg.inner_text("#b-pause").startswith("Pause"), "resume at the end restarts")
    pg.wait_for_timeout(600)
    check(int(st()) < 20, "restarted from tick 0")
    pg.evaluate("document.getElementById('b-pause').click()")

    pg.wait_for_function("document.getElementById('s-ckpt').textContent !== '-'", timeout=60000)
    pg.keyboard.press("d")
    pg.wait_for_timeout(300)
    check("precomputed" in pg.inner_text("#s-src"), "dev menu shows precomputed source")
    pg.keyboard.press("d")

    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Live'")
    check(pg.locator("#l-tick").is_hidden(), "tick slider hidden in live mode")
    pg.evaluate("document.getElementById('b-pause').textContent.startsWith('Resume') && document.getElementById('b-pause').click()")
    pg.keyboard.press("d")
    pg.wait_for_function("+document.getElementById('s-tick').textContent >= 2", timeout=30000)
    check("live" in pg.inner_text("#s-src"), "live mode computes ticks in the worker")
    pg.keyboard.press("d")
    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Precomputed'")
    pg.wait_for_function("+document.getElementById('o-tick').textContent >= 0")
    check(pg.locator("#l-tick").is_visible(), "tick slider back in precomputed mode")

    pg.keyboard.press("m")
    pg.wait_for_function("document.getElementById('mstat').textContent.startsWith('live')", timeout=60000)
    check(True, "Model view traces a precomputed tick")
    pg.keyboard.press("m")

    pg.set_viewport_size({"width": 390, "height": 800})
    pg.wait_for_timeout(300)
    check(pg.evaluate("document.documentElement.scrollWidth <= innerWidth"), "390px: no horizontal scroll")
    pg.screenshot(path=f"{out}/pre_phone.png")
    pg.set_viewport_size({"width": 1400, "height": 850})
    pg.evaluate("document.getElementById('b-pause').textContent.startsWith('Pause') && document.getElementById('b-pause').click()")
    pg.evaluate("(() => { const r = document.getElementById('r-tick'); r.value = 30; r.dispatchEvent(new Event('input')); })()")
    pg.wait_for_timeout(3500)
    pg.screenshot(path=f"{out}/pre_field.png")
    check(not logs, f"no console errors {logs}")
    b.close()
sys.exit(1 if errors else 0)
