import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter-tiled/index.html"
out = sys.argv[1] if len(sys.argv) > 1 else "."
errors = []


def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)


APP = "window.__app()"
FR = "(() => { const c = window.__view.ctx; return { tick: c.state.frame.tick, sim: c.state.sim.tick, flags: c.state.frame.flags, unet: c.state.frame.unet ? c.state.frame.unet.tile : -1, rings: c.W.rings.count, n: c.state.frame.n, nt: c.W.nt }; })()"
PROJ = """(i) => { const c = window.__view.ctx, s = c.state, W = c.W, v = new c.camera.position.constructor(); const L = W.L, p = s.frame.pos;
  v.set(-13 + p[2 * i] / L * 26, 24 + p[2 * i + 1] / L * 26, 0.35).project(c.camera); return [(v.x + 1) * innerWidth / 2, (1 - v.y) * innerHeight / 2]; }"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1500, "height": 880}).new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(base + "?nt=2")
    pg.wait_for_function("window.__app && window.__app().tick >= 2", timeout=300000)
    check(pg.locator("#stage3d").is_visible() and pg.locator("#b-ghost").is_visible(), "3D view and mechanism toggles shown")
    check(pg.locator("a[href*=walkthrough]").count() == 0, "no walkthrough link")
    bad = []
    for _ in range(12):
        f = pg.evaluate(FR)
        if not (f["tick"] == f["sim"] == pg.evaluate(APP)["tick"]):
            bad.append(f)
        pg.wait_for_timeout(300)
    check(not bad, f"model frame, sim frame and readout always show the same tick {bad[:2]}")
    pg.keyboard.press("Space")
    t0 = pg.evaluate(APP)["tick"]
    f0 = pg.evaluate(FR)
    check(f0["rings"] > 0 and f0["flags"] == {"ghosts": True, "fx": True, "far": True}, f"ghost rings drawn ({f0['rings']}), all mechanisms on")
    s = pg.evaluate(APP)
    check(s["bytes"] / s["frames"] < 4.5e6, f"frame size {s['bytes'] / s['frames'] / 1e6:.2f} MB, {s['frames']} frames")

    # toggling a mechanism branches the model from the tick on screen
    for key, name in (("g", "ghosts"), ("f", "far"), ("x", "fx")):
        pos_before = pg.evaluate("Array.from(window.__view.ctx.state.frame.pos)")
        pg.keyboard.press(key)
        pg.wait_for_function(f"window.__app().flags.{name} === false", timeout=5000)
        pg.keyboard.press(".")
        pg.wait_for_function(f"window.__view.ctx.state.frame.flags.{name} === false", timeout=120000)
        f = pg.evaluate(FR)
        pos_after = pg.evaluate("Array.from(window.__view.ctx.state.frame.pos)")
        d = max(abs(a - c) for a, c in zip(pos_before, pos_after))
        check(f["flags"][name] is False and f["tick"] == f["sim"], f"{name} off: next frame (tick {f['tick']}) computed without it, still in lockstep, max step {d:.2f}")
        if name == "ghosts":
            check(f["rings"] == 0, "no ghost rings with halo bodies off")
        pg.keyboard.press(key)
        pg.wait_for_function(f"window.__app().flags.{name} === true", timeout=5000)
    pg.screenshot(path=out + "/ui_toggled.png")

    # inspecting another tile moves the U-Net to it
    pg.evaluate("window.__view.focus('tiles')")
    pg.wait_for_timeout(400)
    tk = pg.evaluate(FR)["tick"]
    pg.keyboard.press(".")
    pg.wait_for_function(f"window.__view.ctx.state.frame.tick > {tk}", timeout=120000)
    xy = pg.evaluate("""(() => { const c = window.__view.ctx, W = c.W, b = W.blocks.get('t0'), v = new c.camera.position.constructor(); const g = b.group.position; v.set(g.x, -g.z, 0).project(c.camera); return [(v.x + 1) * innerWidth / 2, (1 - v.y) * innerHeight / 2]; })()""")
    pg.mouse.move(xy[0], xy[1])
    pg.wait_for_timeout(300)
    tip = pg.inner_text("#tip") if pg.locator("#tip:visible").count() else ""
    check("tile 0,0" in tip, f"hover on tile 0,0 gives a tooltip: {tip[:60]!r}")
    pg.mouse.click(xy[0], xy[1])
    pg.wait_for_function("window.__app().inspect === 0", timeout=5000)
    pg.keyboard.press(".")
    pg.wait_for_function("window.__view.ctx.state.frame.unet && window.__view.ctx.state.frame.unet.tile === 0", timeout=180000)
    check(True, "U-Net now shows tile 0,0")

    # hover a model body
    pg.evaluate("window.__view.focus('sim')")
    pg.wait_for_timeout(400)
    xy = pg.evaluate("(" + PROJ + ")(5)")
    pg.mouse.move(xy[0], xy[1])
    pg.wait_for_timeout(400)
    tip = pg.inner_text("#tip") if pg.locator("#tip:visible").count() else ""
    check("body" in tip and "sim pos" in tip, f"hover on a model body shows model and sim state: {tip[:50]!r}")
    pg.screenshot(path=out + "/ui_hover.png")

    # layers and stats
    pg.keyboard.press("2")
    pg.keyboard.press("d")
    pg.wait_for_timeout(500)
    check(pg.locator("#stats").is_visible(), "stats panel opens")
    pg.keyboard.press("1")

    # grid size switch
    pg.click("#b-nt4")
    pg.wait_for_function("window.__app().nt === 4 && window.__app().tick >= 0", timeout=300000)
    f = pg.evaluate(FR)
    check(f["nt"] == 4 and f["n"] == 300, f"4x4 run started with {f['n']} bodies")
    check(not logs, f"no console errors {logs[:3]}")
    b.close()
sys.exit(1 if errors else 0)
