import sys
from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter/index.html"
out = sys.argv[1] if len(sys.argv) > 1 else "."
errors = []

def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)

SIM = "window.__sim()"
PHI = "(() => { const d = window.__model.stage.block('phi').planes[0].data; let s = 0; for (let i = 0; i < d.length; i += 7) s = (s * 31 + d[i]) % 1000003; return s; })()"
SET_CH = "(v) => { const r = document.getElementById('r-ch'); r.value = v; r.dispatchEvent(new Event('input')); }"
DIST = "window.__model.stage.camera.position.distanceTo(window.__model.stage.controls.target)"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}).new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(base)
    pg.wait_for_function("window.__sim && window.__sim().tick >= 1", timeout=90000)
    check(pg.locator("#stage").count() == 0 and pg.locator("#b-field").count() == 0, "2D canvas and Field/Fit controls are gone")
    check(pg.locator("#stage3d").is_visible() and pg.locator("#r-ch").is_visible(), "3D view is the page, channel slider shown")
    check(pg.get_attribute("#r-ch", "min") == "-1", "channel slider reaches -1")
    check("260,419" in pg.inner_text("#mcap"), "caption shows parameter count from weights")
    check(pg.locator(".lab:visible").count() > 5, "block name labels visible")

    # C: wall framing
    c = pg.evaluate("""(() => {
      const st = window.__model.stage, d = st.camera.getWorldDirection(new st.camera.position.constructor()), n = st.block('enc0').planes[0].mesh.getWorldDirection(new st.camera.position.constructor());
      return { cam: [d.x, d.y, d.z], normal: [n.x, n.y, n.z], dot: d.dot(n), up: st.camera.up.y };
    })()""")
    check(c["cam"][2] < -0.95, f"default camera looks horizontally along the wall normal {c['cam']}")
    check(c["normal"][2] > 0.99 and abs(c["normal"][1]) < 0.01, f"slabs stand vertical, normal faces +z {c['normal']}")
    check(c["dot"] < -0.95, f"slab normals face the camera (dot {c['dot']:.3f})")
    ys = pg.evaluate("(() => { const s = window.__model.stage; const y = (id) => { const v = s.block(id).group.getWorldPosition(new s.camera.position.constructor()); return v.y; }; return [y('phik'), y('enc0'), y('dec0')]; })()")
    check(ys[0] > ys[1] > ys[2], f"kernel branch above encoder above decoder on the wall {ys}")

    # A: bodies and every slab show the same tick while playing
    seen, bad = {}, []
    for _ in range(40):
        s = pg.evaluate(SIM)
        sh = s["shown"]
        if not (sh["bodies"] == sh["slabs"] == s["tick"] == int(pg.inner_text("#o-tick"))):
            bad.append((sh, s["tick"]))
        seen.setdefault(s["tick"], pg.evaluate(PHI))
        pg.wait_for_timeout(250)
    check(not bad, f"bodies and slabs always show the same tick as the readout ({len(seen)} ticks sampled) {bad[:2]}")
    check(len(set(seen.values())) >= 3, f"slab texels change with the tick ({len(set(seen.values()))} distinct checksums)")
    s = pg.evaluate(SIM)
    check(s["bytes"] / s["frames"] < 3.5e6, f"frame size {s['bytes'] / s['frames'] / 1e6:.2f} MB, {s['frames']} frames held ({s['bytes'] / 1e6:.0f} MB)")
    print("     shown {:.2f} ticks/s, computed {:.2f} frames/s".format(s["shownRate"], s["prodRate"]))

    # lookahead: pausing lets the producer run ahead of the playhead
    pg.keyboard.press("Space")
    t0 = pg.evaluate(SIM)["tick"]
    pg.wait_for_timeout(9000)
    s = pg.evaluate(SIM)
    check(s["tick"] == t0 and s["ahead"] >= 5, f"paused playhead holds while {s['ahead']} frames are buffered ahead")
    check(pg.inner_text("#mstat").count("buffer") == 1 and f"buffer {s['ahead']}/100" in pg.inner_text("#mstat") or "buffer" in pg.inner_text("#mstat"), "buffer indicator shown: " + pg.inner_text("#mstat")[:70])
    pg.keyboard.press(".")
    pg.wait_for_timeout(200)
    s2 = pg.evaluate(SIM)
    check(s2["tick"] == t0 + 1 and s2["shown"]["bodies"] == s2["shown"]["slabs"] == t0 + 1, "step shows the buffered next tick for bodies and slabs together")
    k = sorted(seen)[len(seen) // 2]
    pg.evaluate("(k) => { const r = document.getElementById('r-tick'); r.value = k; r.dispatchEvent(new Event('input')); }", k)
    pg.wait_for_timeout(200)
    check(pg.evaluate(PHI) == seen[k] and pg.evaluate(SIM)["tick"] == k, f"scrubbing back to buffered tick {k} reproduces its slabs")
    pg.evaluate("(k) => { const r = document.getElementById('r-tick'); r.value = r.max; r.dispatchEvent(new Event('input')); }", 0)

    # B: all-channel overlay
    pg.evaluate(f"({SET_CH})(-1)")
    pg.wait_for_timeout(1500)
    check(pg.inner_text("#o-ch") == "all", "slider label reads 'all' at -1")
    r = pg.evaluate("""(() => {
      const st = window.__model.stage, res = {};
      for (const id of ['enc0', 'dec2', 'h0']) {
        const b = st.block(id), n = b.W * b.H;
        const vis = b.allPlanes.filter((m) => m.visible).length;
        let big = -1, bigV = 0, neg = -1, negV = 0, zero = -1, zeroV = 1e9;
        for (let c = 0; c < b.C; c++) for (let i = 0; i < n; i++) {
          const v = b.tensor[c * n + i];
          if (v > bigV) { bigV = v; big = [c, i]; }
          if (v < negV) { negV = v; neg = [c, i]; }
          if (Math.abs(v) < zeroV) { zeroV = Math.abs(v); zero = [c, i]; }
        }
        const px = (q) => Array.from(b.allPlanes[q[0]].userData.data.slice(4 * q[1], 4 * q[1] + 4));
        res[id] = { vis, base: b.planes[0].mesh.visible, big: px(big), neg: px(neg), zero: px(zero), bigV, negV, zeroV };
      }
      const inp = st.block('input');
      res.stack = { z: Array.from(inp.planes.map((p) => p.mesh.position.y)) };
      return res;
    })()""")
    for id in ("enc0", "dec2", "h0"):
        q = r[id]
        check(q["vis"] == 32 and not q["base"], f"{id}: 32 channel planes shown, single plane hidden")
        check(q["zero"][3] == 0, f"{id}: alpha 0 where |v| ~ 0 (|v|={q['zeroV']:.2e})")
        check(q["big"][3] > 20, f"{id}: alpha {q['big'][3]} where v={q['bigV']:.3f}")
        check(q["neg"][3] > 0 and q["neg"][:3] != q["big"][:3], f"{id}: sign changes colour {q['big'][:3]} vs {q['neg'][:3]}")
    check(abs(r["stack"]["z"][1] - r["stack"]["z"][0]) < 0.3, "input stack planes overlaid")
    pg.screenshot(path=out + "/model_all.png")
    pg.evaluate("window.__model.stage.setPose([20, -6, 0], [75, 12, 60])")
    pg.wait_for_timeout(600)
    pg.screenshot(path=out + "/model_all_oblique.png")
    shown = False
    for x in range(500, 1300, 25):
        for y in range(300, 700, 25):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(10)
            if pg.locator("#tip").is_visible() and "strongest" in pg.inner_text("#tip"):
                shown = True
                break
        if shown:
            break
    check(shown, "all mode tooltip names the strongest channel: " + pg.inner_text("#tip").replace("\n", " | ")[:80])
    pg.mouse.move(5, 5)
    pg.evaluate(f"({SET_CH})(3)")
    pg.wait_for_timeout(600)
    check(pg.evaluate("window.__model.stage.block('enc0').planes[0].mesh.visible") and pg.inner_text("#o-ch") == "3", "back to single channel 3")
    pg.evaluate("window.__model.fit()")
    pg.wait_for_timeout(300)
    pg.screenshot(path=out + "/model_single.png")

    # zoom survives hover
    pg.mouse.move(600, 450)
    d0 = pg.evaluate(DIST)
    for _ in range(5):
        pg.mouse.wheel(0, -300)
        pg.wait_for_timeout(60)
    d1 = pg.evaluate(DIST)
    check(d1 < d0 - 1e-3, f"wheel zooms ({d0:.1f} -> {d1:.1f})")
    for x in range(200, 1000, 40):
        for y in (260, 330, 400, 480):
            pg.mouse.move(x, y)
            pg.wait_for_timeout(25)
    pg.wait_for_timeout(1500)
    check(abs(pg.evaluate(DIST) - d1) < 1e-6, "hover and new frames do not reset zoom")
    pg.mouse.move(5, 5)
    pg.evaluate("window.__model.fit()")

    # past tick 100 without shipped data
    pg.keyboard.press("Space")
    pg.wait_for_function("window.__sim().tick >= 101", timeout=240000)
    s = pg.evaluate(SIM)
    check(s["tick"] >= 101 and s["shown"]["bodies"] == s["shown"]["slabs"], f"sim runs past tick 100 (tick {s['tick']}, buffer {s['ahead']})")
    check(s["frames"] <= 135, f"frame window bounded ({s['frames']} frames, {s['bytes'] / 1e6:.0f} MB)")

    pg.click("#b-mode")
    pg.wait_for_function("document.getElementById('b-mode').textContent === 'Random'")
    pg.wait_for_function("window.__sim().tick >= 2", timeout=60000)
    check(pg.evaluate(SIM)["epoch"] >= 2 and "random" in pg.evaluate("document.getElementById('s-src').textContent"), "random mode restarts the run")
    check(not logs, f"no console errors {logs[:3]}")

    ph = b.new_context(viewport={"width": 390, "height": 800}, has_touch=True).new_page()
    logs2 = []
    ph.on("console", lambda m: logs2.append(m.text) if m.type == "error" else None)
    ph.on("pageerror", lambda e: logs2.append(str(e)))
    ph.goto(base)
    ph.wait_for_function("window.__sim && window.__sim().tick >= 2", timeout=90000)
    ph.wait_for_timeout(800)
    check(ph.evaluate("document.documentElement.scrollWidth <= innerWidth"), "390px: no horizontal scroll")
    ph.screenshot(path=out + "/model_phone.png")
    check(not logs2, f"390px: no console errors {logs2[:3]}")
    b.close()

print("FAIL: " + "; ".join(errors) if errors else "ALL OK")
sys.exit(1 if errors else 0)
