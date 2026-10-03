"""Smoke-tests the Pages site: landing page links, and the three model pages load,
run a few ticks and produce no console/page errors. Adapted from the pages
worktree's original single-model version (.claude/worktrees/pages).

Usage: python3 -m http.server 8765 --directory web &
       python3 scripts/test_web_ui.py [base_url] [chromium_executable]
"""
import sys
import time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
EXE = sys.argv[2] if len(sys.argv) > 2 else None
ARGS = ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        check.failed += 1


check.failed = 0

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=ARGS)
    pg = b.new_context(viewport={"width": 1400, "height": 900}).new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

    pg.goto(BASE + "/")
    cards = pg.eval_on_selector_all("a.card", "els => els.map(e => e.getAttribute('href'))")
    check("landing page has all three cards", sorted(cards) == ["gravity/", "scatter/", "token/"], cards)

    pg.click("a.card[href='token/']")
    pg.wait_for_url("**/token/**", timeout=10000)
    pg.wait_for_function("document.getElementById('s-balls').textContent !== '0'", timeout=20000)
    time.sleep(1.5)
    ticks1 = pg.evaluate("document.getElementById('s-tick').textContent")
    time.sleep(1)
    ticks2 = pg.evaluate("document.getElementById('s-tick').textContent")
    check("token model ticks", int(ticks2) > int(ticks1), (ticks1, ticks2))

    pg.goto(BASE + "/gravity/")
    pg.wait_for_function("document.getElementById('s-bodies').textContent !== '0'", timeout=20000)
    time.sleep(1.5)
    ticks1 = pg.evaluate("document.getElementById('s-tick').textContent")
    time.sleep(1)
    ticks2 = pg.evaluate("document.getElementById('s-tick').textContent")
    check("central-force model ticks", int(ticks2) > int(ticks1), (ticks1, ticks2))

    back = pg.get_attribute("header a", "href")
    check("gravity page links back to landing", back == "../")

    workers = []
    pg.on("worker", lambda w: workers.append(w.url))
    pg.goto(BASE + "/scatter/")
    pg.wait_for_function("document.getElementById('s-ckpt').textContent !== '-'", timeout=60000)
    pg.keyboard.press("d")
    check("d opens dev statistics", pg.is_visible("#debug"))
    pg.wait_for_function("parseInt(document.getElementById('s-tick').textContent) >= 3", timeout=60000)
    check("truth off by default (stat)", pg.evaluate("document.getElementById('s-err').textContent") == "off")
    check("truth checkbox unchecked", not pg.is_checked("#c-truth"))
    check("no truth worker created", not any("truth_worker" in w for w in workers), workers)
    check("error plot hidden while truth off", not pg.is_visible("#c-err"))
    check("energy plot visible in dev menu", pg.is_visible("#c-energy"))
    check("no truth button in nav", pg.query_selector("#b-truth") is None)
    check("header has no link", pg.query_selector("header a") is None and pg.query_selector("header p") is None)
    check("no aside plot panel", pg.query_selector("aside") is None)
    pg.keyboard.press("d")
    check("dev statistics hidden after toggle", not pg.is_visible("#debug"))
    pg.screenshot(path="/tmp/scatter_default.png")
    pg.keyboard.press("d")

    pg.mouse.move(700, 300)
    time.sleep(0.4)
    check("nav faded when pointer elsewhere", float(pg.evaluate("getComputedStyle(document.querySelector('nav')).opacity")) < 0.5)
    box = pg.eval_on_selector("nav", "e => { const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }")
    pg.mouse.move(box[0], box[1])
    time.sleep(0.4)
    check("nav opaque when hovered", float(pg.evaluate("getComputedStyle(document.querySelector('nav')).opacity")) == 1.0)

    def zoom_txt():
        return pg.evaluate("document.getElementById('s-zoom').textContent")

    def centre_txt():
        return pg.evaluate("document.getElementById('s-centre').textContent")

    time.sleep(0.6)
    check("starts in auto view", "auto" in zoom_txt(), zoom_txt())
    pg.mouse.move(700, 400)
    pg.mouse.wheel(0, -600)
    time.sleep(0.6)
    z = zoom_txt()
    check("wheel zoom changes zoom stat", "manual" in z and not z.startswith("1.00x"), z)
    pg.keyboard.press("d")
    pg.keyboard.press("d")
    c0 = centre_txt()
    pg.mouse.move(700, 400)
    pg.mouse.down()
    pg.mouse.move(600, 350, steps=5)
    pg.mouse.up()
    time.sleep(0.6)
    check("drag pan changes centre", centre_txt() != c0, (c0, centre_txt()))
    pg.screenshot(path="/tmp/scatter_zoomed.png")
    pg.click("#b-fit")
    time.sleep(0.6)
    check("Fit returns to auto", "auto" in zoom_txt(), zoom_txt())
    pg.keyboard.press("+")
    time.sleep(0.6)
    check("+ key zooms", "manual" in zoom_txt(), zoom_txt())
    pg.keyboard.press("0")
    time.sleep(0.6)
    check("0 key returns to auto", "auto" in zoom_txt(), zoom_txt())

    pg.check("#c-truth")
    pg.wait_for_function("document.getElementById('s-err').textContent !== 'off' && document.getElementById('s-err').textContent !== '-'", timeout=60000)
    check("truth worker created on enable", any("truth_worker" in w for w in workers), workers)
    check("error plot visible with truth on", pg.is_visible("#c-err"))
    check("truth since stat set", pg.evaluate("document.getElementById('s-tsince').textContent").startswith("tick"))
    pg.screenshot(path="/tmp/scatter_truth.png")
    with pg.expect_download(timeout=20000) as dl:
        pg.click("#b-export")
    check("export state downloads json", dl.value.suggested_filename.startswith("scatter_state_tick"), dl.value.suggested_filename)
    pg.uncheck("#c-truth")
    check("truth off again", pg.evaluate("document.getElementById('s-err').textContent") == "off")
    pg.keyboard.press("d")

    check("no page/console errors across all four pages", not errs, errs[:5])
    b.close()

sys.exit(1 if check.failed else 0)
