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

    pg.goto(BASE + "/scatter/")
    pg.wait_for_function("parseInt(document.getElementById('s-tick').textContent) >= 3", timeout=60000)
    err_txt = pg.evaluate("document.getElementById('s-err').textContent")
    check("scatter-field model ticks and reports error", err_txt != "-", err_txt)
    check("dev statistics hidden by default", not pg.is_visible("#debug"))
    pg.keyboard.press("d")
    check("d opens dev statistics", pg.is_visible("#debug"))
    with pg.expect_download(timeout=20000) as dl:
        pg.click("#b-export")
    check("export state downloads json", dl.value.suggested_filename.startswith("scatter_state_tick"), dl.value.suggested_filename)
    check("scatter page links back to landing", pg.get_attribute("header a", "href") == "../")

    check("no page/console errors across all four pages", not errs, errs[:5])
    b.close()

sys.exit(1 if check.failed else 0)
