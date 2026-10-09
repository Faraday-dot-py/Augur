import json
import sys

from playwright.sync_api import sync_playwright

base = "http://localhost:8765/scatter/index.html"
errors = []


def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        errors.append(msg)


with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_context(viewport={"width": 1400, "height": 850}, accept_downloads=True).new_page()
    pg.goto(base)
    pg.wait_for_function("window.__sim && window.__sim().tick >= 3", timeout=90000)
    pg.keyboard.press("d")
    with pg.expect_download() as dl:
        pg.click("#b-export")
    snap = json.load(open(dl.value.path()))
    check(snap["seed"] == 4738 and snap["mode"] == "fixed", "export has seed 4738 and mode")
    check(snap["meta"] and snap["meta"]["checkpoint"] and snap["exportedAt"], "export has meta and exportedAt")
    log = snap["startLog"]
    check(len(log["E"]) >= 4 and len(log["E"]) == len(log["outside"]), f"startLog has {len(log['E'])} ticks of E and outside")
    check(all(isinstance(x, (int, float)) for x in log["E"]) and log["outside"][0] == 0, "startLog values numeric, 0 outside at tick 0")
    b.close()
sys.exit(1 if errors else 0)
