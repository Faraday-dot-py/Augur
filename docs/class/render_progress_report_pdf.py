"""Render docs/class/progress-report.html to docs/class/progress-report.pdf (US Letter, 1 inch margins).

Usage: python3 docs/class/render_progress_report_pdf.py
"""
import pathlib

from playwright.sync_api import sync_playwright
from pypdf import PdfReader

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "progress-report.html"
OUT = HERE / "progress-report.pdf"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(SRC.as_uri())
    pg.pdf(path=str(OUT), format="Letter", margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
           print_background=True, prefer_css_page_size=True)
    b.close()

pages = len(PdfReader(str(OUT)).pages)
print(f"pages: {pages}")
if pages > 2:
    raise SystemExit("over the 2 page limit")
