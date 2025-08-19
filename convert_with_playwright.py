# convert_with_playwright.py
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
import time

def convert(html_path, pdf_path, wait=1.0):
    html_uri = Path(html_path).absolute().as_uri()
    pdf_path = str(Path(pdf_path).absolute())

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(html_uri, wait_until="networkidle")
        time.sleep(wait)

        # hide UI that doesn't print well (tweak selectors if needed)
        page.add_style_tag(content="""
          .navbar, .topbar, .profile-header, .pg-controls, .btn, .dropdown { display: none !important; }
          .content, .container, .report-container { max-width: 100% !important; width: 100% !important; }
          .card, .section, .variable { page-break-inside: avoid; }
          * { -webkit-print-color-adjust: exact; color-adjust: exact; }
        """)
        page.add_style_tag(content="@page { size: A4; margin: 12mm; }")

        page.pdf(path=pdf_path, format="A4", print_background=True)
        browser.close()
    return pdf_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_with_playwright.py <reports/file.html> [out.pdf]")
        sys.exit(1)
    html = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else (Path(html).with_suffix(".pw.pdf"))
    print("Converting:", html, "->", out)
    print(convert(html, out))
    print("Done.")
