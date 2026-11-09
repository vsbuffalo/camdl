# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright>=1.45"]
# ///
"""Screenshot tabs of a running camdl-scope viewer (`camdl-watch`).

Drives the installed Google Chrome headlessly (no browser download) and saves
one PNG per (run, tab) as OUT_DIR/scope-<name>-<tab>.png. Each SHOT is
NAME=RUN_ID:TAB[,TAB...], where RUN_ID is the fit directory the viewer lists
(e.g. negbin-1a2b3c4d) and NAME is the short name used in the file.

Usage: uv run scope_screenshots.py URL OUT_DIR SHOT [SHOT...]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(url: str, out_dir: str, shots: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(
            viewport={"width": 1100, "height": 900},
            device_scale_factor=2,
            color_scheme="light",
        )
        for shot in shots:
            name, spec = shot.split("=", 1)
            run_id, tabs = spec.split(":", 1)
            for tab in tabs.split(","):
                page.goto(f"{url}/?run={run_id}&tab={tab}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1500)  # charts draw after their data arrives
                # Keep the tab's own content: everything below the tab bar.
                bar = page.get_by_role("tab", name=tab.capitalize()).bounding_box()
                if bar is None:
                    sys.exit(f"no '{tab}' tab on {run_id}")
                top = bar["y"] + bar["height"] + 12
                height = page.evaluate("document.documentElement.scrollHeight") - top
                path = out / f"scope-{name}-{tab}.png"
                page.screenshot(
                    path=str(path), full_page=True,
                    clip={"x": 0, "y": top, "width": 1100, "height": height},
                )
                print(f"wrote {path}")
        browser.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
