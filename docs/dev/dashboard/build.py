#!/usr/bin/env python3
"""Issue-triage dashboard generator.

Reads the *live* GitHub backlog via `gh` and renders a single self-contained
`index.html` (no external assets, no JS deps) that visualizes the label
taxonomy from ../issue-labels.md: the kind x area heat map, the blocker
critical path, the upstream-audit cohort, the needs-RFC list, and the
s-class/effort batch queue. Every issue is a link.

Stdlib only. Requires `gh` on PATH, authenticated.

    python3 build.py                 # write index.html
    python3 build.py --serve         # build, then serve (regenerates per load)
    python3 build.py --serve --port 8799

To refresh a static build, just re-run it; under --serve every page load
re-fetches from gh, so the dashboard is always current.
"""

from __future__ import annotations

import argparse
import functools
import html
import json
import subprocess
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "index.html"

# Canonical display order (axes from ../issue-labels.md).
KIND_ORDER = [
    "kind/bug", "kind/feature", "kind/refactor",
    "kind/design", "kind/docs", "kind/question",
]
AREA_ORDER = [
    "area/compiler", "area/engine", "area/inference", "area/cli",
    "area/obs-model", "area/ir-schema", "area/testing",
]


def sh(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"`{' '.join(args)}` failed:\n{r.stderr}")
    return r.stdout


def repo_slug() -> str:
    return sh(["gh", "repo", "view", "--json", "nameWithOwner",
               "-q", ".nameWithOwner"]).strip()


def fetch_issues() -> list[dict]:
    raw = sh(["gh", "issue", "list", "--state", "open", "--limit", "500",
              "--json", "number,title,labels,url"])
    out = []
    for it in json.loads(raw):
        names = [lbl["name"] for lbl in it["labels"]]
        out.append({
            "n": it["number"],
            "t": it["title"],
            "url": it["url"],
            "kind": next((n for n in names if n.startswith("kind/")), None),
            "areas": [n for n in names if n.startswith("area/")],
            "effort": next((n for n in names if n.startswith("effort/")), None),
            "status": next((n for n in names if n.startswith("status/")), None),
            "blocker": "blocker" in names,
            "audit": "upstream-audit" in names,
        })
    return sorted(out, key=lambda x: x["n"])


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def search_url(repo: str, *labels: str, extra: str = "") -> str:
    q = "is:open " + " ".join(f'label:"{l}"' for l in labels)
    if extra:
        q += " " + extra
    return f"https://github.com/{repo}/issues?q=" + urllib.parse.quote(q)


def blob(repo: str, path: str) -> str:
    return f"https://github.com/{repo}/blob/main/{path}"


def heat(count: int, mx: int) -> str:
    if count == 0:
        return "background:#f6f8fa;color:#aab"
    t = count / mx if mx else 0.0
    return f"background:rgba(29,118,219,{0.10 + 0.55 * t:.3f});color:#0b1f33"


def issue_li(it: dict) -> str:
    badges = ""
    if it["blocker"]:
        badges += ' <span class="badge blk">blocker</span>'
    if it["effort"]:
        badges += f' <span class="badge">{esc(it["effort"])}</span>'
    if it["status"]:
        badges += f' <span class="badge">{esc(it["status"])}</span>'
    if it["audit"]:
        badges += ' <span class="badge aud">audit</span>'
    return (f'<li><a class="iss" href="{esc(it["url"])}">#{it["n"]}</a> '
            f'{esc(it["t"])}{badges}</li>')


def details(summary: str, items: list[dict], open_: bool = False) -> str:
    if not items:
        body = '<p class="empty">— none —</p>'
    else:
        body = "<ul>" + "".join(issue_li(i) for i in items) + "</ul>"
    op = " open" if open_ else ""
    return f"<details{op}><summary>{summary} <b>({len(items)})</b></summary>{body}</details>"


def render(repo: str, issues: list[dict]) -> str:
    total = len(issues)
    by_kind = {k: [i for i in issues if i["kind"] == k] for k in KIND_ORDER}
    by_area = {a: [i for i in issues if a in i["areas"]] for a in AREA_ORDER}
    blockers = [i for i in issues if i["blocker"]]
    design = by_kind["kind/design"]
    audit = [i for i in issues if i["audit"]]
    sclass = [i for i in issues
              if i["status"] == "status/s-class" or i["effort"] == "effort/S"]
    unclassified = [i for i in issues if i["kind"] is None or not i["areas"]]

    # kind x area cross-tab
    cell = {(k, a): sum(1 for i in by_kind[k] if a in i["areas"])
            for k in KIND_ORDER for a in AREA_ORDER}
    mx = max(cell.values()) if cell else 0

    head = "".join(
        f'<th><a href="{search_url(repo, a)}">{esc(a.split("/")[1])}</a></th>'
        for a in AREA_ORDER)
    rows = ""
    for k in KIND_ORDER:
        if not by_kind[k] and sum(cell[(k, a)] for a in AREA_ORDER) == 0:
            continue
        tds = ""
        for a in AREA_ORDER:
            c = cell[(k, a)]
            link = f'<a href="{search_url(repo, k, a)}">{c}</a>' if c else c
            tds += f'<td style="{heat(c, mx)}">{link}</td>'
        kname = k.split("/")[1]
        rows += (f'<tr><th class="rk"><a href="{search_url(repo, k)}">'
                 f'{esc(kname)}</a> <span class="ct">{len(by_kind[k])}</span>'
                 f'</th>{tds}</tr>')

    def stat(label, n, href=None):
        inner = (f'<a href="{href}">{n}</a>' if href else n)
        return f'<div class="stat"><div class="num">{inner}</div><div class="lab">{label}</div></div>'

    stats = "".join([
        stat("open", total, f"https://github.com/{repo}/issues"),
        stat("blockers", len(blockers), search_url(repo, "blocker")),
        stat("bugs", len(by_kind["kind/bug"]), search_url(repo, "kind/bug")),
        stat("features", len(by_kind["kind/feature"]), search_url(repo, "kind/feature")),
        stat("needs RFC", len(design), search_url(repo, "kind/design")),
        stat("audit cohort", len(audit), search_url(repo, "upstream-audit")),
    ])

    kind_blocks = "".join(
        details(esc(k), by_kind[k]) for k in KIND_ORDER if by_kind[k])
    area_blocks = "".join(
        details(esc(a), by_area[a]) for a in AREA_ORDER if by_area[a])

    unclass_note = ""
    if unclassified:
        unclass_note = (
            '<p class="warn">⚠ '
            f'{len(unclassified)} issue(s) missing a kind/ or area/: '
            + ", ".join(f'<a href="{esc(i["url"])}">#{i["n"]}</a>'
                        for i in unclassified) + "</p>")

    sclass_hint = ""
    if not sclass:
        sclass_hint = ('<p class="empty">Empty until the effort pass runs — '
                       '<code>effort/</code> and <code>status/s-class</code> '
                       'are set by reading each issue + a code peek, not from '
                       'titles. See the playbook below.</p>')

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    labels_doc = blob(repo, "docs/dev/issue-labels.md")
    tiers_doc = blob(repo, "docs/dev/issue-triage-tiers.md")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>camdl · issue triage</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0; color: #1b1f24; background: #fbfcfd; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 28px 22px 80px; }}
  h1 {{ font-size: 22px; margin: 0 0 2px; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: .04em;
        color: #57606a; margin: 34px 0 12px; border-bottom: 1px solid #e7ebef; padding-bottom: 6px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
  a {{ color: #1d76db; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .stat {{ flex: 1 1 120px; background: #fff; border: 1px solid #e7ebef; border-radius: 10px;
           padding: 14px 16px; }}
  .num {{ font-size: 26px; font-weight: 650; }}
  .lab {{ font-size: 12px; color: #6e7781; text-transform: uppercase; letter-spacing: .03em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e7ebef; padding: 6px 9px; text-align: center; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  td a {{ color: inherit; font-weight: 600; }}
  th.rk {{ text-align: left; white-space: nowrap; }}
  .ct {{ color: #8b949e; font-weight: 400; font-size: 11px; }}
  details {{ background: #fff; border: 1px solid #e7ebef; border-radius: 8px;
             margin: 7px 0; padding: 4px 12px; }}
  summary {{ cursor: pointer; padding: 6px 0; font-weight: 550; }}
  details ul {{ margin: 4px 0 10px; padding-left: 20px; }}
  details li {{ margin: 3px 0; }}
  a.iss {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  .badge {{ font-size: 11px; background: #eef1f4; color: #57606a; border-radius: 5px;
            padding: 1px 6px; margin-left: 2px; white-space: nowrap; }}
  .badge.blk {{ background: #ffd7d5; color: #b60205; font-weight: 600; }}
  .badge.aud {{ background: #efe5ff; color: #5319e7; }}
  .empty, .warn {{ color: #8b949e; font-size: 13px; }}
  .warn {{ color: #b60205; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 720px) {{ .cols {{ grid-template-columns: 1fr; }} }}
  .play td, .play th {{ text-align: left; }}
  code {{ background: #eef1f4; border-radius: 4px; padding: 1px 5px; font-size: 12px; }}
  footer {{ margin-top: 40px; color: #8b949e; font-size: 12px; }}
</style></head><body><div class="wrap">

<h1>camdl &middot; issue triage</h1>
<div class="meta">{esc(repo)} &middot; generated {now} &middot; axes per
  <a href="{labels_doc}">issue-labels.md</a></div>

<div class="stats">{stats}</div>
{unclass_note}

<h2>Critical path &mdash; blockers</h2>
{details("silent-wrong on the inference/sim path", blockers, open_=True)}

<h2>kind &times; area</h2>
<table><tr><th></th>{head}</tr>{rows}</table>

<h2>Batch queue &mdash; s-class / small</h2>
{details("effort/S or status/s-class", sclass, open_=True)}
{sclass_hint}

<h2>Cohorts</h2>
<div class="cols">
  <div>{details("upstream-audit", audit)}{details("needs an RFC (kind/design)", design)}</div>
  <div>{details("docs", by_kind["kind/docs"])}{details("refactor / tech-debt", by_kind["kind/refactor"])}</div>
</div>

<h2>By kind</h2>
{kind_blocks}

<h2>By area</h2>
{area_blocks}

<h2>Reduction playbook</h2>
<table class="play">
  <tr><th>Lever</th><th>Mechanism</th><th>Risk</th></tr>
  <tr><td>1 &middot; stale / dup sweep</td><td>verify against <code>main</code>, close with evidence &mdash; no code</td><td>none</td></tr>
  <tr><td>2 &middot; s-class batch</td><td>small, isolated, collision-free bugs w/ clean red&rarr;green, ~4 per worktree</td><td>low</td></tr>
  <tr><td>3 &middot; umbrella collapse</td><td>close satellites as one design lift lands (e.g. obs-model under #172)</td><td>mixed</td></tr>
  <tr><td>4 &middot; audit-cohort sprint</td><td>clear the <code>upstream-audit</code> engine backlog as a focused pass</td><td>med</td></tr>
  <tr><td>5 &middot; blocker correctness</td><td>careful inference pass &mdash; gates trust, not count</td><td>high</td></tr>
  <tr><td>6 &middot; design RFCs</td><td>proposal first, then build</td><td>&mdash;</td></tr>
</table>

<footer>Order &amp; discipline: <a href="{tiers_doc}">issue-triage-tiers.md</a> &middot;
  rebuild: <code>python3 docs/dev/dashboard/build.py</code></footer>

</div></body></html>
"""


def build() -> Path:
    repo = repo_slug()
    issues = fetch_issues()
    OUT.write_text(render(repo, issues), encoding="utf-8")
    return OUT


def serve(port: int) -> None:
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # regenerate on each top-level load → always live
            if self.path in ("/", "/index.html"):
                try:
                    build()
                except SystemExit as e:
                    self.send_error(500, str(e))
                    return
            return super().do_GET()

        def log_message(self, *a):  # quiet
            pass

    handler = functools.partial(Handler, directory=str(HERE))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving live dashboard at http://localhost:{port}/  (Ctrl-C to stop)")
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true",
                    help="serve and regenerate on each page load")
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    path = build()
    print(f"wrote {path}")
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
