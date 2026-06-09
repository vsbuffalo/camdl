# Issue-triage dashboard

A local, self-contained HTML view of the open GitHub backlog, classified by the
label taxonomy in [`../issue-labels.md`](../issue-labels.md): the kind × area
heat map, the blocker critical path, the upstream-audit cohort, the needs-RFC
list, and the s-class/effort batch queue. Every issue links to GitHub; the table
cells and stat tiles link to filtered issue searches.

Issues authored by anyone other than the maintainer (`MAINTAINER` in `build.py`,
default `vsbuffalo`) carry an `@handle` **external** badge and roll up into an
"external reporters" cohort — a dashboard-only marker derived from issue
authorship, not a GitHub label.

The dashboard is **generated**, not committed — `index.html` is gitignored. The
data is the _live_ GitHub label state, pulled via `gh`, so the dashboard is only
as current as its last build.

## Use

```bash
# build a static snapshot → docs/dev/dashboard/index.html
python3 docs/dev/dashboard/build.py

# build + serve; every page load re-fetches from gh (always live)
python3 docs/dev/dashboard/build.py --serve            # http://localhost:8799/
python3 docs/dev/dashboard/build.py --serve --port 9000
```

Stdlib only — no `uv`/pip deps. Needs `gh` on PATH and authenticated
(`gh auth status`). To open a static build without the server, just open
`index.html` in a browser.

## Momentum / trends

The top panel shows direction over `WINDOWS` (default **1d / 3d / 7d** — edit
the constant in `build.py`; `7` → `5` for a work-week). Down is green, up is
red. Two data sources, because GitHub's API gives current state, not history:

- **Open-issue trend is exact and backfilled** from issue `createdAt`/`closedAt`
  timestamps — `Δ = opened − closed` over the window. Works immediately, no log.
- **Label trends (blockers, bugs, …) accrue forward** from a snapshot log,
  because the taxonomy labels carry no usable past (they were created the day
  the scheme landed). Until a snapshot is old enough, those cells read `·`.

```bash
python3 docs/dev/dashboard/build.py --snapshot   # record today's counts
```

Run `--snapshot` once a day (cron / a GitHub Action) to build the series. The
log is `snapshots.ndjson` (one row per UTC date, last write wins) and is
gitignored — commit it, or have an Action commit it, for durable shared history.
A plain `--serve` never writes the log.

## What it shows

Read-only over the labels — it does not change any issue. To act on it, apply
labels (`gh issue edit …`) and reload. The reduction strategy the panels are
organized around is in [`../issue-triage-tiers.md`](../issue-triage-tiers.md).
