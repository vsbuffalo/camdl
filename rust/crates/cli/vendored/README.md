# Vendored runtime dependencies for `camdl survey --render`

The HTML rendered by `camdl survey --render` is fully self-contained:
no CDN, no network at viewing time. The bundles below are vendored
into the binary via `include_str!` and embedded as `<script>` tags
in the output.

## plotly-2.35.2.min.js

- Source: https://cdn.plot.ly/plotly-2.35.2.min.js
- Version: 2.35.2 (released 2024-09-12)
- License: MIT (see header comment in the file)
- Approx size: 4.5 MB minified
- Pinned by: `survey/landscape_html.rs` `include_str!("../vendored/plotly-2.35.2.min.js")`

To bump the version, replace the file *and* update the `include_str!`
path. The version string also appears in the proposal
(`docs/dev/proposals/2026-05-03-survey-subcommand.md`) and is pinned
to a fixed bundle so output HTMLs reproduce byte-for-byte across
camdl versions until we deliberately roll the dependency.

## pairplot.js

- Source: hand-written for camdl, lives under our own license
  (same as the rest of the repo)
- Implements the pair-plot visual style ported from
  `camdl-book/.claude/worktrees/typhoid/_lib/identifiability.py`'s
  `pairplot()` function: three-layer diagonal histograms (gray /
  green / red), viridis_r off-diagonal scatter, top-5 red stars
- Reads two `<script type="application/json">` blocks emitted by
  `landscape_html.rs`:
  - `landscape-data` — TSV rows + estimated names + log-axis flags
  - `landscape-bounds` — explicit (lo, hi) per estimated param
- Exposes interactive controls:
  - `#top-pct` slider — top-K percentile cutoff (live re-bin)
  - `#color-by` dropdown — drives off-diagonal viridis (loglik /
    loglik_se / mean_ess)
  - axis-label click toggles log / linear
  - hover, brushing — provided by plotly's selectedpoints API
