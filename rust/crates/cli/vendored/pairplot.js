// camdl survey pair-plot renderer.
//
// Reads:
//   <script type="application/json" id="landscape-data">  // TSV → JSON
//   <script type="application/json" id="landscape-bounds"> // [(lo,hi), ...]
//
// Visual style ported from
// camdl-book/.claude/worktrees/typhoid/_lib/identifiability.py
// (pairplot function): three-layer diagonal histograms, viridis_r
// off-diagonal scatter colored by |loglik - loglik_max| on a log
// scale, top-5 red stars.

(function () {
  "use strict";

  const COLORS = {
    GREY_BG:   "#e0e0e0",
    GREY_HIST: "#cccccc",
    TOP10_GREEN: "#16a085",
    TOP1_RED:    "#c0392b",
    REF_BLACK: "#000000",
  };

  const dataNode = document.getElementById("landscape-data");
  const boundsNode = document.getElementById("landscape-bounds");
  if (!dataNode || !boundsNode) {
    console.error("camdl survey: missing landscape data block");
    return;
  }
  const D = JSON.parse(dataNode.textContent);
  const declaredBounds = JSON.parse(boundsNode.textContent);

  const params = D.estimated;
  const nParams = params.length;
  const rows = D.rows.filter((r) => Number.isFinite(r.loglik));
  if (!rows.length) {
    document.getElementById("plot").innerText =
      "no finite-loglik rows in landscape.tsv — nothing to plot";
    return;
  }
  // Pick log axis when the declared bounds span > 2 decades
  // (matches the Python prototype's heuristic).
  const logAxis = params.map((_, i) => {
    const [lo, hi] = declaredBounds[i] || [NaN, NaN];
    return Number.isFinite(lo) && Number.isFinite(hi)
      && lo > 0 && hi / lo > 100;
  });

  // Top-K cutoff — the slider drives this; default 10%.
  const slider = document.getElementById("top-pct");
  const sliderDisplay = document.getElementById("top-pct-display");
  const colorBy = document.getElementById("color-by");

  // Hide the mean_ess option when not present.
  if (!D.has_mean_ess) {
    const opt = colorBy.querySelector('option[value="mean_ess"]');
    if (opt) opt.disabled = true;
  }

  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  // ── Color scaling helpers ────────────────────────────────────────
  // viridis_r palette (10 sample points, evenly spaced).
  const VIRIDIS_R = [
    "#fde725", "#b5de2b", "#6ece58", "#35b779", "#1f9e89",
    "#26828e", "#31688e", "#3e4989", "#482878", "#440154",
  ];
  function viridis(v) {
    const idx = Math.floor(clamp01(v) * (VIRIDIS_R.length - 1));
    return VIRIDIS_R[idx];
  }

  // Map |loglik - loglik_max| to a [0, 1] color coord on a log scale,
  // matching the Python LogNorm behaviour.
  function colorCoord(rowsArr, key) {
    const vals = rowsArr.map((r) => r[key]).filter(Number.isFinite);
    if (!vals.length) return rowsArr.map(() => 0);
    if (key === "loglik") {
      const maxLL = Math.max(...vals);
      const deltas = rowsArr.map((r) =>
        Number.isFinite(r.loglik) ? Math.max(maxLL - r.loglik, 1e-3) : NaN);
      const finiteDelt = deltas.filter(Number.isFinite);
      const lo = Math.min(...finiteDelt);
      const hi = Math.max(...finiteDelt);
      if (hi <= lo) return deltas.map(() => 0);
      return deltas.map((d) =>
        Number.isFinite(d) ? (Math.log(d) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : 0);
    } else {
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      if (hi <= lo) return rowsArr.map(() => 0);
      return rowsArr.map((r) =>
        Number.isFinite(r[key]) ? (r[key] - lo) / (hi - lo) : 0);
    }
  }

  // ── Rebuild the pair-plot from the current top-K + color-by ─────
  function render() {
    const topPct = parseFloat(slider.value);
    sliderDisplay.textContent = (100.0 * topPct).toFixed(1) + "%";
    const colorKey = colorBy.value;

    // Sort indices by loglik desc, then split into:
    //  - bottom (1 - topPct) "gray"
    //  - middle topPct "green"  (top X%)
    //  - top 1% "red"           (always carved out; needed for the diagonals)
    //  - top 5 "stars"          (red stars on off-diagonals)
    const sortedIdx = [...rows.keys()].sort((a, b) => rows[b].loglik - rows[a].loglik);
    const nGreen = Math.max(1, Math.floor(rows.length * topPct));
    const nRed = Math.max(1, Math.floor(rows.length * 0.01));
    const greenIdx = new Set(sortedIdx.slice(0, nGreen));
    const redIdx = new Set(sortedIdx.slice(0, nRed));
    const starIdx = new Set(sortedIdx.slice(0, Math.min(5, rows.length)));

    const bandFor = (i) => {
      if (redIdx.has(i)) return "red";
      if (greenIdx.has(i)) return "green";
      return "gray";
    };

    // Color coords for the off-diagonal viridis layer (top-K only).
    const cc = colorCoord(rows, colorKey);

    const traces = [];
    const layout = {
      grid: { rows: nParams, columns: nParams, pattern: "independent" },
      showlegend: false,
      hovermode: "closest",
      margin: { l: 60, r: 30, t: 30, b: 60 },
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      annotations: [],
    };

    for (let i = 0; i < nParams; i++) {
      for (let j = 0; j < nParams; j++) {
        const subplotId = i * nParams + j + 1;
        const xRef = j === 0 ? "xaxis" : `xaxis${subplotId}`;
        const yRef = i === 0 ? "yaxis" : `yaxis${subplotId}`;
        const xkey = j === 0 ? "x" : `x${subplotId}`;
        const ykey = i === 0 ? "y" : `y${subplotId}`;

        // Set axis layout for this subplot.
        layout[xRef] = {
          title: i === nParams - 1 ? params[j] : "",
          type: logAxis[j] ? "log" : "linear",
          showgrid: true, gridcolor: "#f0f0f0", zeroline: false,
        };
        layout[yRef] = {
          title: j === 0 ? params[i] : "",
          type: i === j ? "linear" : (logAxis[i] ? "log" : "linear"),
          showgrid: true, gridcolor: "#f0f0f0", zeroline: false,
        };

        if (i === j) {
          // ── Diagonal: three-layer histogram.
          const allVals = rows.map((r) => r.params[i]);
          // Bin edges: 40 bins across the declared bounds.
          const [lo, hi] = declaredBounds[i] || [
            Math.min(...allVals.filter(Number.isFinite)),
            Math.max(...allVals.filter(Number.isFinite)),
          ];
          const useLog = logAxis[i] && lo > 0;
          const nbins = 40;
          const edges = [];
          for (let b = 0; b <= nbins; b++) {
            const t = b / nbins;
            edges.push(useLog ? lo * Math.pow(hi / lo, t) : lo + t * (hi - lo));
          }
          const grayVals = [], greenVals = [], redVals = [];
          rows.forEach((r, k) => {
            const v = r.params[i];
            const band = bandFor(k);
            if (band === "red")        redVals.push(v);
            else if (band === "green") greenVals.push(v);
            else                       grayVals.push(v);
          });
          // Plotly histograms with explicit xbins to share bins across layers.
          const histogram = (vals, color, name) => ({
            type: "histogram", x: vals,
            xbins: { start: edges[0], end: edges[edges.length - 1],
              size: useLog ? null : (edges[1] - edges[0]) },
            marker: { color },
            opacity: 1.0, name,
            xaxis: xkey, yaxis: ykey,
          });
          traces.push(histogram(grayVals,  COLORS.GREY_HIST,  "bottom"));
          traces.push(histogram(greenVals, COLORS.TOP10_GREEN, "top X%"));
          traces.push(histogram(redVals,   COLORS.TOP1_RED,   "top 1%"));
        } else {
          // ── Off-diagonal: gray scatter + top-K viridis + top-5 stars.
          const grayPts = { x: [], y: [], text: [] };
          const topPts  = { x: [], y: [], text: [], color: [] };
          const starPts = { x: [], y: [], text: [] };
          rows.forEach((r, k) => {
            const xv = r.params[j], yv = r.params[i];
            if (!Number.isFinite(xv) || !Number.isFinite(yv)) return;
            const tooltip = paramHover(r, params, j, i);
            if (starIdx.has(k)) {
              starPts.x.push(xv); starPts.y.push(yv); starPts.text.push(tooltip);
            }
            if (greenIdx.has(k) || redIdx.has(k)) {
              topPts.x.push(xv); topPts.y.push(yv); topPts.text.push(tooltip);
              topPts.color.push(viridis(cc[k]));
            } else {
              grayPts.x.push(xv); grayPts.y.push(yv); grayPts.text.push(tooltip);
            }
          });
          traces.push({
            type: "scattergl", mode: "markers",
            x: grayPts.x, y: grayPts.y, text: grayPts.text,
            marker: { color: COLORS.GREY_HIST, size: 4, opacity: 0.4,
              line: { width: 0 } },
            hoverinfo: "text",
            xaxis: xkey, yaxis: ykey, showlegend: false,
          });
          traces.push({
            type: "scattergl", mode: "markers",
            x: topPts.x, y: topPts.y, text: topPts.text,
            marker: { color: topPts.color, size: 7, opacity: 0.85,
              line: { color: "#000", width: 0.3 } },
            hoverinfo: "text",
            xaxis: xkey, yaxis: ykey, showlegend: false,
          });
          traces.push({
            type: "scattergl", mode: "markers",
            x: starPts.x, y: starPts.y, text: starPts.text,
            marker: { color: COLORS.TOP1_RED, size: 12, symbol: "star",
              line: { color: "#000", width: 1 } },
            hoverinfo: "text",
            xaxis: xkey, yaxis: ykey, showlegend: false,
          });
        }
      }
    }

    Plotly.react("plot", traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "autoScale2d"],
    });
  }

  function paramHover(r, paramNames, jx, iy) {
    const lines = [];
    paramNames.forEach((p, k) => {
      lines.push(`${p}: ${formatNum(r.params[k])}`);
    });
    lines.push(`loglik: ${formatNum(r.loglik)}`);
    if (Number.isFinite(r.loglik_se)) lines.push(`se: ${formatNum(r.loglik_se)}`);
    lines.push(`point_id: ${r.point_id}`);
    return lines.join("<br>");
  }
  function formatNum(v) {
    if (!Number.isFinite(v)) return "" + v;
    if (Math.abs(v) >= 1e5 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) {
      return v.toExponential(3);
    }
    return v.toPrecision(4);
  }

  slider.addEventListener("input", render);
  colorBy.addEventListener("change", render);
  // Click on an axis title swaps log/linear for that parameter.
  document.getElementById("plot").addEventListener("plotly_clickannotation", () => render());
  // Cheap toggle: keyboard shortcut "L" toggles every axis.
  document.addEventListener("keydown", (e) => {
    if (e.key === "L" || e.key === "l") {
      for (let i = 0; i < nParams; i++) logAxis[i] = !logAxis[i];
      render();
    }
  });

  render();
})();
