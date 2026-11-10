//! Render a self-contained interactive pair-plot HTML from a
//! `landscape.tsv` written by `camdl survey --render`.
//!
//! Visual style (port of
//! `camdl-book/.claude/worktrees/typhoid/_lib/identifiability.py`'s
//! `pairplot()` function):
//! - Three-layer diagonal histograms: gray bottom, green mid (top
//!   X%), red top (top 1%)
//! - viridis_r off-diagonals encoding |loglik − loglik_max| on a log
//!   scale, with top-5 red stars
//! - log axes when the parameter's bounds span > 2 decades
//! - Color palette: GREY_BG = #e0e0e0, GREY_HIST = #cccccc,
//!   TOP10_GREEN = #16a085, TOP1_RED = #c0392b, REF_BLACK = #000
//!
//! Embedded inline (no CDN) via `include_str!` of a vendored
//! `plotly.min.js`. Pinned at compile time.

use std::path::Path;
use std::collections::HashMap;

/// The render-side projection of a survey's display fields — the subset the
/// pair-plot needs. Decouples `landscape_html` from the (now content-addressed)
/// survey inputs (gh#147 M3.3); the caller fills it from the resolved survey.
pub struct LandscapeMeta {
    /// Page title (model stem); `None` → "camdl survey".
    pub stem: Option<String>,
    /// Short run hash for the subtitle.
    pub hash_short: String,
    /// Estimated param names (the LHS box axes).
    pub estimated: Vec<String>,
    /// Per-param user-declared bounds → axis ranges.
    pub bounds: HashMap<String, (f64, f64)>,
    pub n_points: usize,
    pub eval_method: crate::run_meta::SurveyEvalMethod,
}

/// Vendored plotly.js bundle. Pinned to plotly.js v2.35.2 (released
/// 2024-09-12). MIT-licensed; see vendored/plotly-2.35.2.min.js
/// header for the upstream license text. Bumping the version → bump
/// the include_str! path.
const PLOTLY_BUNDLE: &str = include_str!("../vendored/plotly-2.35.2.min.js");

/// Render `<run_dir>/landscape.html` given the freshly-written
/// `landscape.tsv` and the survey's typed inputs (used only for
/// the run-hash + parameter list embedded in the page).
///
/// Layout:
///
/// ```html
/// <!doctype html>
/// <html>
///   <head>
///     <meta charset="utf-8">
///     <title>camdl survey — {stem}</title>
///     <style>...</style>
///   </head>
///   <body>
///     <div id="header">...</div>
///     <div id="plot"></div>
///     <div id="controls">...</div>
///     <script>{plotly bundle}</script>
///     <script type="application/json" id="landscape-data">
///       {tsv-as-json}
///     </script>
///     <script>{pairplot.js}</script>
///   </body>
/// </html>
/// ```
pub fn render(html_path: &Path, meta: &LandscapeMeta) -> Result<(), String> {
    let landscape_tsv = html_path.with_file_name("landscape.tsv");
    let tsv_text = std::fs::read_to_string(&landscape_tsv)
        .map_err(|e| format!("cannot read {}: {}", landscape_tsv.display(), e))?;
    let parsed = parse_landscape_tsv(&tsv_text)
        .map_err(|e| format!("parse error in landscape.tsv: {}", e))?;
    let json_payload = serde_json::to_string(&parsed)
        .map_err(|e| format!("json encode failure: {}", e))?;

    let title = meta.stem.clone()
        .unwrap_or_else(|| "camdl survey".to_string());
    let hash_short = meta.hash_short.clone();

    let html = build_html(&title, &hash_short, &json_payload, meta);
    let tmp = html_path.with_extension("html.tmp");
    std::fs::write(&tmp, html.as_bytes())
        .map_err(|e| format!("cannot write {}: {}", tmp.display(), e))?;
    std::fs::rename(&tmp, html_path)
        .map_err(|e| format!("cannot rename to {}: {}", html_path.display(), e))?;
    Ok(())
}

/// JSON schema posted into `<script type="application/json" id="landscape-data">`.
#[derive(serde::Serialize)]
pub(crate) struct LandscapeJson {
    /// Parameter names, matching the TSV's leading columns.
    estimated: Vec<String>,
    /// Whether each estimated param uses a log scale by default
    /// (mirrors the Python prototype's `use_log` heuristic: bounds
    /// span more than 2 decades and lower bound > 0).
    log_axis: Vec<bool>,
    /// Per-parameter bounds (lo, hi). Used by the JS to draw the
    /// LHS box and to set initial axis ranges.
    bounds: Vec<(f64, f64)>,
    /// Whether the TSV carries a `mean_ess` column (PF eval) or not
    /// (simulate eval). Drives the color-by dropdown options.
    has_mean_ess: bool,
    /// One row per LHS point. `params` aligns with `estimated`.
    rows: Vec<LandscapeRowJson>,
}

#[derive(serde::Serialize)]
struct LandscapeRowJson {
    point_id: usize,
    params: Vec<f64>,
    loglik: f64,
    loglik_se: f64,
    mean_ess: Option<f64>,
    n_replicates: usize,
}

pub(crate) fn parse_landscape_tsv(text: &str) -> Result<LandscapeJson, String> {
    // Skip leading comment lines.
    let mut lines = text.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header_line = lines.next().ok_or("empty TSV (no header)")?;
    let header: Vec<&str> = header_line.split('\t').collect();
    let n = header.len();
    // Columns are read by name, not position: the parameters are every
    // column before `loglik`, and anything the writer appends after the
    // columns read here (`loglik_type`, gh#280) is not this page's.
    let column = |name: &str| header.iter().position(|c| *c == name);
    let required = |name: &str| column(name)
        .ok_or_else(|| format!("no '{name}' column in header: {}", header.join(", ")));
    let loglik_idx = required("loglik")?;
    let loglik_se_idx = required("loglik_se")?;
    let nrep_idx = required("n_replicates")?;
    let point_idx = required("point_id")?;
    let mean_ess_idx = column("mean_ess");
    let has_mean_ess = mean_ess_idx.is_some();
    if loglik_idx == 0 {
        return Err("not enough param columns; need >=1 before 'loglik', got 0".to_string());
    }
    let estimated: Vec<String> = header[..loglik_idx].iter()
        .map(|s| s.to_string()).collect();

    let mut rows: Vec<LandscapeRowJson> = Vec::new();
    for (line_idx, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != n {
            return Err(format!(
                "row {}: expected {} columns, got {}", line_idx + 2, n, fields.len()));
        }
        let mut params = Vec::with_capacity(estimated.len());
        for field in &fields[..loglik_idx] {
            params.push(parse_f64_field(field)?);
        }
        let loglik = parse_f64_field(fields[loglik_idx])?;
        let loglik_se = parse_f64_field(fields[loglik_se_idx])?;
        let mean_ess = mean_ess_idx.map(|i| parse_f64_field(fields[i])).transpose()?;
        let n_replicates: usize = fields[nrep_idx].trim().parse()
            .map_err(|_| format!("row {}: invalid n_replicates '{}'", line_idx + 2, fields[nrep_idx]))?;
        let point_id: usize = fields[point_idx].trim().parse()
            .map_err(|_| format!("row {}: invalid point_id '{}'", line_idx + 2, fields[point_idx]))?;
        rows.push(LandscapeRowJson {
            point_id, params, loglik, loglik_se, mean_ess, n_replicates,
        });
    }

    // Bounds + log_axis come from the calling SurveyInputs but the
    // parser doesn't have those — populate from row min/max as a
    // fallback (the caller overwrites with the real bounds).
    let mut bounds: Vec<(f64, f64)> = vec![(f64::INFINITY, f64::NEG_INFINITY); estimated.len()];
    for r in &rows {
        for (i, &v) in r.params.iter().enumerate() {
            if v.is_finite() {
                bounds[i].0 = bounds[i].0.min(v);
                bounds[i].1 = bounds[i].1.max(v);
            }
        }
    }
    let log_axis: Vec<bool> = bounds.iter().map(|(lo, hi)| {
        *lo > 0.0 && hi.is_finite() && lo.is_finite() && (hi / lo) > 100.0
    }).collect();

    Ok(LandscapeJson { estimated, log_axis, bounds, has_mean_ess, rows })
}

fn parse_f64_field(s: &str) -> Result<f64, String> {
    let t = s.trim();
    match t {
        "NaN"  => Ok(f64::NAN),
        "Inf"  => Ok(f64::INFINITY),
        "-Inf" => Ok(f64::NEG_INFINITY),
        _      => t.parse::<f64>().map_err(|_| format!("cannot parse f64 '{}'", t)),
    }
}

/// Vanilla-JS pair-plot renderer using plotly.js. Reads the
/// `landscape-data` JSON block, builds a grid of subplots, supports
/// top-K slider, color-by dropdown, axis-scale toggle, hover, and
/// brushing as described in proposal §"Interactive controls (v1)".
const PAIRPLOT_JS: &str = include_str!("../vendored/pairplot.js");

const PAIRPLOT_CSS: &str = "
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; padding: 0; background: #fff; color: #222; }
#header { padding: 12px 18px; border-bottom: 1px solid #eee; }
#header h1 { margin: 0; font-size: 16px; font-weight: 600; }
#header .subtitle { margin-top: 4px; font-size: 12px; color: #888; }
#controls { padding: 8px 18px; border-bottom: 1px solid #eee;
  display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  font-size: 13px; }
#controls label { display: inline-flex; align-items: center; gap: 6px; }
#controls input[type=range] { width: 160px; }
/* Square plot container — pair-plot grid is rows = columns = nParams,
   so the figure displays right when the outer container is 1:1.
   max-width keeps the plot from filling a 4K monitor; margin auto
   centers it under the controls bar. CSS aspect-ratio is supported
   in every evergreen browser. */
#plot { width: 100%; max-width: 1400px; aspect-ratio: 1;
  padding: 8px; margin: 0 auto; }
.note { color: #666; font-size: 11px; margin-left: auto; }
";

fn build_html(
    title: &str,
    hash_short: &str,
    landscape_json: &str,
    meta: &LandscapeMeta,
) -> String {
    // Override the parsed bounds with the declared ones so the axis ranges
    // reflect the user-declared LHS box.
    let mut bounds_pairs: Vec<(String, (f64, f64))> = meta.estimated.iter()
        .map(|name| {
            let b = meta.bounds.get(name).copied().unwrap_or((0.0, 1.0));
            (name.clone(), b)
        }).collect();
    bounds_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let bounds_json = serde_json::to_string(
        &meta.estimated.iter()
            .map(|name| meta.bounds.get(name).copied().unwrap_or((f64::NAN, f64::NAN)))
            .collect::<Vec<_>>()
    ).unwrap_or_else(|_| "[]".into());

    format!(
"<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>camdl survey — {title}</title>
<style>{css}</style>
</head>
<body>
<div id=\"header\">
  <h1>camdl survey — {title}</h1>
  <div class=\"subtitle\">run hash: <code>{hash_short}</code> · {n_points} points · eval: {eval}</div>
</div>
<div id=\"controls\">
  <label>top-K cutoff
    <input type=\"range\" id=\"top-pct\" min=\"0.001\" max=\"0.5\" step=\"0.001\" value=\"0.05\">
    <span id=\"top-pct-display\">5.0%</span>
  </label>
  <label>color by
    <select id=\"color-by\">
      <option value=\"loglik\">loglik</option>
      <option value=\"loglik_se\">loglik_se</option>
      <option value=\"mean_ess\">mean_ess</option>
    </select>
  </label>
  <span class=\"note\">click an axis label to toggle log/linear · drag to brush</span>
</div>
<div id=\"plot\"></div>
<script>{plotly}</script>
<script type=\"application/json\" id=\"landscape-data\">{data}</script>
<script type=\"application/json\" id=\"landscape-bounds\">{bounds}</script>
<script>{pairplot}</script>
</body>
</html>
",
        title = html_escape(title),
        css = PAIRPLOT_CSS,
        hash_short = html_escape(hash_short),
        n_points = meta.n_points,
        eval = meta.eval_method.as_str(),
        plotly = PLOTLY_BUNDLE,
        data = landscape_json,
        bounds = bounds_json,
        pairplot = PAIRPLOT_JS,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_landscape_tsv_pfilter_columns() {
        let tsv = "\
# camdl survey landscape; run_hash=h
# eval=pfilter
beta\tgamma\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id
0.3\t0.15\t-123.4\t0.5\t180.0\t3\t0
0.1\t0.05\t-200.0\t0.6\t120.0\t3\t1
";
        let parsed = parse_landscape_tsv(tsv).unwrap();
        assert_eq!(parsed.estimated, vec!["beta", "gamma"]);
        assert!(parsed.has_mean_ess);
        assert_eq!(parsed.rows.len(), 2);
        assert_eq!(parsed.rows[0].point_id, 0);
        assert_eq!(parsed.rows[0].mean_ess, Some(180.0));
    }

    #[test]
    fn parse_landscape_tsv_simulate_columns() {
        let tsv = "\
# survey
# eval=simulate
beta\tloglik\tloglik_se\tn_replicates\tpoint_id
0.3\t-123.4\t0.0\t1\t0
";
        let parsed = parse_landscape_tsv(tsv).unwrap();
        assert_eq!(parsed.estimated, vec!["beta"]);
        assert!(!parsed.has_mean_ess);
        assert_eq!(parsed.rows.len(), 1);
        assert!(parsed.rows[0].mean_ess.is_none());
    }

    fn sample() -> LandscapeMeta {
        let mut bounds = HashMap::new();
        bounds.insert("beta".into(), (0.001, 1.0));
        LandscapeMeta {
            stem: Some("sir".into()),
            hash_short: "abc12345".into(),
            estimated: vec!["beta".into()],
            bounds,
            n_points: 10,
            eval_method: crate::run_meta::SurveyEvalMethod::Pfilter,
        }
    }

    #[test]
    fn build_html_contains_data_block() {
        let html = build_html("sir", "abc12345", "[]", &sample());
        assert!(html.contains(r#"<script type="application/json" id="landscape-data">"#));
        assert!(html.contains(r#"<script type="application/json" id="landscape-bounds">"#));
        assert!(html.contains("plotly"));
        assert!(html.contains("camdl survey"));
    }

    /// Byte-identical golden: the rendered HTML for a fixed (title, hash, data,
    /// inputs) is pinned by digest. This guards the `SurveyInputs` →
    /// `LandscapeMeta` decouple (gh#147 M3.3) — the field-by-field copy must
    /// reproduce the pair-plot byte-for-byte (title/hash params + the computed
    /// bounds/estimated/n_points/eval the renderer reads), not just plausibly.
    #[test]
    fn render_output_is_byte_stable() {
        let html = build_html("sir", "abc12345", "[]", &sample());
        let digest = runid::ContentHash::digest_bytes(html.as_bytes()).to_hex();
        assert_eq!(digest,
            "f29263f98fb3d491fea64399f013d12717a3d6f679a504c0bddff81a1c60ef9f",
            "rendered HTML digest changed — the decouple must be byte-identical");
    }
}
