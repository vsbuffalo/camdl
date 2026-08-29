//! The one writer for prequential trace artifacts (gh#650).
//!
//! `fit run` (PFilter stage) and `camdl pfilter --save-prequential` both
//! persist a `PrequentialTrace` as `{stem}.tsv` + `{stem}.json` under one
//! filename convention. The tidy/long TSV schema (gh#269) landed first in
//! the pfilter writer only, leaving `fit run` emitting a narrow v1 header
//! under the same filename — the same "the fix landed at one site" shape as
//! gh#648/gh#268. Both callers route here so the schemas cannot drift again.

use std::io::Write;

/// Write a `PrequentialTrace` to `{stem}.tsv` (tidy/long) + `{stem}.json`
/// (full typed trace, serde). Downstream tools join on `stem` to avoid
/// re-running the PF; `camdl compare` consumes the JSON.
///
/// gh#269: the TSV is tidy/long with a `stream` column. Each step writes a
/// `joint` row (the cross-stream summary scores, `stream="joint"`) FOLLOWED by
/// one row per scheduled, non-hole stream (`stream=<district>`, its own
/// `y_obs`/`log_score`/`crps`/`pit`). The `ess` column repeats the step's joint
/// ESS on every row (ESS is a filter-wide quantity, not per stream). The JSON
/// carries the full nested structure (per-step `per_stream` array) for tooling.
///
/// Columns: `t  stream  y_obs  y_pred_q05..q95  log_score  crps  pit  ess`.
/// The y_pred_q* columns are the plot-ready predictive interval (median +
/// 50%/90% bands) for the forecast-vs-observed panel; they survive
/// `--no-save-samples` (computed in `build_trace` before samples are cleared).
pub fn write_prequential_outputs(
    stem: &str,
    trace: &sim::inference::prequential::PrequentialTrace,
) -> std::io::Result<()> {
    let tsv_path = format!("{}.tsv", stem);
    let json_path = format!("{}.json", stem);
    let mut tsv = std::io::BufWriter::new(std::fs::File::create(&tsv_path)?);
    writeln!(tsv, "t\tstream\ty_obs\ty_pred_q05\ty_pred_q25\ty_pred_q50\ty_pred_q75\ty_pred_q95\tlog_score\tcrps\tpit\tess")?;
    for s in &trace.steps {
        let iv = &s.interval;
        // Joint summary row.
        writeln!(tsv, "{}\tjoint\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.2}",
            s.t, s.y_obs, iv.q05, iv.q25, iv.q50, iv.q75, iv.q95,
            s.log_score, s.crps, s.pit, s.ess)?;
        // Per-stream rows (ess repeats the joint ESS — filter-wide quantity).
        for ss in &s.per_stream {
            let iv = &ss.interval;
            writeln!(tsv, "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.2}",
                s.t, ss.stream, ss.y_obs, iv.q05, iv.q25, iv.q50, iv.q75, iv.q95,
                ss.log_score, ss.crps, ss.pit, s.ess)?;
        }
    }
    drop(tsv);
    let json = serde_json::to_string_pretty(trace)
        .map_err(std::io::Error::other)?;
    std::fs::write(&json_path, json)?;
    Ok(())
}
