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

/// Rows of the surprise table: the worst-scored observations a trace's
/// aggregates are read against.
pub const SURPRISE_ROWS: usize = 5;

/// The surprise table: the [`SURPRISE_ROWS`] worst-scored observations of a
/// trace, worst first, with their share of the elpd — printed wherever the
/// elpd is, because the elpd is a sum and a sum hides which terms carry it.
///
/// The Ebola case this exists for: an elpd of a few hundred nats over 103
/// days read as ordinary, and so did the mean CRPS and the PIT coverage.
/// One day — a cumulative count re-issued and floored to zero — carried a
/// log score of −26.6 on its own, a PIT of 0 and a filter ESS of 3, and the
/// only place that stood was one row of `prequential.tsv` nobody opened.
/// This table opens it.
///
/// Each row is a scored step, ranked by the joint log score; `share` is the
/// step's fraction of the elpd. On a multi-stream model the joint `y_obs`
/// is a cross-stream sum, so the row names the step's worst stream and
/// shows *that* stream's observed value, predictive band and log score —
/// the row a reader can look up in the data. `ess` is the filter's ESS at
/// the step: a low value beside a bad score says the predictive was drawn
/// from a handful of particles, which is the gh#685 reading.
///
/// Empty when the trace scored nothing.
pub fn surprise_table(trace: &sim::inference::prequential::PrequentialTrace) -> String {
    let worst = trace.worst_scored(SURPRISE_ROWS);
    if worst.is_empty() {
        return String::new();
    }
    let elpd = trace.elpd();
    let share = |ls: f64| -> String {
        if ls.is_finite() && elpd.is_finite() && elpd != 0.0 {
            format!("{:.1}%", 100.0 * ls / elpd)
        } else {
            "—".to_string()
        }
    };
    let num = |v: f64| -> String {
        if v.is_finite() {
            format!("{v:.1}")
        } else if v == f64::NEG_INFINITY {
            "-inf".to_string()
        } else {
            "NaN".to_string()
        }
    };
    // A band edge is an interpolated sample quantile — `843.85` on a count
    // stream — so it is rounded to what a reader compares `y_obs` against:
    // whole units from ten up, hundredths below (a proportion stream).
    let edge = |v: f64| -> String {
        if !v.is_finite() {
            num(v)
        } else if v.abs() >= 10.0 {
            crate::quantile::fmt_value(v.round())
        } else {
            crate::quantile::fmt_value((v * 100.0).round() / 100.0)
        }
    };
    let top_ls: f64 = worst.iter().map(|s| s.log_score).sum();
    let mut out = format!(
        "  worst-scored observations ({} of {} scored; together {} of elpd {}):\n",
        worst.len(),
        trace.n_scored(),
        share(top_ls),
        num(elpd),
    );
    out.push_str(&format!(
        "    {:>8} {:<14} {:>9} {:>17} {:>14} {:>7} {:>5} {:>8}\n",
        "t", "stream", "y_obs", "pred q05–q95", "log_score", "share", "pit", "ess"
    ));
    for s in &worst {
        // The row a reader can look up: the worst stream's own numbers when
        // the step has a breakdown, the joint otherwise.
        let (stream, y_obs, iv, row_ls) = match s.worst_stream() {
            Some(ss) => (ss.stream.as_str(), ss.y_obs, &ss.interval, ss.log_score),
            None => ("joint", s.y_obs, &s.interval, s.log_score),
        };
        let band = format!("[{}, {}]", edge(iv.q05), edge(iv.q95));
        // A per-stream score below the joint is shown as `joint (stream)`
        // so the ranking column stays the joint score every row shares.
        let ls_col = if s.per_stream.is_empty() {
            num(s.log_score)
        } else {
            format!("{} ({})", num(s.log_score), num(row_ls))
        };
        out.push_str(&format!(
            "    {:>8} {:<14} {:>9} {:>17} {:>14} {:>7} {:>5.2} {:>8.1}\n",
            crate::quantile::fmt_time(s.t),
            stream,
            crate::quantile::fmt_value(y_obs),
            band,
            ls_col,
            share(s.log_score),
            s.pit,
            s.ess,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::prequential::{
        Conditioning, PredInterval, PrequentialStep, PrequentialTrace, Provenance, StreamScore,
    };

    fn step(t: f64, y: f64, ls: f64, ess: f64, streams: &[(&str, f64, f64)]) -> PrequentialStep {
        PrequentialStep {
            t, y_obs: y, y_pred_samples: vec![], log_score: ls, crps: 0.0, pit: 0.5, ess,
            interval: PredInterval { q05: 1.0, q25: 2.0, q50: 3.0, q75: 4.0, q95: 5.0 },
            per_stream: streams.iter().map(|(n, y, ls)| StreamScore {
                stream: n.to_string(), y_obs: *y, y_pred_samples: vec![], log_score: *ls,
                crps: 0.0, pit: 0.5,
                interval: PredInterval { q05: 10.0, q25: 0.0, q50: 0.0, q75: 0.0, q95: 20.0 },
            }).collect(),
        }
    }

    fn trace(steps: Vec<PrequentialStep>) -> PrequentialTrace {
        PrequentialTrace {
            schema_version: 3, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample, steps, warnings: vec![],
            score_from: None, pit_randomization_seed: None,
        }
    }

    #[test]
    fn the_table_ranks_worst_first_and_shares_sum_against_the_elpd() {
        // elpd = -40; the -30 step is 75% of it and comes first.
        let t = trace(vec![
            step(7.0, 3.0, -4.0, 100.0, &[]),
            step(14.0, 0.0, -30.0, 3.0, &[]),
            step(21.0, 2.0, -6.0, 90.0, &[]),
        ]);
        let s = surprise_table(&t);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0],
            "  worst-scored observations (3 of 3 scored; together 100.0% of elpd -40.0):");
        assert!(lines[2].starts_with("          14 joint"), "{}", lines[2]);
        assert!(lines[2].contains("-30.0"), "{}", lines[2]);
        assert!(lines[2].contains("75.0%"), "{}", lines[2]);
        assert!(lines[2].trim_end().ends_with("3.0"), "ess is the last column: {}", lines[2]);
        assert!(lines[3].starts_with("          21 joint"), "{}", lines[3]);
        assert!(lines[4].starts_with("           7 joint"), "{}", lines[4]);
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn the_table_is_capped_and_names_the_worst_stream() {
        let mut steps: Vec<PrequentialStep> = (0..8)
            .map(|i| step(i as f64, 1.0, -1.0 - i as f64, 50.0, &[("a", 1.0, -0.5), ("b", 0.0, -0.5 - i as f64)]))
            .collect();
        steps[3].log_score = f64::NEG_INFINITY;
        let t = trace(steps);
        let s = surprise_table(&t);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2 + SURPRISE_ROWS, "header, columns, five rows:\n{s}");
        assert!(lines[0].starts_with("  worst-scored observations (5 of 8 scored; together — of elpd -inf)"),
            "{}", lines[0]);
        // The -inf step ranks first, shows the worst stream's own y_obs and
        // band, the joint score with the stream score in parentheses, and no
        // share (it is not a finite fraction of anything).
        assert!(lines[2].starts_with("           3 b"), "{}", lines[2]);
        assert!(lines[2].contains("[10, 20]"), "the stream's band, not the joint's: {}", lines[2]);
        assert!(lines[2].contains("-inf (-3.5)"), "{}", lines[2]);
        assert!(lines[2].contains(" — "), "{}", lines[2]);
        // Then the finite ones, worst first.
        assert!(lines[3].starts_with("           7 b"), "{}", lines[3]);
    }

    #[test]
    fn band_edges_round_to_units_from_ten_and_hundredths_below() {
        let mut big = step(1.0, 1088.0, -15.3, 100.0, &[]);
        big.interval = PredInterval { q05: 261.0, q25: 0.0, q50: 0.0, q75: 0.0, q95: 843.85 };
        let mut small = step(2.0, 0.05, -2.0, 100.0, &[]);
        small.interval = PredInterval { q05: 0.0234, q25: 0.0, q50: 0.0, q75: 0.0, q95: 9.996 };
        let s = surprise_table(&trace(vec![big, small]));
        let lines: Vec<&str> = s.lines().collect();
        assert!(lines[2].contains("[261, 844]"), "{}", lines[2]);
        assert!(lines[3].contains("[0.02, 10]"), "{}", lines[3]);
    }

    #[test]
    fn an_empty_trace_prints_nothing() {
        assert_eq!(surprise_table(&trace(vec![])), "");
    }
}
