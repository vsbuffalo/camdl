//! The one writer for prequential trace artifacts (gh#650).
//!
//! `fit run` (PFilter method) and `camdl pfilter --save-prequential` both
//! persist a `PrequentialTrace` as `{stem}.tsv` + `{stem}.streams.tsv` +
//! `{stem}.json` under one filename convention. The tidy TSV schema (gh#269)
//! landed first in the pfilter writer only, leaving `fit run` emitting a
//! narrow v1 header under the same filename — the same "the fix landed at
//! one site" shape as gh#648/gh#268. Both callers route here so the schemas
//! cannot drift again.

use std::io::Write;
use std::path::{Path, PathBuf};

/// The per-stream sibling of a joint TSV path: `.streams` inserted before the
/// final extension (`scores.tsv` → `scores.streams.tsv`), or appended when the
/// path has none (`scores` → `scores.streams`). `prequential.tsv` and
/// `compare --pointwise` both name their per-stream file this way.
pub fn streams_sibling(path: &Path) -> PathBuf {
    match (path.file_stem(), path.extension()) {
        (Some(stem), Some(ext)) => {
            let mut name = stem.to_os_string();
            name.push(".streams.");
            name.push(ext);
            path.with_file_name(name)
        }
        _ => {
            let mut name = path.as_os_str().to_os_string();
            name.push(".streams");
            PathBuf::from(name)
        }
    }
}

/// Write a `PrequentialTrace` to `{stem}.tsv` (the joint score per step),
/// `{stem}.streams.tsv` (each stream's own scores) and `{stem}.json` (the
/// full typed trace, serde). Downstream tools join on `stem` to avoid
/// re-running the PF; `camdl compare` consumes the JSON.
///
/// `{stem}.tsv` columns: `t  log_score  ess` — one row per scored step, the
/// log of the joint predictive density of every stream observed there (its
/// sum is the elpd) and the filter's ESS.
///
/// `{stem}.streams.tsv` columns: `t  stream  y_obs  y_pred_q05..q95
/// log_score  crps  pit` — one row per step × scheduled, non-hole stream
/// (gh#269). The y_pred_q* columns are the stream's plot-ready predictive
/// interval (median + 50%/90% bands) for the forecast-vs-observed panel; they
/// survive `--no-save-samples` (computed in `build_trace` before samples are
/// cleared). A stream's `log_score` is its own; the stream rows at a step do
/// not sum to the joint row. CRPS and PIT exist only here: across streams
/// there is no one observed quantity to score them against.
pub fn write_prequential_outputs(
    stem: &str,
    trace: &sim::inference::prequential::PrequentialTrace,
) -> std::io::Result<()> {
    let tsv_path = format!("{}.tsv", stem);
    let streams_path = streams_sibling(Path::new(&tsv_path));
    let json_path = format!("{}.json", stem);
    let mut tsv = std::io::BufWriter::new(std::fs::File::create(&tsv_path)?);
    writeln!(tsv, "t\tlog_score\tess")?;
    for s in &trace.steps {
        writeln!(tsv, "{}\t{:.6}\t{:.2}", s.t, s.log_score, s.ess)?;
    }
    drop(tsv);
    let mut tsv = std::io::BufWriter::new(std::fs::File::create(&streams_path)?);
    writeln!(tsv, "t\tstream\ty_obs\ty_pred_q05\ty_pred_q25\ty_pred_q50\ty_pred_q75\ty_pred_q95\tlog_score\tcrps\tpit")?;
    for s in &trace.steps {
        for ss in &s.per_stream {
            let iv = &ss.interval;
            writeln!(tsv, "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                s.t, ss.stream, ss.y_obs, iv.q05, iv.q25, iv.q50, iv.q75, iv.q95,
                ss.log_score, ss.crps, ss.pit)?;
        }
    }
    drop(tsv);
    let json = serde_json::to_string_pretty(trace)
        .map_err(std::io::Error::other)?;
    std::fs::write(&json_path, json)?;
    Ok(())
}

/// The one-line summary `camdl pfilter --save-prequential` and `fit run`
/// print beside the trace: the elpd, then each stream's mean CRPS and PIT
/// 90% coverage. A single-stream trace reads `elpd=…, mean_crps=…, PIT 90%
/// cov=…`; a multi-stream one names each stream, because no CRPS or coverage
/// combines streams. A trace with no per-stream breakdown has neither.
pub fn summary_line(trace: &sim::inference::prequential::PrequentialTrace) -> String {
    let elpd = format!("elpd={:.2}", trace.elpd());
    let figures = |s: &str| format!("mean_crps={:.3}, PIT 90% cov={:.2}",
        trace.mean_crps(s).value, trace.pit_coverage(s, 0.90).value);
    match trace.streams().as_slice() {
        [] => format!("{elpd} (no per-stream breakdown: CRPS and PIT coverage not available)"),
        [only] => format!("{elpd}, {}", figures(only)),
        many => format!("{elpd}; per stream: {}", many.iter()
            .map(|s| format!("{s} {}", figures(s))).collect::<Vec<_>>().join("; ")),
    }
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
/// step's fraction of the elpd. A step has no observed value of its own, so
/// the row names the step's worst stream and shows *that* stream's observed
/// value, predictive band, log score and PIT — the row a reader can look up
/// in the data. `ess` is the filter's ESS at the step: a low value beside a
/// bad score says the predictive was drawn from a handful of particles, which
/// is the gh#685 reading. A step with no per-stream breakdown (a schema-1
/// trace) shows `—` for the stream's columns.
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
        // The row a reader can look up: the worst stream's own numbers. The
        // log score column is `joint (stream)`, so the ranking column stays
        // the joint score every row shares.
        let (stream, y_obs, band, ls_col, pit) = match s.worst_stream() {
            Some(ss) => (
                ss.stream.as_str(),
                crate::quantile::fmt_value(ss.y_obs),
                format!("[{}, {}]", edge(ss.interval.q05), edge(ss.interval.q95)),
                format!("{} ({})", num(s.log_score), num(ss.log_score)),
                format!("{:.2}", ss.pit),
            ),
            None => ("—", "—".to_string(), "—".to_string(), num(s.log_score), "—".to_string()),
        };
        out.push_str(&format!(
            "    {:>8} {:<14} {:>9} {:>17} {:>14} {:>7} {:>5} {:>8.1}\n",
            crate::quantile::fmt_time(s.t),
            stream,
            y_obs,
            band,
            ls_col,
            share(s.log_score),
            pit,
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

    /// A step with the given joint score and ESS and these streams, each
    /// `(name, y_obs, log_score)` with the band [10, 20] and PIT 0.5.
    fn step(t: f64, ls: f64, ess: f64, streams: &[(&str, f64, f64)]) -> PrequentialStep {
        PrequentialStep {
            t, log_score: ls, ess,
            per_stream: streams.iter().map(|(n, y, ls)| StreamScore {
                stream: n.to_string(), y_obs: *y, y_pred_samples: vec![], log_score: *ls,
                crps: 0.25, pit: 0.5,
                interval: PredInterval { q05: 10.0, q25: 0.0, q50: 0.0, q75: 0.0, q95: 20.0 },
                coverage: Default::default(),
            }).collect(),
        }
    }

    fn trace(steps: Vec<PrequentialStep>) -> PrequentialTrace {
        PrequentialTrace {
            schema_version: 3, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample, steps, warnings: vec![],
            score_from: None, pit_randomization_seed: None,
            derivation: None,
        }
    }

    #[test]
    fn the_table_ranks_worst_first_and_shares_sum_against_the_elpd() {
        // elpd = -40; the -30 step is 75% of it and comes first.
        let t = trace(vec![
            step(7.0, -4.0, 100.0, &[("cases", 3.0, -4.0)]),
            step(14.0, -30.0, 3.0, &[("cases", 0.0, -30.0)]),
            step(21.0, -6.0, 90.0, &[("cases", 2.0, -6.0)]),
        ]);
        let s = surprise_table(&t);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0],
            "  worst-scored observations (3 of 3 scored; together 100.0% of elpd -40.0):");
        assert!(lines[2].starts_with("          14 cases"), "{}", lines[2]);
        assert!(lines[2].contains("-30.0"), "{}", lines[2]);
        assert!(lines[2].contains("75.0%"), "{}", lines[2]);
        assert!(lines[2].contains(" 0.50 "), "the stream's PIT: {}", lines[2]);
        assert!(lines[2].trim_end().ends_with("3.0"), "ess is the last column: {}", lines[2]);
        assert!(lines[3].starts_with("          21 cases"), "{}", lines[3]);
        assert!(lines[4].starts_with("           7 cases"), "{}", lines[4]);
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn the_table_is_capped_and_names_the_worst_stream() {
        let mut steps: Vec<PrequentialStep> = (0..8)
            .map(|i| step(i as f64, -1.0 - i as f64, 50.0, &[("a", 1.0, -0.5), ("b", 0.0, -0.5 - i as f64)]))
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
        assert!(lines[2].contains("[10, 20]"), "the stream's band: {}", lines[2]);
        assert!(lines[2].contains("-inf (-3.5)"), "{}", lines[2]);
        assert!(lines[2].contains(" — "), "{}", lines[2]);
        // Then the finite ones, worst first.
        assert!(lines[3].starts_with("           7 b"), "{}", lines[3]);
    }

    #[test]
    fn band_edges_round_to_units_from_ten_and_hundredths_below() {
        let mut big = step(1.0, -15.3, 100.0, &[("cases", 1088.0, -15.3)]);
        big.per_stream[0].interval = PredInterval { q05: 261.0, q25: 0.0, q50: 0.0, q75: 0.0, q95: 843.85 };
        let mut small = step(2.0, -2.0, 100.0, &[("prop", 0.05, -2.0)]);
        small.per_stream[0].interval = PredInterval { q05: 0.0234, q25: 0.0, q50: 0.0, q75: 0.0, q95: 9.996 };
        let s = surprise_table(&trace(vec![big, small]));
        let lines: Vec<&str> = s.lines().collect();
        assert!(lines[2].contains("[261, 844]"), "{}", lines[2]);
        assert!(lines[3].contains("[0.02, 10]"), "{}", lines[3]);
    }

    /// A step with no per-stream breakdown has no observed value, band or
    /// PIT to show: its row names the joint score and nothing else.
    #[test]
    fn a_step_without_a_stream_breakdown_shows_no_stream_figures() {
        let s = surprise_table(&trace(vec![step(7.0, -4.0, 100.0, &[])]));
        let row = s.lines().nth(2).unwrap();
        assert!(row.starts_with("           7 —"), "{row}");
        assert!(row.contains("-4.0") && !row.contains("0.50"), "{row}");
    }

    #[test]
    fn an_empty_trace_prints_nothing() {
        assert_eq!(surprise_table(&trace(vec![])), "");
    }

    #[test]
    fn the_streams_sibling_inserts_streams_before_the_extension() {
        assert_eq!(streams_sibling(Path::new("out/preq.tsv")), Path::new("out/preq.streams.tsv"));
        assert_eq!(streams_sibling(Path::new("out/preq")), Path::new("out/preq.streams"));
        assert_eq!(streams_sibling(Path::new("a.b.tsv")), Path::new("a.b.streams.tsv"));
    }

    /// D14: `{stem}.tsv` holds one row per step with the joint score and the
    /// ESS; `{stem}.streams.tsv` one row per step × stream with that stream's
    /// observed value, band and scores. No row combines streams.
    #[test]
    fn the_trace_is_written_as_a_joint_file_and_a_per_stream_file() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("preq");
        let t = trace(vec![
            step(7.0, -4.0, 100.0, &[("cases", 3.0, -2.5), ("deaths", 1.0, -1.0)]),
            step(14.0, -3.0, 80.0, &[("cases", 2.0, -3.0)]),
        ]);
        write_prequential_outputs(&stem.to_string_lossy(), &t).unwrap();
        let joint = std::fs::read_to_string(dir.path().join("preq.tsv")).unwrap();
        assert_eq!(joint, "t\tlog_score\tess\n7\t-4.000000\t100.00\n14\t-3.000000\t80.00\n");
        let streams = std::fs::read_to_string(dir.path().join("preq.streams.tsv")).unwrap();
        let lines: Vec<&str> = streams.lines().collect();
        assert_eq!(lines[0], "t\tstream\ty_obs\ty_pred_q05\ty_pred_q25\ty_pred_q50\ty_pred_q75\ty_pred_q95\tlog_score\tcrps\tpit");
        assert_eq!(lines.len(), 4, "one row per step × stream: {streams}");
        assert_eq!(lines[2],
            "7\tdeaths\t1\t10.000000\t0.000000\t0.000000\t0.000000\t20.000000\t-1.000000\t0.250000\t0.500000");
        assert!(!streams.contains("joint"), "{streams}");
        let back: PrequentialTrace = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("preq.json")).unwrap()).unwrap();
        assert_eq!(back.n_scored(), 2);
    }
}
