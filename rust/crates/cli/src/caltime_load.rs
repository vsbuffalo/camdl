//! Dated-data loader support — the runtime half of the calendar-time
//! boundary translator (2026-05-22 proposal, phase 2).
//!
//! This is the *only* place a `--data` time column becomes internal `f64`
//! time. It sits on top of `ir::caltime` (the shared conversion) and adds:
//!
//!   * **whole-column type detection**: every cell numeric → numeric column
//!     (today's behaviour, no `origin` needed); every cell an ISO date →
//!     dated column (converted via `caltime` using the model's `origin` +
//!     `time_unit`); mixed → hard error naming the offending row.
//!   * a `--time-format numeric|date` override, honoured *before* detection.
//!   * the **distinct-substep check** (proposal §5.4): after conversion,
//!     distinct observation times must map to distinct integrator substeps
//!     under the run's `dt`; a collision is a hard error naming both rows.
//!   * an **off-grid warning** when converted times don't land on the `dt`
//!     grid (snapped within `dt` by the integrator — most common under
//!     `'months`/`'years`, which are never integer-aligned).
//!
//! Numeric/indexed data takes the all-numeric branch and is byte-identical
//! to the pre-date behaviour — the §9.0 backward-compatibility wall.

use sim::time::interval_steps;

/// Explicit `--time-format` override for a data file's time column.
/// `Auto` (the default) runs whole-column detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeFormat {
    /// Detect numeric-vs-date over the whole column.
    #[default]
    Auto,
    /// Force `f64` parsing; a date cell is then a parse error.
    Numeric,
    /// Force ISO-date parsing; requires `origin`.
    Date,
}

impl std::str::FromStr for TimeFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(TimeFormat::Auto),
            "numeric" => Ok(TimeFormat::Numeric),
            "date" => Ok(TimeFormat::Date),
            // `internal-days` is the proposal-named opt-in for the
            // legitimate "numeric time column under an anchored model,
            // values are day-offsets from origin" pattern (proposal
            // §3.6). Behaviourally identical to `numeric` but signals
            // intent and suppresses the W326 warning. See
            // docs/dev/proposals/2026-05-22-typed-time-and-dsl-ergonomics.md.
            "internal-days" => Ok(TimeFormat::Numeric),
            other => Err(format!(
                "unknown --time-format '{other}' \
                 (expected numeric|date|internal-days|auto)"
            )),
        }
    }
}

/// Calendar context + integrator step needed to interpret a time column.
/// Threaded from the compiled model (`origin`, `time_unit`) and the run (`dt`).
#[derive(Debug, Clone)]
pub struct TimeOpts<'a> {
    /// I/O calendar anchor (the date mapped to internal `t = 0`). `None`
    /// when the model declares no origin — dated columns then hard-error.
    pub origin: Option<&'a str>,
    /// The model's `time_unit` (`days`/`weeks`/`months`/`years`).
    pub time_unit: &'a str,
    /// Integrator step, for the distinct-substep + off-grid checks. The
    /// fit window starts at `t_start`.
    pub dt: f64,
    /// Integration start time, for the distinct-substep check
    /// (`interval_steps(t_start, obs, dt)` must be injective).
    pub t_start: f64,
    /// Explicit format override; `Auto` runs detection.
    pub format: TimeFormat,
}

/// Whether a cell parses as a bare `f64`.
fn is_numeric_cell(cell: &str) -> bool {
    cell.trim().parse::<f64>().is_ok()
}

/// Whether a cell parses as an ISO date (zone discarded, civil date).
fn is_date_cell(cell: &str) -> bool {
    ir::caltime::parse_iso_date(cell.trim()).is_ok()
}

/// Detected column kind after whole-column scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColKind {
    Numeric,
    Date,
}

/// Detect the column kind over *all* cells (proposal §6.3). A cell that is
/// both numeric and date-shaped cannot occur (a date has dashes; a bare
/// number never parses as a date), so the two predicates partition the
/// non-degenerate cells. `row_offset` is the TSV line number of `cells[0]`
/// (typically 2: line 1 is the header), used for error messages.
fn detect_kind(cells: &[&str], row_offset: usize) -> Result<ColKind, String> {
    let mut first_numeric: Option<usize> = None;
    let mut first_date: Option<usize> = None;
    for (i, &c) in cells.iter().enumerate() {
        let c = c.trim();
        if c.is_empty() {
            continue;
        }
        if is_numeric_cell(c) {
            first_numeric.get_or_insert(i);
        } else if is_date_cell(c) {
            first_date.get_or_insert(i);
        } else {
            // Neither numeric nor a valid date: surface the date parse
            // error (the more informative of the two) at this row.
            let why = ir::caltime::parse_iso_date(c)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unparseable".to_string());
            return Err(format!(
                "line {}: time cell '{}' is neither a number nor an ISO date ({})",
                row_offset + i,
                c,
                why
            ));
        }
    }
    match (first_numeric, first_date) {
        (Some(_), None) => Ok(ColKind::Numeric),
        (None, Some(_)) => Ok(ColKind::Date),
        (None, None) => Ok(ColKind::Numeric), // all-empty: harmless, no rows
        (Some(n), Some(d)) => {
            // Mixed column — name both offending rows so the user can find
            // the inconsistency.
            let (num_row, date_row) = (row_offset + n, row_offset + d);
            Err(format!(
                "mixed time column: numeric cell '{}' at line {} and date cell '{}' at line {}. \
                 A time column must be all-numeric or all-dates; use --time-format to force one.",
                cells[n].trim(), num_row, cells[d].trim(), date_row
            ))
        }
    }
}

/// Convert a column of raw time cells to internal `f64` time, applying
/// detection / the `--time-format` override / date conversion.
///
/// `row_offset` is the TSV line of `cells[0]` (for error messages).
/// On success returns one `f64` per cell, in order.
pub fn convert_time_column(
    cells: &[&str],
    opts: &TimeOpts,
    row_offset: usize,
) -> Result<Vec<f64>, String> {
    let kind = match opts.format {
        TimeFormat::Numeric => ColKind::Numeric,
        TimeFormat::Date => ColKind::Date,
        TimeFormat::Auto => detect_kind(cells, row_offset)?,
    };

    // W326 (proposal §3.6, Rule 6): warn when an Auto-detected
    // numeric time column is loaded under an anchored model. The
    // legitimate idiom (seed-timing fits with `time` column carrying
    // day-offsets from origin — `covid_wa_daily.tsv`,
    // `covid_wa_growth.tsv`) opts in with `--time-format
    // internal-days` (which maps to `TimeFormat::Numeric` and
    // therefore doesn't take the `Auto` branch). The load proceeds
    // normally; this is informational, not blocking.
    if opts.format == TimeFormat::Auto && kind == ColKind::Numeric && opts.origin.is_some()
    {
        // Only fire if there's actually at least one non-empty cell —
        // an empty file shouldn't warn.
        let any_cell = cells.iter().any(|c| !c.trim().is_empty());
        if any_cell {
            eprintln!(
                "[warn W326] --data time column is numeric and the model \
                 declares `origin = date(\"{}\")` — values are interpreted \
                 as internal-time units from origin. \
                 If that's intentional, pass `--time-format internal-days` \
                 to suppress this warning; if you meant calendar dates, \
                 switch the column to ISO YYYY-MM-DD form.",
                opts.origin.unwrap_or("…"),
            );
        }
    }

    match kind {
        ColKind::Numeric => {
            let mut out = Vec::with_capacity(cells.len());
            for (i, &c) in cells.iter().enumerate() {
                let v: f64 = c.trim().parse().map_err(|_| {
                    format!(
                        "line {}: cannot parse time '{}' as a number{}",
                        row_offset + i,
                        c.trim(),
                        if opts.format == TimeFormat::Numeric && is_date_cell(c) {
                            " (--time-format numeric forbids date cells)"
                        } else {
                            ""
                        }
                    )
                })?;
                // Rust's f64 parser accepts "NaN"/"inf"/"-inf" — reject them:
                // a non-finite time is not a valid observation instant (and
                // would later panic the union-axis sort in `bind`).
                if !v.is_finite() {
                    return Err(format!(
                        "line {}: time '{}' is not a finite number — NaN and \
                         infinities are not valid observation times. Fix or \
                         remove the row.",
                        row_offset + i,
                        c.trim(),
                    ));
                }
                out.push(v);
            }
            Ok(out)
        }
        ColKind::Date => {
            let origin = opts.origin.ok_or_else(|| {
                "data has dated time cells but the model declares no `origin`. \
                 Add `origin = date(\"YYYY-MM-DD\")` to the model, or supply numeric times."
                    .to_string()
            })?;
            let mut out = Vec::with_capacity(cells.len());
            for (i, &c) in cells.iter().enumerate() {
                let t = ir::caltime::date_to_internal(origin, c.trim(), opts.time_unit)
                    .map_err(|e| format!("line {}: {}", row_offset + i, e))?;
                out.push(t);
            }
            Ok(out)
        }
    }
}

/// After conversion, enforce the distinct-substep invariant (proposal §5.4)
/// and emit the off-grid warning. `times` is the converted internal-time
/// vector; `rows` the corresponding TSV line numbers (for error messages).
///
/// The distinct-substep check protects every unit: two *distinct* observation
/// times that collapse onto the same integrator substep would silently merge
/// observations. Identical times (multi-stream obs at the same point) are
/// allowed and skipped.
pub fn check_substeps_and_grid(
    times: &[f64],
    rows: &[usize],
    opts: &TimeOpts,
    was_dated: bool,
) -> Result<(), String> {
    if opts.dt <= 0.0 || !opts.dt.is_finite() {
        return Ok(()); // dt validity is the integrator's contract, not ours.
    }
    // Distinct-substep check: map each obs to its substep relative to
    // t_start, then assert distinct times → distinct substeps.
    // `t_start = -inf` (numeric_only) disables it (no fit window).
    if opts.t_start.is_finite() {
        // step index per row (only for obs ≥ t_start; an earlier obs is the
        // §5.5 underflow landmine, handled by the integration-window spec,
        // not here).
        use std::collections::HashMap;
        let mut step_of: HashMap<i64, (usize, f64)> = HashMap::new();
        for (k, &t) in times.iter().enumerate() {
            if t < opts.t_start {
                continue;
            }
            let step = interval_steps(opts.t_start, t, opts.dt) as i64;
            if let Some(&(prev_k, prev_t)) = step_of.get(&step) {
                // A collision is only a problem if the *times* differ.
                if (prev_t - t).abs() > 1e-12 {
                    return Err(format!(
                        "observations at t={} (line {}) and t={} (line {}) collapse onto the \
                         same integrator substep under dt={}. Distinct observation times must \
                         map to distinct substeps; reduce --dt below the observation spacing.",
                        prev_t, rows[prev_k], t, rows[k], opts.dt
                    ));
                }
            } else {
                step_of.insert(step, (k, t));
            }
        }
    }

    // Off-grid warning: a converted time not on the dt grid is snapped
    // within dt by the integrator. Warn (most useful under months/years,
    // which are never integer-aligned). Only emitted for dated columns —
    // numeric off-grid times are the user's own choice, unchanged behaviour.
    if was_dated {
        let off_grid = times.iter().any(|&t| {
            let r = (t / opts.dt).round() * opts.dt;
            (t - r).abs() > 1e-9
        });
        if off_grid {
            eprintln!(
                "[warn] some converted observation times don't align to dt={} \
                 (snapped within dt). Common under time_unit = '{}'.",
                opts.dt, opts.time_unit
            );
        }
        if opts.time_unit == "months" || opts.time_unit == "years" {
            eprintln!(
                "[warn] dated data under time_unit = '{}': months/years are *average* \
                 calendar lengths, so monthly/yearly dates land off the integer grid. \
                 The integrator snaps within dt; verify this is intended.",
                opts.time_unit
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_days(origin: Option<&'static str>) -> TimeOpts<'static> {
        TimeOpts {
            origin,
            time_unit: "days",
            dt: 1.0,
            t_start: 0.0,
            format: TimeFormat::Auto,
        }
    }

    #[test]
    fn all_numeric_takes_numeric_path() {
        let cells = ["0", "1", "2", "7", "14"];
        let o = opts_days(None);
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![0.0, 1.0, 2.0, 7.0, 14.0]);
    }

    #[test]
    fn fractional_numeric_unchanged() {
        let cells = ["0.0", "0.5", "1.0"];
        let o = opts_days(None);
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn non_finite_numeric_time_is_a_located_error() {
        // Rust's f64 parser accepts these as Ok(non-finite); they must be
        // rejected with a located message (not flow downstream to panic the
        // union-axis sort in bind).
        let o = opts_days(None);
        for (cells, tok) in [
            (["0", "NaN", "2"], "NaN"),
            (["0", "inf", "2"], "inf"),
            (["0", "-inf", "2"], "-inf"),
        ] {
            let err = convert_time_column(&cells, &o, 2).unwrap_err();
            assert!(err.contains("finite"), "must name the rule: {err}");
            assert!(err.contains("line 3"), "must locate the row: {err}");
            assert!(err.contains(tok), "must echo the token: {err}");
        }
    }

    #[test]
    fn all_dates_convert_against_origin() {
        let cells = ["2020-03-01", "2020-03-08", "2020-03-15"];
        let o = opts_days(Some("2020-03-01"));
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![0.0, 7.0, 14.0]);
    }

    #[test]
    fn dated_without_origin_errors() {
        let cells = ["2020-03-01", "2020-03-08"];
        let o = opts_days(None);
        let err = convert_time_column(&cells, &o, 2).unwrap_err();
        assert!(err.contains("origin"), "{err}");
    }

    #[test]
    fn mixed_column_errors_naming_both_rows() {
        let cells = ["0", "1", "2020-03-15"];
        let o = opts_days(Some("2020-01-01"));
        let err = convert_time_column(&cells, &o, 2).unwrap_err();
        assert!(err.contains("mixed"), "{err}");
        assert!(err.contains("line 2"), "should name numeric row: {err}");
        assert!(err.contains("line 4"), "should name date row: {err}");
    }

    #[test]
    fn time_format_numeric_forbids_dates() {
        let cells = ["0", "2020-03-15"];
        let mut o = opts_days(Some("2020-01-01"));
        o.format = TimeFormat::Numeric;
        let err = convert_time_column(&cells, &o, 2).unwrap_err();
        assert!(err.contains("forbids date"), "{err}");
    }

    #[test]
    fn time_format_date_forces_conversion() {
        // A column that *looks* numeric but is forced to date would fail to
        // parse — verify the override path engages for real dates.
        let cells = ["2020-03-01", "2020-03-02"];
        let mut o = opts_days(Some("2020-03-01"));
        o.format = TimeFormat::Date;
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![0.0, 1.0]);
    }

    #[test]
    fn dated_byte_identical_to_hand_converted() {
        // The core promise: a dated TSV column yields the same internal-time
        // vector as the user's old fetch-script day-numbers.
        let origin = "2020-02-28";
        let dates = ["2020-02-28", "2020-03-06", "2020-03-13", "2020-01-21"];
        let o = opts_days(Some(origin));
        let from_dates = convert_time_column(&dates, &o, 2).unwrap();
        // hand-converted day numbers (rata_die delta):
        let hand: Vec<f64> = dates
            .iter()
            .map(|d| {
                let (oy, om, od) = ir::caltime::parse_iso_date(origin).unwrap();
                let (y, m, dd) = ir::caltime::parse_iso_date(d).unwrap();
                (ir::caltime::rata_die(y, m, dd) - ir::caltime::rata_die(oy, om, od)) as f64
            })
            .collect();
        assert_eq!(from_dates, hand);
    }

    #[test]
    fn distinct_substep_collision_errors() {
        // Two distinct times 0.2 apart, dt=1 → both round to step 0.
        let times = [10.0, 10.2];
        let rows = [2, 3];
        let o = TimeOpts {
            origin: None,
            time_unit: "days",
            dt: 1.0,
            t_start: 0.0,
            format: TimeFormat::Auto,
        };
        let err = check_substeps_and_grid(&times, &rows, &o, false).unwrap_err();
        assert!(err.contains("same integrator substep"), "{err}");
        assert!(err.contains("line 2") && err.contains("line 3"), "{err}");
    }

    #[test]
    fn distinct_substep_ok_when_spaced() {
        let times = [0.0, 1.0, 2.0, 7.0];
        let rows = [2, 3, 4, 5];
        let o = opts_days(None);
        assert!(check_substeps_and_grid(&times, &rows, &o, false).is_ok());
    }

    #[test]
    fn identical_times_allowed() {
        // Multi-stream obs at the same point: identical times collide on a
        // substep but are NOT an error.
        let times = [7.0, 7.0, 14.0];
        let rows = [2, 3, 4];
        let o = opts_days(None);
        assert!(check_substeps_and_grid(&times, &rows, &o, false).is_ok());
    }

    // ── W326 + --time-format internal-days (Phase 1 of typed-time
    //    proposal 2026-05-22, Rule 6) ──────────────────────────────

    #[test]
    fn time_format_internal_days_is_an_alias_for_numeric() {
        // The proposal-named opt-in `--time-format internal-days`
        // must parse and map to `TimeFormat::Numeric`. This is the
        // documented suppress-W326 path.
        let tf: TimeFormat = "internal-days".parse().unwrap();
        assert_eq!(tf, TimeFormat::Numeric);
    }

    #[test]
    fn time_format_internal_days_loads_numeric_column() {
        // With --time-format internal-days, a numeric time column under
        // an anchored model loads and DOES NOT emit W326. We can't
        // directly assert on stderr in this test harness, but we can
        // assert the load succeeds and gives the expected numeric values.
        let cells = ["0", "1", "7", "14"];
        let mut o = opts_days(Some("2020-01-01"));
        o.format = TimeFormat::Numeric;
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![0.0, 1.0, 7.0, 14.0]);
    }

    #[test]
    fn numeric_with_origin_under_auto_still_loads() {
        // The W326 warning is informational only; the load still
        // succeeds. This is the proposal's "no hard breakage; one
        // warning at the data boundary" contract.
        let cells = ["-34", "-33", "0", "5"];
        let o = opts_days(Some("2020-02-28"));
        let t = convert_time_column(&cells, &o, 2).unwrap();
        assert_eq!(t, vec![-34.0, -33.0, 0.0, 5.0]);
    }

    #[test]
    fn time_format_parse_rejects_unknown() {
        let r: Result<TimeFormat, _> = "ymd".parse();
        assert!(r.is_err());
        let e = r.unwrap_err();
        // Updated error message mentions internal-days in the alternatives.
        assert!(e.contains("internal-days"), "got: {e}");
    }
}
