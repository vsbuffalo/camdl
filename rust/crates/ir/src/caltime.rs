//! Calendar-time conversion — the **boundary translator** between ISO calendar
//! dates and camdl's internal continuous time axis (2026-05-22 proposal).
//!
//! Dates live only at the I/O edge: this module is the *only* place a calendar
//! date becomes (or is recovered from) an `f64` internal time. Below it,
//! everything is `f64` time in units of the model's `time_unit`, measured from
//! `origin`.
//!
//! **Cross-language contract.** The day-number (`rata_die`) algorithm and the
//! `days_per_unit` table MUST match the OCaml compiler
//! (`ocaml/lib/compiler/expander.ml`: `days_of_date` / `parse_date_to_float`)
//! exactly, so a `date()` literal in a model and the same date in a data file
//! convert identically. Both are pinned by the golden table in
//! `ir/golden/caltime.tsv`, checked by a Rust test here and an OCaml test.
//!
//! v1 scope: **dates only** (`YYYY-MM-DD`), naive (no timezone semantics). A
//! bare civil date is already zone-free and unambiguous, so a trailing zone
//! designator supplies information this module has chosen not to model and is
//! *refused*, never discarded (gh#846). Times of day (`…THH:MM:SS`) are
//! likewise rejected; they are deferred (proposal F2).

/// Error parsing or converting a calendar instant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalError {
    /// Not a `YYYY-MM-DD` date.
    BadFormat(String),
    /// Month not in 1..=12, or day not valid for the month/year.
    OutOfRange(String),
    /// A time-of-day component (`T…` / space + time) — deferred to F2.
    DatetimeUnsupported(String),
    /// A trailing timezone designator (`Z`, `+HH:MM`, `-HH:MM`) on an
    /// otherwise well-formed date. camdl models civil calendar dates and has
    /// no timezone semantics, so the offset is information it cannot
    /// represent; silently dropping it is the one response that cannot be
    /// right, so it is refused instead (gh#846). `date` is the civil date the
    /// cell reduces to, `zone` the designator that was refused — carried
    /// separately so the diagnostic can name both.
    ZoneUnsupported { date: String, zone: String },
    /// `time_unit` is not a recognised calendar unit.
    UnknownUnit(String),
}

impl std::fmt::Display for CalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalError::BadFormat(s) => {
                write!(f, "expected an ISO date 'YYYY-MM-DD', got '{s}'")
            }
            CalError::ZoneUnsupported { date, zone } => write!(
                f,
                "'{date}{zone}' carries a timezone offset '{zone}', but camdl \
                 models civil calendar dates and has no timezone semantics. \
                 The offset cannot be honoured, and dropping it silently would \
                 change what the data says, so it is refused: if '{date}' is \
                 the civil day you mean, write it without the offset."
            ),
            CalError::OutOfRange(s) => write!(f, "date out of range: '{s}'"),
            CalError::DatetimeUnsupported(s) => write!(
                f,
                "time-of-day is not supported (dates only in v1): '{s}'"
            ),
            CalError::UnknownUnit(s) => write!(
                f,
                "'{s}' is not a calendar time unit (expected days/weeks/months/years)"
            ),
        }
    }
}

impl std::error::Error for CalError {}

/// Proleptic-Gregorian day number — **identical formula to the OCaml
/// `days_of_date`** (Hatcher/Richards; the `-694025` epoch offset is arbitrary
/// but shared, so absolute day numbers match too; only deltas are load-bearing).
/// Valid for dates CE 1583+.
pub fn rata_die(y: i64, m: i64, d: i64) -> i64 {
    let (yy, mm) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    365 * yy + yy / 4 - yy / 100 + yy / 400 + (153 * (mm + 1)) / 5 + d - 694025
}

/// Canonical duration of one `time_unit` in **days**. Matches the OCaml `D`
/// table. `months`/`years` are *average* lengths (365.2425-day Gregorian year).
pub fn days_per_unit(time_unit: &str) -> Result<f64, CalError> {
    match time_unit {
        "days" => Ok(1.0),
        "weeks" => Ok(7.0),
        "months" => Ok(365.2425 / 12.0),
        "years" => Ok(365.2425),
        other => Err(CalError::UnknownUnit(other.to_string())),
    }
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Parse an ISO calendar date `YYYY-MM-DD`, returning `(year, month, day)`.
///
/// Rejects a trailing **zone designator** (`Z`, `+HH:MM`, `-HH:MM`) as
/// `ZoneUnsupported`: a bare date already denotes a civil-calendar day with no
/// timezone semantics, so an offset is information camdl does not model and
/// must not silently delete (gh#846). Rejects time-of-day forms (`T…` / space +
/// time) as `DatetimeUnsupported` (v1). Validates month and day (leap-aware).
pub fn parse_iso_date(s: &str) -> Result<(i64, i64, i64), CalError> {
    let s = s.trim();
    // The date portion is the first 10 chars: YYYY-MM-DD.
    if s.len() < 10 {
        return Err(CalError::BadFormat(s.to_string()));
    }
    let (date_part, rest) = s.split_at(10);

    // Classify the remainder: a bare zone designator → refused, naming the
    // offset (gh#846); a time-of-day (T/space then digits) → datetime,
    // rejected in v1. The `is_zone` shape is deliberately narrow, so camdl's
    // own fractional-day `--dates` suffix (`+0.25d`, gh#839) does not match
    // it and keeps its own diagnostic.
    if !rest.is_empty() {
        let is_zone = rest == "Z"
            || rest == "z"
            || ((rest.starts_with('+') || rest.starts_with('-'))
                && rest.len() == 6
                && rest.as_bytes()[3] == b':'
                && rest[1..3].chars().all(|c| c.is_ascii_digit())
                && rest[4..6].chars().all(|c| c.is_ascii_digit()));
        if is_zone {
            return Err(CalError::ZoneUnsupported {
                date: date_part.to_string(),
                zone: rest.to_string(),
            });
        }
        // A `T` or space followed by time-of-day, or any other trailer.
        return Err(CalError::DatetimeUnsupported(s.to_string()));
    }

    let bytes = date_part.as_bytes();
    // Strict YYYY-MM-DD shape.
    let shape_ok = bytes[4] == b'-'
        && bytes[7] == b'-'
        && date_part[0..4].chars().all(|c| c.is_ascii_digit())
        && date_part[5..7].chars().all(|c| c.is_ascii_digit())
        && date_part[8..10].chars().all(|c| c.is_ascii_digit());
    if !shape_ok {
        return Err(CalError::BadFormat(s.to_string()));
    }
    let y: i64 = date_part[0..4].parse().map_err(|_| CalError::BadFormat(s.to_string()))?;
    let m: i64 = date_part[5..7].parse().map_err(|_| CalError::BadFormat(s.to_string()))?;
    let d: i64 = date_part[8..10].parse().map_err(|_| CalError::BadFormat(s.to_string()))?;

    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return Err(CalError::OutOfRange(s.to_string()));
    }
    Ok((y, m, d))
}

/// Convert an ISO date string to internal time, given the `origin` date string
/// and the model's `time_unit`:
/// `t = (rata_die(date) − rata_die(origin)) / days_per_unit(unit)`.
/// May be negative (date before origin); fractional under non-day units.
pub fn date_to_internal(origin: &str, date: &str, time_unit: &str) -> Result<f64, CalError> {
    let (oy, om, od) = parse_iso_date(origin)?;
    let (ty, tm, td) = parse_iso_date(date)?;
    let delta = rata_die(ty, tm, td) - rata_die(oy, om, od);
    Ok(delta as f64 / days_per_unit(time_unit)?)
}

/// Civil date from a rata-die day number (inverse of `rata_die`). Used to render
/// internal times back as dates. Algorithm: Howard Hinnant's `civil_from_days`,
/// shifted by the same `-694025` epoch offset `rata_die` uses.
pub fn date_from_rata_die(rd: i64) -> (i64, i64, i64) {
    // Convert our epoch (rata_die) to days-since-1970 used by the civil algo.
    // rata_die(1970,1,1):
    let z = rd - rata_die(1970, 1, 1) + 719_468; // days since 0000-03-01 era
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// Day number relative to the **Unix epoch** (1970-01-01 → 0) for a civil
/// date. This is the framing wall-clock / provenance timestamp parsers use
/// (run-record `created`/`finished` times, `fit`-listing ages) — *not* model
/// time, which is the `f64` axis from `origin`. It equals
/// `rata_die(y, m, d) − rata_die(1970, 1, 1)`, so those parsers share the one
/// Gregorian arithmetic here instead of re-deriving Hinnant's era formula at
/// each call site.
pub fn unix_epoch_days(y: i64, m: i64, d: i64) -> i64 {
    rata_die(y, m, d) - rata_die(1970, 1, 1)
}

/// Civil `(year, month, day)` for a Unix-epoch day count — inverse of
/// [`unix_epoch_days`].
pub fn civil_from_unix_epoch_days(days: i64) -> (i64, i64, i64) {
    date_from_rata_die(days + rata_die(1970, 1, 1))
}

/// Render an internal time back to an ISO date, given `origin` and `time_unit`
/// (inverse of [`date_to_internal`]). Rounds to the nearest whole day, so the
/// result is always a bare `YYYY-MM-DD`.
///
/// This is the **point-estimate annotation** renderer: a single fitted instant
/// (`fit summary`'s `instant`-kind estimands) shows a readable calendar date
/// while the numeric `t` stays canonical. There is one row per estimate, so a
/// rounded date can never coalesce distinct points. For the `--dates` *column*
/// of simulation output — one row per `dt`-grid timepoint, which a user may
/// join on — use [`internal_to_date_hires`], which keeps sub-day timepoints
/// distinct (gh#108).
pub fn internal_to_date(origin: &str, t: f64, time_unit: &str) -> Result<String, CalError> {
    let (oy, om, od) = parse_iso_date(origin)?;
    let delta_days = (t * days_per_unit(time_unit)?).round() as i64;
    let (y, m, d) = date_from_rata_die(rata_die(oy, om, od) + delta_days);
    Ok(format!("{y:04}-{m:02}-{d:02}"))
}

/// Render an internal time back to a calendar label that stays **one-to-one
/// with the timepoint**, for the `--dates` output column (gh#108).
///
/// A whole-day offset renders as a bare `YYYY-MM-DD` — identical to
/// [`internal_to_date`], so integer-day steps are unchanged. A **sub-day**
/// offset (a fractional `dt < 1 day`, the canonical hot-epidemic regime)
/// renders the floor date with the fractional day appended as a `+<frac>d`
/// suffix (e.g. `2020-01-01+0.25d`). Without this, `dt = 0.5` would coalesce
/// `t = 0.0, 0.5, 1.0` onto `origin+0, origin+1, origin+1` — silently lossy for
/// a column users group/join on. The suffix is a fractional-day **delta**,
/// deliberately not the `YYYY-MM-DDTHH:MM` datetime form (datetimes are rejected
/// on input and out of scope — `docs/dates.md`).
pub fn internal_to_date_hires(origin: &str, t: f64, time_unit: &str) -> Result<String, CalError> {
    let (oy, om, od) = parse_iso_date(origin)?;
    let offset = t * days_per_unit(time_unit)?;
    // Floor to the calendar day; the remainder is the fractional day in [0, 1).
    // `floor`/remainder is well-defined for negative offsets (pre-origin dates),
    // unlike `round`, which would split a tie inconsistently across the sign.
    let floor_day = offset.floor();
    // Quantize the fractional day to absorb float noise from non-day units
    // (e.g. a "whole" date under `'years` lands on 13.999999999 days). Six
    // decimals resolves any practical sub-day `dt` while snapping near-integers
    // back to a bare date.
    let frac = ((offset - floor_day) * 1e6).round() / 1e6;
    let (floor_day, frac) = if frac >= 1.0 {
        // Quantization pushed the fraction up to a full day; carry it.
        (floor_day + 1.0, 0.0)
    } else {
        (floor_day, frac)
    };
    let (y, m, d) = date_from_rata_die(rata_die(oy, om, od) + floor_day as i64);
    let date = format!("{y:04}-{m:02}-{d:02}");
    if frac == 0.0 {
        Ok(date)
    } else {
        // Trim trailing zeros from the fractional-day suffix: 0.250000 → 0.25.
        let frac_str = format!("{frac:.6}");
        let frac_str = frac_str.trim_end_matches('0');
        Ok(format!("{date}+{frac_str}d"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_and_century_rules() {
        // 2000 is a leap year (divisible by 400); 1900 is not (by 100, not 400).
        assert_eq!(rata_die(2000, 3, 1) - rata_die(2000, 2, 28), 2); // Feb 29 exists
        assert_eq!(rata_die(1900, 3, 1) - rata_die(1900, 2, 28), 1); // no Feb 29
        assert_eq!(rata_die(2020, 3, 1) - rata_die(2020, 2, 28), 2); // 2020 leap
        assert!(parse_iso_date("2020-02-29").is_ok());
        assert!(matches!(parse_iso_date("2021-02-29"), Err(CalError::OutOfRange(_))));
        assert!(matches!(parse_iso_date("1900-02-29"), Err(CalError::OutOfRange(_))));
        assert!(parse_iso_date("2000-02-29").is_ok());
    }

    #[test]
    fn month_boundary_deltas() {
        assert_eq!(rata_die(2020, 2, 1) - rata_die(2020, 1, 1), 31); // Jan→Feb
        assert_eq!(rata_die(2020, 3, 1) - rata_die(2020, 2, 1), 29); // Feb→Mar (leap)
        assert_eq!(rata_die(2021, 1, 1) - rata_die(2020, 12, 1), 31); // Dec→Jan
    }

    #[test]
    fn sign_and_zero() {
        assert_eq!(date_to_internal("2020-02-28", "2020-02-28", "days").unwrap(), 0.0);
        assert_eq!(date_to_internal("2020-02-28", "2020-02-18", "days").unwrap(), -10.0);
        // antisymmetry
        let a = date_to_internal("2020-01-01", "2020-03-01", "days").unwrap();
        let b = date_to_internal("2020-03-01", "2020-01-01", "days").unwrap();
        assert_eq!(a, -b);
    }

    #[test]
    fn per_unit_division() {
        // 14 days under each unit.
        let d14 = || date_to_internal("2020-01-01", "2020-01-15", "days").unwrap();
        assert_eq!(d14(), 14.0); // exact integer f64 under 'days
        assert_eq!(date_to_internal("2020-01-01", "2020-01-15", "weeks").unwrap(), 14.0 / 7.0);
        assert!((date_to_internal("2020-01-01", "2020-01-15", "years").unwrap()
            - 14.0 / 365.2425)
            .abs()
            < 1e-12);
        assert!(matches!(days_per_unit("fortnights"), Err(CalError::UnknownUnit(_))));
    }

    #[test]
    fn subday_timepoints_render_distinctly() {
        // gh#108: with a sub-day `dt` the `--dates` column must not coalesce
        // distinct timepoints onto the same calendar label. t=0.0 and t=0.25
        // are genuinely different instants and must render to DISTINCT strings.
        let d0 = internal_to_date_hires("2020-01-01", 0.0, "days").unwrap();
        let d_quarter = internal_to_date_hires("2020-01-01", 0.25, "days").unwrap();
        let d_half = internal_to_date_hires("2020-01-01", 0.5, "days").unwrap();
        assert_ne!(d0, d_quarter, "t=0.0 and t=0.25 must render distinctly");
        assert_ne!(d_quarter, d_half, "t=0.25 and t=0.5 must render distinctly");
        // Whole-day offsets keep the bare YYYY-MM-DD form (no fractional suffix).
        assert_eq!(d0, "2020-01-01");
        assert_eq!(internal_to_date_hires("2020-01-01", 1.0, "days").unwrap(), "2020-01-02");
        // The fractional suffix is a `+<frac>d` delta on the floor date.
        assert_eq!(d_quarter, "2020-01-01+0.25d");
        assert_eq!(d_half, "2020-01-01+0.5d");
        // The full sub-day-step sequence from the issue stays one-to-one:
        // t = 0.0, 0.5, 1.0, 1.5, 2.0 must yield five distinct labels.
        let seq: Vec<String> = [0.0, 0.5, 1.0, 1.5, 2.0]
            .iter()
            .map(|&t| internal_to_date_hires("2020-01-01", t, "days").unwrap())
            .collect();
        let mut uniq = seq.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), seq.len(), "sub-day steps must not coalesce: {seq:?}");
    }

    #[test]
    fn hires_negative_subday_and_whole_day_carry() {
        // Negative (pre-origin) sub-day offset: floor goes to the day below and
        // the fraction is the positive remainder — distinct from the bare date.
        // -0.25 days → floor day -1 (2019-12-31), frac 0.75.
        let neg = internal_to_date_hires("2020-01-01", -0.25, "days").unwrap();
        assert_eq!(neg, "2019-12-31+0.75d");
        let neg_whole = internal_to_date_hires("2020-01-01", -1.0, "days").unwrap();
        assert_eq!(neg_whole, "2019-12-31", "a whole pre-origin day stays bare");
        // Whole-day round-trip parity with the rounded renderer for integer t.
        for date in ["2019-01-01", "2020-02-29", "2020-12-31", "2026-05-22"] {
            let t = date_to_internal("2020-02-28", date, "days").unwrap();
            assert_eq!(internal_to_date_hires("2020-02-28", t, "days").unwrap(), date);
        }
    }

    #[test]
    fn rounded_renderer_keeps_bare_date_for_subday() {
        // `internal_to_date` is the point-estimate annotation renderer: it must
        // keep rounding to a bare YYYY-MM-DD (the `fit summary` contract), even
        // for a fractional instant. Only `internal_to_date_hires` is sub-day.
        assert_eq!(internal_to_date("2020-01-01", 0.25, "days").unwrap(), "2020-01-01");
        assert_eq!(internal_to_date("2020-01-01", 0.5, "days").unwrap(), "2020-01-02");
        assert_eq!(internal_to_date("2020-01-01", 1.4, "days").unwrap(), "2020-01-02");
    }

    #[test]
    fn round_trip() {
        for date in ["2019-01-01", "2020-02-29", "2020-12-31", "1861-10-01", "2026-05-22"] {
            let t = date_to_internal("2020-02-28", date, "days").unwrap();
            let back = internal_to_date("2020-02-28", t, "days").unwrap();
            assert_eq!(back, date, "round-trip failed for {date}");
        }
        // Negative internal time round-trips to a date before the origin.
        let t = date_to_internal("2020-02-28", "2020-01-21", "days").unwrap();
        assert!(t < 0.0);
        assert_eq!(internal_to_date("2020-02-28", t, "days").unwrap(), "2020-01-21");
    }

    #[test]
    fn grammar_rejects_zone_designators(){
        // gh#846: camdl has no timezone semantics, so an offset is information
        // it cannot represent. It is refused rather than discarded, and the
        // message names the offset so the user can see what was rejected.
        for (s, zone) in [
            ("2020-03-15Z", "Z"),
            ("2020-03-15z", "z"),
            ("2020-03-15+06:00", "+06:00"),
            ("2020-03-15-03:00", "-03:00"),
            ("2020-03-15+05:45", "+05:45"),
        ] {
            let e = parse_iso_date(s).expect_err(&format!("must reject '{s}'"));
            assert!(matches!(e, CalError::ZoneUnsupported { .. }), "for {s}: {e:?}");
            let msg = e.to_string();
            assert!(msg.contains(zone), "message must name the offset: {msg}");
            assert!(msg.contains(s), "message must echo the cell: {msg}");
        }
        // The bare civil date it reduces to is still accepted.
        assert_eq!(parse_iso_date("2020-03-15").unwrap(), (2020, 3, 15));
    }

    #[test]
    fn grammar_rejects() {
        // datetime forms (v1)
        assert!(matches!(parse_iso_date("2020-03-15T12:00"), Err(CalError::DatetimeUnsupported(_))));
        assert!(matches!(parse_iso_date("2020-03-15 12:00"), Err(CalError::DatetimeUnsupported(_))));
        // malformed
        for s in ["2020/03/15", "15-03-2020", "20-03-15", "2020-3-15", "", "2020-03", "abc"] {
            assert!(parse_iso_date(s).is_err(), "should reject '{s}'");
        }
        // out of range
        assert!(matches!(parse_iso_date("2020-13-01"), Err(CalError::OutOfRange(_))));
        assert!(matches!(parse_iso_date("2020-02-30"), Err(CalError::OutOfRange(_))));
        assert!(matches!(parse_iso_date("2020-00-10"), Err(CalError::OutOfRange(_))));
    }

    /// Civil dates are the whole of the time axis: a bare date is already
    /// zone-free and unambiguous, so distinct dates give consecutive integer
    /// `t` and an offset-bearing string never converts at all (gh#846).
    #[test]
    fn civil_dates_convert_and_offsets_do_not() {
        // distinct civil dates → consecutive integers
        let t15 = date_to_internal("2020-03-01", "2020-03-15", "days").unwrap();
        let t16 = date_to_internal("2020-03-01", "2020-03-16", "days").unwrap();
        let t17 = date_to_internal("2020-03-01", "2020-03-17", "days").unwrap();
        assert_eq!((t15, t16, t17), (14.0, 15.0, 16.0));
        // An offset is refused on either side of the conversion, so a zone can
        // never reach the internal axis through the origin either.
        for s in ["2020-03-15+01:00", "2020-03-15Z"] {
            assert!(date_to_internal("2020-03-01", s, "days").is_err(), "date {s}");
            assert!(date_to_internal(s, "2020-03-15", "days").is_err(), "origin {s}");
        }
    }

    /// Consolidation guard (2026-06-22 quality review, finding X-1): the
    /// wall-clock timestamp parsers (browse/cas/fit_table/table_row + sim
    /// diagnostic) used to each inline Howard Hinnant's `days_from_civil`
    /// (epoch 1970). `unix_epoch_days` / `civil_from_unix_epoch_days` route
    /// that through the canonical `rata_die`; this pins that they reproduce
    /// the inlined formula exactly and round-trip.
    #[test]
    fn unix_epoch_days_matches_inlined_hinnant_and_round_trips() {
        // Well-known anchors.
        assert_eq!(unix_epoch_days(1970, 1, 1), 0);
        assert_eq!(unix_epoch_days(1969, 12, 31), -1);
        assert_eq!(unix_epoch_days(2000, 1, 1), 10_957);
        assert_eq!(unix_epoch_days(2020, 1, 1), 18_262);

        // The exact `days_from_civil` formula the fork sites inlined.
        fn days_from_civil_inlined(y: i64, m: i64, d: i64) -> i64 {
            let y = if m <= 2 { y - 1 } else { y };
            let era = if y >= 0 { y } else { y - 399 } / 400;
            let yoe = y - era * 400;
            let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
            let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
            era * 146_097 + doe - 719_468
        }
        for &(y, m, d) in &[
            (1970, 1, 1), (1969, 12, 31), (1900, 2, 28), (2000, 2, 29),
            (2020, 1, 1), (2026, 6, 23), (1583, 1, 1), (2099, 12, 31),
        ] {
            let days = unix_epoch_days(y, m, d);
            assert_eq!(days, days_from_civil_inlined(y, m, d), "civil→days {y}-{m}-{d}");
            assert_eq!(civil_from_unix_epoch_days(days), (y, m, d), "round-trip {y}-{m}-{d}");
        }
    }
}
