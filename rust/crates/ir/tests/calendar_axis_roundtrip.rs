//! A date camdl writes reads back to the model time it was written from
//! (`docs/dev/temporal-architecture.md` §3.4; gh#839).
//!
//! [`CalendarAxis::date_of`] renders a model time as `YYYY-MM-DD`, or as
//! `YYYY-MM-DD+<frac>d` inside a day; [`CalendarAxis::parse_instant`] is the
//! loader's reading of the same text. These properties hold the pair to
//! returning the identical `f64`, not a nearby one: a sub-day output reloaded
//! as data must land on the times it was simulated at.

use ir::caltime::{CalendarAxis, IsoDate, TimeUnit};
use proptest::prelude::*;

fn origin() -> impl Strategy<Value = IsoDate> {
    (1900i64..2100, 1i64..=12, 1i64..=28)
        .prop_map(|(y, m, d)| IsoDate::parse(&format!("{y:04}-{m:02}-{d:02}")).unwrap())
}

fn axis(time_unit: TimeUnit, origin: IsoDate) -> CalendarAxis {
    CalendarAxis { time_unit, origin: Some(origin) }
}

/// Render then parse.
fn round_trip(a: &CalendarAxis, t: f64) -> f64 {
    let s = a.date_of(t).unwrap_or_else(|| panic!("t = {t} has no date"));
    a.parse_instant(s.as_str()).unwrap_or_else(|e| panic!("'{s}' (t = {t:?}) does not load: {e}"))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Under `days`, every time renders to text that loads back to it exactly.
    #[test]
    fn any_time_under_days_round_trips(
        o in origin(),
        t in prop_oneof![-20_000.0f64..50_000.0, -1.0f64..1.0],
    ) {
        let a = axis(TimeUnit::Days, o);
        prop_assert_eq!(round_trip(&a, t), t);
    }

    /// The sub-day grids simulations run on — hourly, tenths, thirds,
    /// quarters, halves — computed as `k · dt`, the way an output schedule
    /// produces them.
    #[test]
    fn sub_day_grid_times_round_trip(
        o in origin(),
        k in -2_000i64..20_000,
        dt in prop::sample::select(vec![1.0 / 24.0, 0.1, 1.0 / 3.0, 0.25, 0.5, 0.05]),
    ) {
        let a = axis(TimeUnit::Days, o);
        let t = k as f64 * dt;
        prop_assert_eq!(round_trip(&a, t), t);
    }

    /// Every time the loader produces from a dated cell — a whole date, or a
    /// date with a fractional-day suffix — renders back to text that loads to
    /// the same time, under each unit an origin is allowed with.
    #[test]
    fn loaded_times_round_trip(
        o in origin(),
        unit in prop::sample::select(vec![TimeUnit::Days, TimeUnit::Weeks]),
        days in -5_000i64..20_000,
        frac in 0u32..1_000_000,
    ) {
        let a = axis(unit, o);
        let day = a.date_of(days as f64 / if unit == TimeUnit::Weeks { 7.0 } else { 1.0 })
            .unwrap();
        let date = &day.as_str()[..10];
        let cell = if frac == 0 {
            date.to_string()
        } else {
            let f = format!("{:06}", frac);
            format!("{date}+0.{}d", f.trim_end_matches('0'))
        };
        let t = a.parse_instant(&cell).unwrap_or_else(|e| panic!("'{cell}': {e}"));
        prop_assert_eq!(round_trip(&a, t), t);
    }

    /// A whole date renders as the bare date it was read from.
    #[test]
    fn whole_dates_render_as_themselves(
        o in origin(),
        unit in prop::sample::select(vec![TimeUnit::Days, TimeUnit::Weeks]),
        target in origin(),
    ) {
        let a = axis(unit, o);
        let cell = target.to_string();
        let t = a.parse_instant(&cell).unwrap();
        let rendered = a.date_of(t).unwrap();
        prop_assert_eq!(rendered.as_str(), cell.as_str());
    }

    /// Under `weeks` a time is a count of days divided by seven, and some
    /// doubles are not the quotient of any double by seven: those cannot be
    /// written as a days offset that reads back exactly. They read back to an
    /// adjacent double, never further.
    #[test]
    fn any_time_under_weeks_reads_back_within_one_ulp(
        o in origin(),
        t in -3_000.0f64..7_000.0,
    ) {
        let a = axis(TimeUnit::Weeks, o);
        let back = round_trip(&a, t);
        prop_assert!(back == t || back == t.next_up() || back == t.next_down(),
            "t = {t:?} read back as {back:?}");
    }
}

/// A model with no origin has no dates to write.
#[test]
fn no_origin_no_date() {
    let a = CalendarAxis { time_unit: TimeUnit::Days, origin: None };
    assert!(a.date_of(3.5).is_none());
}

/// The suffix the renderer writes is the one gh#839 reported refused.
#[test]
fn the_reported_cell_loads() {
    let a = axis(TimeUnit::Days, IsoDate::parse("2026-07-01").unwrap());
    assert_eq!(a.parse_instant("2026-07-01+0.5d").unwrap(), 0.5);
    assert_eq!(a.date_of(0.5).unwrap().as_str(), "2026-07-01+0.5d");
}

/// A malformed suffix is named as camdl's own sub-day form, not as a
/// time-of-day; a timezone offset is still refused as one (gh#846).
#[test]
fn malformed_suffixes_are_refused_by_name() {
    let a = axis(TimeUnit::Days, IsoDate::parse("2020-01-01").unwrap());
    for cell in ["2020-01-01+1.5d", "2020-01-01+0d", "2020-01-01+0.0d", "2020-01-01+.5d",
                 "2020-01-01+0.5e-3d", "2020-01-01+-0.5d", "2020-01-01+d"] {
        let e = a.parse_instant(cell).expect_err(cell);
        assert!(matches!(e, ir::caltime::CalError::BadSubDay(_)), "{cell}: {e:?}");
        assert!(e.to_string().contains("YYYY-MM-DD+<frac>d"), "{e}");
    }
    let e = a.parse_instant("2020-01-01+05:00").unwrap_err();
    assert!(matches!(e, ir::caltime::CalError::ZoneUnsupported { .. }), "{e:?}");
    let none = CalendarAxis { time_unit: TimeUnit::Days, origin: None };
    assert!(matches!(none.parse_instant("2020-01-01"), Err(ir::caltime::CalError::NoOrigin(_))));
}
