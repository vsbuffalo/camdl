//! The range on which a time function is defined, and the record of a run
//! evaluating it outside that range.
//!
//! A forcing built from a data table (`interpolated`, read linearly, as a
//! cubic spline or piecewise-constant) is defined from its first data point to
//! its last; a `bspline` from the start of its window to the end. Outside that
//! range the lookup holds the end value. Whether a run may rely on the hold is
//! the command's decision (an error by default, a warning under
//! `--allow-forcing-extrapolation`; spec `docs/dev/temporal-architecture.md`
//! §4.6, D44 and D48), so the engine only records that it happened, through
//! [`crate::run_notes`], and goes on holding the value exactly as before.
//!
//! The record reads the times actually evaluated, not the run's span: a
//! forcing read only under a condition that is false past its range records
//! nothing. Each side of the range has one state, clear, recording or
//! recorded: the first clamped evaluation moves it from clear to recording,
//! records the note, and then marks it recorded, so it is written once per
//! process for each compiled forcing and particle threads never contend on it;
//! a counter would be written on every clamped evaluation. A clamped
//! evaluation on another thread that finds it recording waits until it is
//! recorded, so no evaluation outside the range returns before the note is
//! reported: a command that reads its policy's verdict after a computation
//! sees the verdict on every hold that computation made. The record names
//! the side, not a time: which time outside the range a run over many threads
//! meets first depends on its threads.

use std::sync::atomic::{AtomicU8, Ordering};

/// A side's state: no evaluation outside the range yet.
const CLEAR: u8 = 0;
/// One evaluation outside the range is recording its note.
const RECORDING: u8 = 1;
/// The note is recorded and reported.
const RECORDED: u8 = 2;

use crate::schedule::output_tol;

/// What bounds the range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    /// The first and last time of an `interpolated` forcing's data.
    DataTable,
    /// The window of a `bspline` forcing.
    Window,
}

/// Which side of its range a forcing was evaluated on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Before,
    After,
}

/// A forcing evaluated outside its range, on one side.
#[derive(Debug, Clone, PartialEq)]
pub struct OutsideRange {
    pub forcing: String,
    pub kind: RangeKind,
    /// The range `[lo, hi]` on which the forcing is defined, in model time.
    pub lo: f64,
    pub hi: f64,
    pub side: Side,
    /// Whether the forcing has a `lag`: its range bounds its argument,
    /// `t − lag`, not the run's time.
    pub lagged: bool,
}

impl OutsideRange {
    /// The end of the range whose value is held.
    pub fn held_end(&self) -> f64 {
        match self.side {
            Side::Before => self.lo,
            Side::After => self.hi,
        }
    }

    /// What bounds the range: "its data" or "its window".
    pub fn range_phrase(&self) -> &'static str {
        match self.kind {
            RangeKind::DataTable => "its data",
            RangeKind::Window => "its window",
        }
    }

    /// "after its last data point", "before the start of its window", ….
    pub fn side_phrase(&self) -> &'static str {
        match (self.kind, self.side) {
            (RangeKind::DataTable, Side::Before) => "before its first data point",
            (RangeKind::DataTable, Side::After) => "after its last data point",
            (RangeKind::Window, Side::Before) => "before the start of its window",
            (RangeKind::Window, Side::After) => "after the end of its window",
        }
    }

    /// The span over which the end value is held, as far as the run reached.
    pub fn held_span(&self) -> String {
        match self.side {
            Side::Before => format!("held from the start of the run to t = {}", self.lo),
            Side::After => format!("held from t = {} to the end of the run", self.hi),
        }
    }

    /// The clause a lagged forcing adds: its times are those of its argument.
    pub fn lag_clause(&self) -> &'static str {
        if self.lagged { " (the forcing has a lag, so these are times of its argument, t − lag)" } else { "" }
    }
}

/// The warning said under the override.
impl std::fmt::Display for OutsideRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "forcing `{}` is defined on t = {} to t = {} ({}); the run evaluated it {}, and its \
             value at t = {} is {}{}",
            self.forcing,
            self.lo,
            self.hi,
            self.range_phrase(),
            self.side_phrase(),
            self.held_end(),
            self.held_span(),
            self.lag_clause(),
        )
    }
}

/// The range a compiled forcing is defined on, with one record state per
/// side. Shared (through `Arc`) by every clone of the compiled model, so a run
/// over many threads records each side once.
#[derive(Debug)]
pub struct DefinedRange {
    forcing: String,
    kind: RangeKind,
    lo: f64,
    hi: f64,
    lagged: bool,
    before: AtomicU8,
    after: AtomicU8,
}

impl DefinedRange {
    pub fn new(forcing: &str, kind: RangeKind, lo: f64, hi: f64, lagged: bool) -> Self {
        DefinedRange {
            forcing: forcing.to_string(),
            kind,
            lo,
            hi,
            lagged,
            before: AtomicU8::new(CLEAR),
            after: AtomicU8::new(CLEAR),
        }
    }

    /// Note an evaluation at `t` (the forcing's argument): if `t` is outside
    /// the range, record that it was evaluated on that side (once per side).
    /// A NaN time is never outside: both comparisons are false. A time within
    /// [`output_tol`] of an end is the end, not past it: it absorbs the
    /// rounding of a time built by additions of `dt`, and the held value at
    /// such a time equals the value at the end.
    #[inline]
    pub fn note(&self, t: f64) {
        if t < self.lo - output_tol(self.lo) {
            self.outside(Side::Before);
        } else if t > self.hi + output_tol(self.hi) {
            self.outside(Side::After);
        }
    }

    /// Each side of the range on which this forcing has been evaluated, in
    /// the order before, after: what this compiled forcing has recorded.
    pub fn held(&self) -> Vec<OutsideRange> {
        [(Side::Before, &self.before), (Side::After, &self.after)]
            .into_iter()
            .filter(|(_, state)| state.load(Ordering::Acquire) != CLEAR)
            .map(|(side, _)| self.outside_range(side))
            .collect()
    }

    fn outside_range(&self, side: Side) -> OutsideRange {
        OutsideRange {
            forcing: self.forcing.clone(),
            kind: self.kind,
            lo: self.lo,
            hi: self.hi,
            side,
            lagged: self.lagged,
        }
    }

    #[cold]
    #[inline(never)]
    fn outside(&self, side: Side) {
        let state = match side {
            Side::Before => &self.before,
            Side::After => &self.after,
        };
        if state.load(Ordering::Acquire) == RECORDED {
            return;
        }
        match state.compare_exchange(CLEAR, RECORDING, Ordering::Acquire, Ordering::Acquire) {
            Ok(_) => {
                // Marks the side recorded when the report returns, or unwinds,
                // so a waiting thread never waits on a report that ended. The
                // Release store makes what the report wrote visible to any
                // thread that then reads RECORDED.
                struct Recorded<'a>(&'a AtomicU8);
                impl Drop for Recorded<'_> {
                    fn drop(&mut self) {
                        self.0.store(RECORDED, Ordering::Release);
                    }
                }
                let _recorded = Recorded(state);
                crate::run_notes::record(crate::run_notes::RunNote::ForcingOutsideRange(self.outside_range(side)));
            }
            Err(_) => {
                while state.load(Ordering::Acquire) != RECORDED {
                    std::thread::yield_now();
                }
            }
        }
    }
}
