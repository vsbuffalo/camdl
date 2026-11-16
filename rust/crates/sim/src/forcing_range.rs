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
//! nothing. Each side of the range has one set-once slot holding the first
//! time evaluated beyond it. A clamped evaluation loads the slot and writes it
//! only while it is unset, so the slot is written once per process for each
//! compiled forcing and particle threads never contend on it; a counter would
//! be written on every clamped evaluation.

use std::sync::atomic::{AtomicU64, Ordering};

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

/// A forcing evaluated outside its range: the first such time on one side.
#[derive(Debug, Clone, PartialEq)]
pub struct OutsideRange {
    pub forcing: String,
    pub kind: RangeKind,
    /// The range `[lo, hi]` on which the forcing is defined, in model time.
    pub lo: f64,
    pub hi: f64,
    pub side: Side,
    /// The first time the run evaluated it beyond `side` of the range. For a
    /// forcing with a `lag` this is the forcing's argument, `t − lag`.
    pub at: f64,
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
            "forcing `{}` is defined on t = {} to t = {} ({}); the run evaluated it {}, first at \
             t = {}, and its value at t = {} is {}{}",
            self.forcing,
            self.lo,
            self.hi,
            self.range_phrase(),
            self.side_phrase(),
            self.at,
            self.held_end(),
            self.held_span(),
            self.lag_clause(),
        )
    }
}

/// A slot no time has been written to. Every finite or infinite time's bits
/// differ from it (it is one NaN pattern, and a NaN time is never outside a
/// range: both comparisons are false).
const UNSET: u64 = u64::MAX;

/// The range a compiled forcing is defined on, with one set-once slot per
/// side. Shared (through `Arc`) by every clone of the compiled model, so a run
/// over many threads records each side once.
#[derive(Debug)]
pub struct DefinedRange {
    forcing: String,
    kind: RangeKind,
    lo: f64,
    hi: f64,
    lagged: bool,
    before: AtomicU64,
    after: AtomicU64,
}

impl DefinedRange {
    pub fn new(forcing: &str, kind: RangeKind, lo: f64, hi: f64, lagged: bool) -> Self {
        DefinedRange {
            forcing: forcing.to_string(),
            kind,
            lo,
            hi,
            lagged,
            before: AtomicU64::new(UNSET),
            after: AtomicU64::new(UNSET),
        }
    }

    /// Note an evaluation at `t` (the forcing's argument): if `t` is outside
    /// the range, record the first such time on that side. A time within
    /// [`output_tol`] of an end is the end, not past it: it absorbs the
    /// rounding of a time built by additions of `dt`, and the held value at
    /// such a time equals the value at the end.
    #[inline]
    pub fn note(&self, t: f64) {
        if t < self.lo - output_tol(self.lo) {
            self.first_outside(Side::Before, t);
        } else if t > self.hi + output_tol(self.hi) {
            self.first_outside(Side::After, t);
        }
    }

    #[cold]
    #[inline(never)]
    fn first_outside(&self, side: Side, t: f64) {
        let slot = match side {
            Side::Before => &self.before,
            Side::After => &self.after,
        };
        if slot.load(Ordering::Relaxed) != UNSET {
            return;
        }
        if slot.compare_exchange(UNSET, t.to_bits(), Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            crate::run_notes::record(crate::run_notes::RunNote::ForcingOutsideRange(OutsideRange {
                forcing: self.forcing.clone(),
                kind: self.kind,
                lo: self.lo,
                hi: self.hi,
                side,
                at: t,
                lagged: self.lagged,
            }));
        }
    }
}
