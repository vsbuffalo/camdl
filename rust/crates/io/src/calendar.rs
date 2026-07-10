//! Calendar semantics that travel with an output artifact.
//!
//! The numeric `time` axis in every emitted TSV is in units of `time_unit`
//! measured from `origin`. A consumer needs both to map `time → Date`; without
//! them it must re-derive (or guess) calendar semantics per file. This module
//! is the single producer of the `"calendar"` manifest block, so every exporter
//! — predictive / observed / quantities / trajectories — agrees on what it
//! writes and a consumer reads calendar semantics from one shape everywhere.

use ir::Model;

/// The `"calendar"` block stamped into a manifest.
///
/// `origin` is `None` for a bare-numeric-time model (one with no `origin =
/// date(...)`), which a consumer reads as "plot `time` numerically" — the
/// numeric fallback. When present, a consumer maps `time → date` as `origin +
/// time · days_per_unit` **days**. A calendar-anchored model is constrained to
/// `days`/`weeks` (camdl rejects `origin` combined with `months`/`years` — E320,
/// since a calendar month/year is not a constant number of days), so
/// `days_per_unit` is 1 or 7 whenever there is a date to map and the recipe is
/// exact. It is exported so a consumer converts with no hardcoded unit→days
/// table, and it still reports the (average-Gregorian) unit length for an
/// unanchored `months`/`years` axis.
///
/// We deliberately do NOT export the model's internal `origin_rata_die`: that
/// day-number is keyed to camdl's arbitrary epoch (`caltime.rs`), is meaningful
/// only when differenced with another camdl rata-die, and would mislead an
/// external consumer that read it as a standard Julian/Rata day number. The ISO
/// `origin` string is the portable truth.
#[derive(Clone, Debug, PartialEq)]
pub struct CalendarMeta {
    /// The model's `time_unit`: `days` / `weeks` / `months` / `years`.
    pub time_unit: String,
    /// ISO `YYYY-MM-DD` origin, or `None` when the model is unanchored.
    pub origin: Option<String>,
    /// Exact length of one `time_unit` in days, camdl's canonical convention
    /// (`days`=1, `weeks`=7, `months`=365.2425/12, `years`=365.2425). The factor
    /// a consumer multiplies `time` by to advance `origin`, identical to what
    /// camdl's `date` column uses.
    pub days_per_unit: f64,
}

impl CalendarMeta {
    /// Read the calendar semantics off a compiled model.
    pub fn from_model(m: &Model) -> Self {
        Self {
            time_unit: m.time_unit.clone(),
            origin: m.origin.clone(),
            // `time_unit` is compiler-validated to one of the four known units,
            // so the lookup never falls back.
            days_per_unit: ir::caltime::days_per_unit(&m.time_unit).unwrap_or(1.0),
        }
    }

    /// The `"calendar"` manifest block.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "time_unit": self.time_unit,
            "origin": self.origin,
            "days_per_unit": self.days_per_unit,
        })
    }
}
