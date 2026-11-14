//! What a prequential trace records.

/// What a scored value was accumulated over — the answer a [`StreamTimes`]
/// gives for one of its rows, recorded on every per-stream prequential score so
/// a comparison can check that two traces scored the *same quantity* at a
/// step, not merely at the same time (gh#833; `compare`'s window gate).
///
/// A tagged sum type rather than an `Option<(f64, f64)>` because "no span" has
/// two meanings that must not be confused: an instant reading genuinely has
/// none, and a trace written before this was recorded simply does not say.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Coverage {
    /// The trace predates coverage recording (prequential schema < 4); nothing
    /// is known about what its rows covered.
    #[default]
    Unrecorded,
    /// A state read at an instant — prevalence, a compartment count. No span.
    Instant,
    /// The flow accumulated over `[start, stop)`.
    Interval { start: f64, stop: f64 },
}
