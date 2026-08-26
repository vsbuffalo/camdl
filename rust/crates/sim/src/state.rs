/// Integer compartment state — one `i64` per integer compartment, in model order.
#[derive(Debug, Clone, PartialEq)]
pub struct IntState {
    pub counts: Vec<i64>,
}

impl IntState {
    pub fn new(n: usize) -> Self {
        IntState { counts: vec![0; n] }
    }

    pub fn from_vec(counts: Vec<i64>) -> Self {
        IntState { counts }
    }

    /// Clamp all components to ≥ 0 in-place.
    /// Returns the number of components that were clamped (0 = no violation).
    pub fn clamp_nonneg(&mut self) -> usize {
        let mut clamped = 0;
        for v in &mut self.counts {
            if *v < 0 {
                *v = 0;
                clamped += 1;
            }
        }
        clamped
    }

    /// gh#audit-C5 / S2. Detect a negative count and return the
    /// (local_idx, value) pair without modifying state. Caller
    /// converts to SimError::NegativeCount with the appropriate
    /// cause discriminator. Used by gillespie / chain-binomial to
    /// replace the previous "warn + silently clamp" anti-pattern.
    pub fn first_negative(&self) -> Option<(usize, i64)> {
        self.counts.iter()
            .enumerate()
            .find_map(|(i, &v)| if v < 0 { Some((i, v)) } else { None })
    }

    pub fn total(&self) -> i64 {
        self.counts.iter().sum()
    }
}

/// Real compartment state — one `f64` per real compartment, in model order.
#[derive(Debug, Clone, PartialEq)]
pub struct RealState {
    pub values: Vec<f64>,
}

impl RealState {
    pub fn new(n: usize) -> Self {
        RealState { values: vec![0.0; n] }
    }

    pub fn from_vec(values: Vec<f64>) -> Self {
        RealState { values }
    }

    /// Clamp all components to ≥ 0 in-place.
    /// Returns the number of components that were clamped.
    pub fn clamp_nonneg(&mut self) -> usize {
        let mut clamped = 0;
        for v in &mut self.values {
            if *v < 0.0 {
                *v = 0.0;
                clamped += 1;
            }
        }
        clamped
    }
}

/// Cumulative flow counters — one `u64` per transition, reset at each output boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowVec {
    pub counts: Vec<u64>,
}

impl FlowVec {
    pub fn new(n: usize) -> Self {
        FlowVec { counts: vec![0; n] }
    }

    pub fn from_vec(counts: Vec<u64>) -> Self {
        FlowVec { counts }
    }

    pub fn reset(&mut self) {
        for v in &mut self.counts {
            *v = 0;
        }
    }

    pub fn add(&mut self, transition_idx: usize, n: u64) {
        self.counts[transition_idx] += n;
    }
}

/// Per-transition cumulative flow recorded on a [`Snapshot`].
///
/// Integer for the stochastic backends (Gillespie, chain-binomial): a flow is a
/// genuine count of fired events. Real for the ODE backend: a flow is the
/// continuous `rate·dt` accumulation, and rounding it to an integer would
/// silently zero out sub-unit flows — the regime of slow transitions such as TB
/// reactivation, where the deterministic likelihood then collapses to `-∞`. The
/// two backends produce genuinely different objects, so the type keeps them
/// distinct rather than quantizing the real one through a `u64`.
#[derive(Debug, Clone, PartialEq)]
pub enum Flows {
    Int(Vec<u64>),
    Real(Vec<f64>),
}

impl Flows {
    pub fn len(&self) -> usize {
        match self {
            Flows::Int(v) => v.len(),
            Flows::Real(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Integer flows from a stochastic backend. Panics on `Real` (an ODE
    /// trajectory); callers that can observe either backend must match `Flows`
    /// explicitly rather than assume integer.
    pub fn as_int(&self) -> &[u64] {
        match self {
            Flows::Int(v) => v,
            Flows::Real(_) => panic!("Flows::as_int called on real (ODE) flows"),
        }
    }

    /// Real flows from the ODE backend. Panics on `Int` (a stochastic
    /// trajectory).
    pub fn as_real(&self) -> &[f64] {
        match self {
            Flows::Real(v) => v,
            Flows::Int(_) => panic!("Flows::as_real called on integer (stochastic) flows"),
        }
    }

    /// Numeric value of transition `i`'s flow as `f64` (integer flows widen
    /// losslessly). For backend-agnostic accumulators that span both kinds.
    pub fn value(&self, i: usize) -> f64 {
        match self {
            Flows::Int(v) => v[i] as f64,
            Flows::Real(v) => v[i],
        }
    }
}

/// A single recorded state snapshot at time `t`.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub t: f64,
    pub int_state: IntState,
    pub real_state: RealState,
    /// Cumulative flows since the previous snapshot (or since t_start for the first).
    pub flows: Flows,
}

/// The full time series produced by a simulation run.
///
/// **Initial-row convention (load-bearing for any consumer that differences
/// state against flows).** Every writer of a `(state, flows)` trajectory MUST
/// emit the initial-condition row first: a snapshot at `t_start` carrying the
/// initial state and **zeroed** flows (no interval precedes `t_start`).
/// Subsequent rows carry the flows accumulated over the interval ending at
/// their time. This is exactly what makes the aggregate identity
/// `Σ flow_<tr> == −Δcompartment` hold over the whole path for a compartment
/// whose only dynamics are transitions (no events/balance touching it): the
/// first interval's flow gets its state decrement recorded against the true
/// initial value, not a post-first-substep one.
///
/// The three forward backends (`chain_binomial`, `gillespie`, `ode`) honour
/// this via the `if output_due_at { push t_start row; reset flows }` prologue
/// before their integration loop; the PGAS smoother honours it by prepending
/// the row in `PGASTrajectory::to_trajectory`. Dropping the `t_start` row is a
/// silent-wrong bug (gh#270): the per-step delta still reconciles, but the
/// aggregate is off by exactly the first interval's flow — invisible to a
/// per-step check, caught by `tests/trajectory_flow_reconciliation.rs`.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub snapshots: Vec<Snapshot>,
    /// Per-transition firing diagnostics (populated by Gillespie; empty for chain-binomial).
    pub transition_diagnostics: Vec<crate::transition_diagnostics::TransitionDiagnostics>,
    /// Reactive-policy firings (gh#204) — the source of `reactive_log.tsv`.
    /// `Some` exactly when the run had an active reactive agenda (so the log is
    /// a declared artifact written even with zero firings); `None` when the
    /// model has no active reactive policy (no log artifact). The `Option`
    /// distinguishes "active, never crossed" from "no policy" — an empty `Vec`
    /// alone could not.
    pub reactive_log: Option<Vec<crate::reactive::ReactiveFiring>>,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

impl Trajectory {
    pub fn new() -> Self {
        Trajectory {
            snapshots: Vec::new(),
            transition_diagnostics: Vec::new(),
            reactive_log: None,
        }
    }

    pub fn push(&mut self, snap: Snapshot) {
        self.snapshots.push(snap);
    }
}

