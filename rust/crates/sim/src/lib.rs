pub mod config;
pub mod error;
pub mod state;
pub mod time;
pub mod compiled_model;
pub mod periodic_bspline;  // gh#59 v2 — de Boor evaluator
pub mod propensity;
pub mod resolved_expr;
pub mod eval_stats;
pub mod output;
pub mod rng;
pub mod ode_integrator;
pub mod gillespie;
pub mod tau_leap;
pub mod chain_binomial;
pub mod ode;
pub mod intervention;
pub mod simulate;
pub mod transition_diagnostics;
pub mod inference;

pub use config::{GillespieConfig, TauLeapConfig, ChainBinomialConfig, OdeConfig, SimConfig};
pub use error::{SimError, CollapseKind, NegativeCountCause};
pub use state::{IntState, RealState, FlowVec, Snapshot, Trajectory};
pub use compiled_model::CompiledModel;
pub use simulate::Simulate;
pub use gillespie::GillespieSim;
pub use tau_leap::TauLeapSim;
pub use chain_binomial::ChainBinomialSim;
pub use ode::OdeSim;
pub use transition_diagnostics::{TransitionDiagnostics, write_tsv as write_diagnostics_tsv, warn_zero_firings};

// ── Backend capability constraints ────────────────────────────────────────

bitflags::bitflags! {
    /// Model features that constrain which backends can run a model.
    /// The `CompiledModel` declares what it requires; each backend declares
    /// what it provides.  Mismatch → hard error at dispatch time.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Capabilities: u32 {
        /// Transitions with `overdispersion` (NegBinomial draws).
        /// Supported by tau-leap and chain-binomial, not Gillespie or ODE.
        const OVERDISPERSION    = 1 << 0;
        /// Real-valued compartments with explicit ODE equations (PDMP).
        const REAL_COMPARTMENTS = 1 << 1;
        /// gh#audit-C3. `balance { ... }` compartment-residual fix.
        /// Defined as "the residual compartment after all transitions
        /// and events have fired"; the firing semantics differ across
        /// backends (Gillespie has no substep, ODE conserves
        /// algebraically), so balance is a chain-binomial-only
        /// construct rather than a portable feature. Previously
        /// silently dropped on tau_leap / gillespie / ode — a
        /// model with `balance{}` produced a different trajectory on
        /// each backend with no warning.
        const BALANCE           = 1 << 2;
        /// Individual-sampling / lineage tracking (2026-05-19 proposal).
        /// Declared by backends that can attach a `TransitionObserver`
        /// to the event loop and emit a line list. Gillespie, tau-leap,
        /// and chain-binomial declare it; ODE does not (no individuals).
        ///
        /// Unlike OVERDISPERSION / REAL_COMPARTMENTS / BALANCE, this is
        /// NOT auto-derived by `CompiledModel::required_capabilities()`
        /// from the IR — a model carrying `#[lineage]` annotations still
        /// runs identically with or without tracking. The requirement is
        /// raised only when `--lineages` is explicitly requested, so the
        /// CLI checks `backend.capabilities().contains(LINEAGES)` at the
        /// point of the request rather than via the IR scan.
        const LINEAGES          = 1 << 3;
    }
}

pub mod lineage;
