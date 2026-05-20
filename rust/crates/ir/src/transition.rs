use serde::{Deserialize, Serialize};
use crate::expr::Expr;

/// A single `(compartment_name, delta)` stoichiometry entry.
/// Serialises as a two-element JSON array: `["S", -1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoichiometryEntry(pub String, pub i64);

/// Advisory metadata — the runtime ignores this; it exists for tooling and
/// human readers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionMetadata {
    pub origin_kind:        Option<String>,
    pub source_compartment: Option<String>,
    pub dest_compartment:   Option<String>,
}

/// How event counts are drawn for this transition.
///
/// Rate wrappers (`overdispersed`, `deterministic`) are compiler-recognized
/// forms in the DSL, not general-purpose functions. They are not composable
/// — `overdispersed(deterministic(rate), σ²)` is meaningless and rejected.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawMethod {
    /// Standard Poisson draw: count ~ Poisson(rate × dt).
    /// Default for all transitions.
    #[default]
    Poisson,
    /// Multiplicative Gamma-Poisson (He et al. 2010):
    /// G ~ Gamma(dt/σ², σ²/dt), count ~ Poisson(rate × G × dt).
    /// Var[count] = mean + mean² · σ²/dt (quadratic scaling).
    Overdispersed(Expr),
    /// Deterministic rounding: count = nearbyint(rate × dt).
    /// Used for demographic flows where Poisson noise is unphysical
    /// (e.g., constant immigration into a large population).
    Deterministic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    pub name:           String,
    pub stoichiometry:  Vec<StoichiometryEntry>,
    pub rate:           Expr,
    pub metadata:       Option<TransitionMetadata>,
    /// How event counts are drawn. Defaults to Poisson.
    #[serde(default, skip_serializing_if = "is_poisson")]
    pub draw_method:    DrawMethod,
    /// ∂rate/∂param for each estimated parameter. Populated by the OCaml
    /// compiler's autodiff pass. Empty if not computed (backward compatible).
    /// Maps parameter name → derivative expression.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub rate_grad:      std::collections::HashMap<String, Expr>,
    /// Lineage annotation for `#[lineage]` transitions. `None` for ordinary
    /// transitions (the common case), and omitted from the JSON then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage:        Option<TransitionLineage>,
}

fn is_poisson(m: &DrawMethod) -> bool {
    matches!(m, DrawMethod::Poisson)
}

/// Lineage (individual-sampling) annotation for a `#[lineage]` transition.
///
/// Emitted by the OCaml compiler for transitions marked `#[lineage]` that
/// pass the linear-in-parents check (2026-05-19 individual-sampling-layer
/// proposal). `None` on [`Transition::lineage`] for ordinary transitions.
///
/// `parent_pool_weights` is the linear decomposition of the rate over parent
/// pools: `(parent_compartment, per_pool_weight_expr)` pairs. For `β·S·I/N`
/// with parent `I` this is `[("I", β·S/N)]`. The runtime samples parent pool
/// `b` with probability ∝ `weight_b · count_b`, then samples uniformly within
/// the chosen pool. The weight is a frozen coefficient at the event instant
/// (normalizers like `1/N` are evaluated at the current state), so it does
/// not itself depend linearly on the parent count — that dependence has been
/// factored out into the per-pool entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionLineage {
    pub is_lineage_event:    bool,
    /// `(compartment, weight_expr)` pairs. Serialised as a JSON array of
    /// two-element `[name, expr]` arrays to mirror the OCaml side.
    pub parent_pool_weights: Vec<(String, Expr)>,
}
