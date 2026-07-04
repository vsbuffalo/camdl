//! Derivative entries carried across the IR — the observation/σ² analogue of
//! the transition `rate_grad`, but classified so a live-but-omitted coefficient
//! never masquerades as a genuine zero.
//!
//! The OCaml compiler's autodiff pass (`autodiff.ml`) classifies each
//! differentiable position as `Known | Omitted | Unsupported`. The obs/σ² driver
//! collapses that to the two states the IR needs to carry — a real gradient
//! expression ([`DerivEntry::Grad`]) or a loud, coded refusal
//! ([`DerivEntry::Unsupported`]) — so the fit-time gate consumes the reason
//! rather than re-deriving coverage (proposal
//! `2026-07-03-unified-obs-gradient-autodiff.md` §4.1). An **absent** map key is
//! a genuine zero.
//!
//! Unlike the rate path (a rate `Unsupported` is a compile-time E600, so it never
//! reaches the IR), an obs/σ² `Unsupported` is *serialized* into the IR and hashed
//! into run_id identity. The refusal therefore carries a stable enum
//! [`UnsupportedReason`] **code** — hashed, and the only part of the entry that
//! is — while the human-readable label (`node`) and message are derived for
//! display, so a message copy-edit cannot re-key run_id.

use serde::{Deserialize, Serialize};
use crate::expr::Expr;

/// Why a parameter's derivative through a differentiable position could not be
/// emitted. A STABLE, hashed code — the human-readable message is derived from
/// it at display time (§4.1), so a copy-edit never re-keys run_id.
///
/// New reasons append at the END: the run_id hash (`runid::ir_hash`) tags
/// variants by position, so declaration order == hash index, and that index is
/// permanent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedReason {
    /// The parameter drives a forcing's evaluation-time shift (`lag`, gh#314);
    /// the closed forms differentiate against bare `Time`, so the derivative is
    /// deliberately un-emitted (tier 2b).
    Lag,
    /// A Periodic forcing's step value or period — a live coefficient whose
    /// derivative is deliberately not emitted (tier 2b).
    PeriodicCoeff,
    /// A structural forcing coefficient a parameter cannot drive: Piecewise /
    /// Interpolated / PeriodicSpline knots, precomputed at construction (tier 3).
    StructuralForcing,
    /// An inline-table value reached by a non-constant index (tier 2b/3).
    NonConstTableIndex,
    /// The parameter enters through `mod`, which is not differentiable.
    Mod,
    /// The parameter reaches a Binomial/BetaBinomial `n`, which is rounded to an
    /// integer and must be θ-independent.
    ParametricN,
}

/// One entry in a differentiable position's per-parameter gradient map: either a
/// real derivative expression, or a loud, coded refusal.
///
/// Serialises externally-tagged, mirroring [`crate::observation::Likelihood`]:
/// `Grad` → `{"grad": <expr>}`, `Unsupported` →
/// `{"unsupported": {"node": "…", "code": "<reason>"}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivEntry {
    /// A real ∂arg/∂param derivative expression.
    Grad(Expr),
    /// The derivative could not be emitted. `node` is a human-readable label
    /// (display only — NOT hashed); `code` is the stable, hashed reason.
    Unsupported { node: String, code: UnsupportedReason },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    /// Pin the exact wire shape of `DerivEntry` — the OCaml `serde.ml`
    /// (`deriv_entry_to_json`) must emit byte-identical JSON (verified
    /// cross-language during P2). Externally-tagged, snake_case.
    #[test]
    fn deriv_entry_wire_shape_is_pinned() {
        let grad = DerivEntry::Grad(Expr::param("beta"));
        assert_eq!(
            serde_json::to_string(&grad).unwrap(),
            r#"{"grad":{"param":"beta"}}"#
        );

        let unsupp = DerivEntry::Unsupported {
            node: "time_func:seasonal".into(),
            code: UnsupportedReason::Lag,
        };
        assert_eq!(
            serde_json::to_string(&unsupp).unwrap(),
            r#"{"unsupported":{"node":"time_func:seasonal","code":"lag"}}"#
        );
    }

    #[test]
    fn unsupported_reason_wire_names() {
        for (r, s) in [
            (UnsupportedReason::Lag, "\"lag\""),
            (UnsupportedReason::PeriodicCoeff, "\"periodic_coeff\""),
            (UnsupportedReason::StructuralForcing, "\"structural_forcing\""),
            (UnsupportedReason::NonConstTableIndex, "\"non_const_table_index\""),
            (UnsupportedReason::Mod, "\"mod\""),
            (UnsupportedReason::ParametricN, "\"parametric_n\""),
        ] {
            assert_eq!(serde_json::to_string(&r).unwrap(), s);
        }
    }

    #[test]
    fn deriv_entry_round_trips() {
        for de in [
            DerivEntry::Grad(Expr::param("x")),
            DerivEntry::Unsupported { node: "n".into(), code: UnsupportedReason::Mod },
        ] {
            let json = serde_json::to_string(&de).unwrap();
            let back: DerivEntry = serde_json::from_str(&json).unwrap();
            assert_eq!(de, back);
        }
    }
}
