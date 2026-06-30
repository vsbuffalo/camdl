//! Counterfactual contrasts (proposal 2026-06-25): named differences of two
//! run-rooted operands — the "cases averted" object. The DSL `contrasts {}` block
//! parses, dim-checks, and lowers to these IR *definition* types in the OCaml
//! frontend; the two-arm replay reducer that evaluates them against a fit's keyed
//! `(θ, X)` output is the backend (`sim`) — stage C, not here.
//!
//! Reporting-only and non-identity: excluded from `Model::hash_into` (a contrast
//! must never re-key a sim/fit), exactly like `quantity`.

use serde::{Deserialize, Serialize};
use crate::expr::BinOp;

/// The two symmetric sub-namespaces of a run member: `<run>.quantities.<q>` and
/// `<run>.observations.<stream>`. Neither is special-cased; a quantity and a
/// stream of the same name never collide across them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunNamespace {
    Quantities,
    Observations,
}

/// A contrast body: arithmetic over run-rooted operands. A dedicated ADT outside
/// the shared rate `Expr` (the `ScalarExpr` / trigger precedent) so a run-member
/// reference can never appear in a propensity, and a rate leaf can never appear
/// in a contrast. Externally-tagged single-key objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContrastExpr {
    /// `<run>.<ns>.<member>` — a scenario (or the reserved `fitted`) crossed with
    /// a quantity / observation stream. `run` and `member` are resolved (the
    /// OCaml expander validates both before lowering); the Rust reducer evaluates
    /// the member on the named run's arm trajectory.
    RunMember {
        run: String,
        ns: RunNamespace,
        member: String,
    },
    BinOp {
        op: BinOp,
        left: Box<ContrastExpr>,
        right: Box<ContrastExpr>,
    },
}

/// A named difference of two run-rooted operands (cases averted). There is no
/// window: the counterfactual fork is *derived* in the reducer (the last saved
/// trajectory snapshot strictly before the toggled intervention's fire time) and
/// the result is shaped over `[fork, run-end]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contrast {
    pub name: String,
    pub body: ContrastExpr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_wire_tags() {
        // The exact on-wire shape the OCaml serde must match:
        // averted = no_sia.quantities.total - with_sia.quantities.total.
        let c = Contrast {
            name: "averted".into(),
            body: ContrastExpr::BinOp {
                op: BinOp::Sub,
                left: Box::new(ContrastExpr::RunMember {
                    run: "no_sia".into(),
                    ns: RunNamespace::Quantities,
                    member: "total".into(),
                }),
                right: Box::new(ContrastExpr::RunMember {
                    run: "with_sia".into(),
                    ns: RunNamespace::Quantities,
                    member: "total".into(),
                }),
            },
        };
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"name":"averted","body":{"bin_op":{"op":"sub","left":{"run_member":{"run":"no_sia","ns":"quantities","member":"total"}},"right":{"run_member":{"run":"with_sia","ns":"quantities","member":"total"}}}}}"#
        );
        let back: Contrast = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(c, back, "contrast round-trip changed value");
    }

    #[test]
    fn observation_namespace_wire() {
        assert_eq!(
            serde_json::to_string(&RunNamespace::Observations).unwrap(),
            r#""observations""#
        );
    }
}
