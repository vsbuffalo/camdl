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

use std::collections::HashMap;
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
    /// gh#275: a nonsmooth function of state — `floor`/`ceil`/`abs`/`min`/`max` of
    /// a compartment — reached while differentiating a rate w.r.t. state (WrtPop).
    /// Its state derivative is not smooth, so a gradient method (ODE-NUTS) is
    /// refused. (Append-only, tier 2b: forward sim / IF2 / PF are unaffected.)
    NonsmoothState,
    /// The parameter reaches a `zero_inflated_neg_binomial` likelihood argument.
    /// That family is scoring-only (its mixture log-PMF carries no emitted
    /// gradient), so a gradient method is refused. Detected at the fit gate by
    /// scanning the argument exprs directly — the family has no `Diffable` field
    /// to carry a per-param `Unsupported` marker, so it is never serialized into
    /// the IR (append-only, tier 2b: forward sim / IF2 / PF / MH / PMMH score it).
    ZeroInflated,
}

impl UnsupportedReason {
    /// The human-readable explanation, **derived** from the stable code — never
    /// hashed, never serialized (§4.1). The fit-time preflight (proposal §4.4)
    /// surfaces this as the refusal reason, so a copy-edit here changes only the
    /// user message, never a golden or a run_id. Each clause completes the
    /// sentence "parameter `X` …".
    pub fn reason_message(self) -> &'static str {
        match self {
            UnsupportedReason::Lag =>
                "drives a forcing's evaluation-time shift (`lag`), whose derivative \
                 the compiler does not emit (gh#314)",
            UnsupportedReason::PeriodicCoeff =>
                "drives a Periodic forcing's step value or period, whose derivative \
                 the compiler does not emit (gh#215)",
            UnsupportedReason::StructuralForcing =>
                "drives a structural forcing coefficient (a Piecewise / Interpolated / \
                 PeriodicSpline knot), precomputed at construction and not differentiable",
            UnsupportedReason::NonConstTableIndex =>
                "selects an inline-table value through a non-constant index, whose \
                 derivative the compiler does not emit",
            UnsupportedReason::Mod =>
                "enters through `mod`, which is not differentiable",
            UnsupportedReason::ParametricN =>
                "reaches a Binomial/BetaBinomial `n`, which must be θ-independent — a \
                 constant or an observed data column (it is rounded to an integer)",
            UnsupportedReason::NonsmoothState =>
                "reaches a nonsmooth function of state (`floor`, `ceil`, `abs`, `min`, \
                 or `max` of a compartment); its state derivative is not smooth, so a \
                 gradient method cannot use it — reformulate with a smooth expression",
            UnsupportedReason::ZeroInflated =>
                "reaches a `zero_inflated_neg_binomial` likelihood, which is scoring-only \
                 (no gradient); fit it with a gradient-free method — `mh`, `pmmh`, `if2`, \
                 or `pfilter`",
        }
    }
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

/// `String → DerivEntry` — the raw shape shared by both keyings below. An
/// **absent** key is a genuine zero; a present key is either a real gradient
/// ([`DerivEntry::Grad`]) or a coded refusal ([`DerivEntry::Unsupported`]).
pub type GradMap = HashMap<String, DerivEntry>;

/// A **parameter**-keyed gradient map — `parameter → DerivEntry`. The keying
/// `rate_grad`, `ic_grad`, and every obs/σ² gradient use; resolved by
/// `resolve_grad_map`, whose keys become model-**parameter** indices. An alias,
/// not a newtype: all parameter-keyed maps share one resolver and one index
/// space, so there is no illegal state among them to prevent.
pub type ParamGradMap = GradMap;

/// A **compartment**-keyed gradient map — `compartment → DerivEntry` — carrying
/// `∂rate/∂compartment` (`rate_state_grad`, the `J_x` ingredient, gh#275). A
/// **newtype**, not an alias, and deliberately so: it must NOT be resolved by
/// `resolve_grad_map`, whose keys are looked up as *parameters*. A compartment
/// map needs the compartment-index resolver; the distinct type makes reaching for
/// the wrong resolver a **compile error** — the illegal state being that a
/// compartment map silently mis-indexes through the parameter path. Serialises
/// transparently (as the bare inner map), so the wire shape matches `rate_grad`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompGradMap(pub GradMap);

impl CompGradMap {
    /// Empty ⇒ the compiler emitted no state gradient for this transition (the
    /// pre-WrtPop state and the genuine all-zero case both). Drives
    /// `skip_serializing_if`, so an empty map is omitted and golden-neutral.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, String, DerivEntry> {
        self.0.iter()
    }
}

/// A differentiable position: an expression paired with its per-parameter
/// classified gradient. Bundling the two into one value means a derivative can
/// never be written without a slot for its expression, and a new position cannot
/// be added as a bare [`Expr`] that some passes differentiate and others miss
/// (proposal `2026-07-06-seal-differentiation-coverage-3b.md` §4.1).
///
/// Serialises as the nested shape `{"expr": <expr>}` (grad omitted when empty) or
/// `{"expr": <expr>, "grad": {<param>: <DerivEntry>}}`, so a likelihood field
/// `mean: Diffable` yields `{"mean": {"expr": …, "grad": …}}` on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diffable {
    pub expr: Expr,
    #[serde(default, skip_serializing_if = "GradMap::is_empty")]
    pub grad: GradMap,
    /// `∂expr/∂projected` — the observation FACTOR-2 chain ingredient (gh#275):
    /// `None` = a genuine zero (this argument does not read the projection
    /// output), `Some(Grad e)` = the derivative w.r.t. the projected value,
    /// `Some(Unsupported)` = a nonsmooth-of-projection refusal the ODE-gradient
    /// gate consumes. Sibling of `grad` (`∂expr/∂θ`); the two are the orthogonal
    /// factors of the same argument along θ and along the projection output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proj_grad: Option<DerivEntry>,
}

impl Diffable {
    /// A position with no computed gradient yet (empty map = all genuine zeros).
    pub fn new(expr: Expr) -> Self {
        Diffable { expr, grad: GradMap::new(), proj_grad: None }
    }
}

/// Every differentiable position of a value, in declaration order, as
/// `(field_name, &Diffable)`. Generated by `#[derive(Differentiate)]`
/// (`differentiate-derive`): a new [`Diffable`] field enters this list — and thus
/// every consumer that iterates it (the fit-time preflight scan, the run_id hash)
/// — automatically. A non-`Diffable` field must carry `#[differentiate(skip)]`
/// (e.g. a Binomial `n`); an unskipped non-`Diffable` field is a compile error,
/// so a mistyped new argument can never be silently dropped. [`Expr`] therefore
/// deliberately does **not** implement this trait.
///
/// The seal, demonstrated (gh#342 P5). The derive folds a `Diffable` field and
/// skips a `#[differentiate(skip)]` one — this compiles and `diffables()` reports
/// exactly the differentiable position:
///
/// ```
/// // The items sit at the doctest crate root (explicit `fn main` stops rustdoc
/// // from wrapping them in a function) so the derive's `crate::Diffable` /
/// // `crate::Differentiable` paths resolve — the derive is `ir`-internal, and
/// // this shim lets it expand in an external doctest crate.
/// pub use ir::Differentiate;
/// pub use ir::deriv::{Diffable, Differentiable};
///
/// #[derive(Differentiate)]
/// struct Likelihoodish {
///     mean: Diffable,
///     #[differentiate(skip)]
///     n: ir::expr::Expr, // θ-independent, carries no gradient
/// }
///
/// fn main() {
///     let l = Likelihoodish {
///         mean: Diffable::new(ir::expr::Expr::const_(1.0)),
///         n: ir::expr::Expr::const_(3.0),
///     };
///     let names: Vec<_> = l.diffables().into_iter().map(|(name, _)| name).collect();
///     assert_eq!(names, vec!["mean"]); // `n` is skipped; `mean` is auto-included
/// }
/// ```
///
/// Drop the `skip` — an unskipped non-`Diffable` field — and the SAME code no
/// longer compiles, because the derive emits `&self.n` into a `&Diffable` slot.
/// A new argument accidentally typed `Expr` rather than `Diffable` is rejected
/// loudly, never silently dropped. (The passing case above shares the identical
/// crate-root shim, so this failure is attributable to the field, not the paths.)
///
/// ```compile_fail
/// pub use ir::Differentiate;
/// pub use ir::deriv::{Diffable, Differentiable};
///
/// #[derive(Differentiate)]
/// struct Leaky {
///     mean: Diffable,
///     n: ir::expr::Expr, // NOT skipped, NOT `Diffable` → the derive rejects it
/// }
///
/// fn main() {}
/// ```
pub trait Differentiable {
    fn diffables(&self) -> Vec<(&'static str, &Diffable)>;
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

    /// gh#275: `CompGradMap` is `#[serde(transparent)]` over the inner map, so a
    /// one-entry compartment map serialises as the BARE object — byte-identical
    /// to `rate_grad`'s wire, NOT wrapped in a newtype array/object. Pins the
    /// transparent attr (and the round-trip) before WrtPop starts emitting
    /// `rate_state_grad`, so the cross-language wire shape is locked in advance.
    #[test]
    fn comp_grad_map_wire_shape_is_transparent() {
        let mut inner = std::collections::HashMap::new();
        inner.insert("S".to_string(), DerivEntry::Grad(Expr::param("beta")));
        let cg = CompGradMap(inner);
        assert_eq!(
            serde_json::to_string(&cg).unwrap(),
            r#"{"S":{"grad":{"param":"beta"}}}"#
        );
        let back: CompGradMap =
            serde_json::from_str(&serde_json::to_string(&cg).unwrap()).unwrap();
        assert_eq!(cg, back);
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

    /// `Diffable` serialises as the nested shape (grad omitted when empty). This
    /// is the wire a likelihood field `mean: Diffable` inherits — the obs goldens
    /// move to it in P1 (§4.1).
    #[test]
    fn diffable_wire_shape_nested_grad_skipped_when_empty() {
        let empty = Diffable::new(Expr::param("mu"));
        assert_eq!(
            serde_json::to_string(&empty).unwrap(),
            r#"{"expr":{"param":"mu"}}"#
        );

        let mut grad = GradMap::new();
        grad.insert("beta".into(), DerivEntry::Grad(Expr::const_(1.0)));
        let d = Diffable { expr: Expr::param("mu"), grad, proj_grad: None };
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            r#"{"expr":{"param":"mu"},"grad":{"beta":{"grad":{"const":1.0}}}}"#
        );

        let back: Diffable = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(d, back);
    }

    /// The derive folds every `Diffable` field into `diffables()`, in declaration
    /// order, and skips a `#[differentiate(skip)]` field — the coverage-by-type
    /// guarantee. (A non-`Diffable` field *without* `skip` would fail to compile,
    /// which is the seal; that negative is a P5 compile-fail test.)
    #[test]
    fn derive_folds_diffable_fields_and_honors_skip() {
        #[derive(crate::Differentiate)]
        struct Fake {
            mean: Diffable,
            #[differentiate(skip)]
            #[allow(dead_code)]
            n: Expr,
            dispersion: Diffable,
        }

        let f = Fake {
            mean: Diffable::new(Expr::param("mu")),
            n: Expr::const_(100.0),
            dispersion: Diffable::new(Expr::param("phi")),
        };
        let names: Vec<&str> = f.diffables().into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["mean", "dispersion"], "folds Diffable fields in order, skips `n`");
    }

    /// The enum delegate forwards to the active variant's `diffables()`.
    #[test]
    fn derive_enum_delegates_to_active_variant() {
        #[derive(crate::Differentiate)]
        #[allow(dead_code)]
        struct A {
            x: Diffable,
        }
        #[derive(crate::Differentiate)]
        struct B {
            y: Diffable,
            z: Diffable,
        }
        #[derive(crate::Differentiate)]
        #[allow(dead_code)]
        enum E {
            Va(A),
            Vb(B),
        }

        let e = E::Vb(B { y: Diffable::new(Expr::param("y")), z: Diffable::new(Expr::param("z")) });
        let names: Vec<&str> = e.diffables().into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["y", "z"]);
    }
}
