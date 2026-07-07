use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde::ser::SerializeMap;
use serde::de::{MapAccess, Visitor};
use std::fmt;
use crate::expr::Expr;
use crate::deriv::DerivEntry;

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
#[derive(Debug, Clone, Default, PartialEq)]
pub enum DrawMethod {
    /// Standard Poisson draw: count ~ Poisson(rate × dt).
    /// Default for all transitions.
    #[default]
    Poisson,
    /// Multiplicative Gamma-Poisson (He et al. 2010):
    /// G ~ Gamma(dt/σ², σ²/dt), count ~ Poisson(rate × G × dt).
    /// Var[count] = mean + mean² · σ²/dt (quadratic scaling).
    ///
    /// `sigma_sq_grad` is the `∂σ²/∂param` map for each estimated parameter
    /// (empty ⇒ not computed; absent key ⇒ genuine zero), carried alongside its
    /// expression so a derivative can never be written without a slot for it (the
    /// `Diffable` principle, proposal §4.1). Mirrors the transition `rate_grad`
    /// but for the overdispersion argument.
    Overdispersed { sigma_sq: Expr, sigma_sq_grad: HashMap<String, DerivEntry> },
    /// Deterministic rounding: count = nearbyint(rate × dt).
    /// Used for demographic flows where Poisson noise is unphysical
    /// (e.g., constant immigration into a large population).
    Deterministic,
}

// Hand-written serde for `DrawMethod` (the derive can't express the byte-stable
// shape). `Poisson`/`Deterministic` are bare strings; `Overdispersed` keeps the
// legacy `{"overdispersed": <σ² expr>}` object — the σ² value stays the bare
// expression so existing goldens are byte-identical — and carries its gradient as
// an *adjacent sibling key* `"overdispersed_grad"` that appears only when
// non-empty. Distinct keys (no shape-sniffing) keep the OCaml↔Rust round-trip
// unambiguous.
impl Serialize for DrawMethod {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            DrawMethod::Poisson => s.serialize_str("poisson"),
            DrawMethod::Deterministic => s.serialize_str("deterministic"),
            DrawMethod::Overdispersed { sigma_sq, sigma_sq_grad } => {
                let n = if sigma_sq_grad.is_empty() { 1 } else { 2 };
                let mut m = s.serialize_map(Some(n))?;
                m.serialize_entry("overdispersed", sigma_sq)?;
                if !sigma_sq_grad.is_empty() {
                    m.serialize_entry("overdispersed_grad", sigma_sq_grad)?;
                }
                m.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for DrawMethod {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct DrawMethodVisitor;
        impl<'de> Visitor<'de> for DrawMethodVisitor {
            type Value = DrawMethod;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("\"poisson\", \"deterministic\", or an overdispersed object")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<DrawMethod, E> {
                match v {
                    "poisson" => Ok(DrawMethod::Poisson),
                    "deterministic" => Ok(DrawMethod::Deterministic),
                    other => Err(E::custom(format!(
                        "unknown draw_method \"{other}\" (expected \"poisson\" or \"deterministic\")"
                    ))),
                }
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<DrawMethod, A::Error> {
                let mut sigma_sq: Option<Expr> = None;
                let mut sigma_sq_grad: HashMap<String, DerivEntry> = HashMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "overdispersed" => sigma_sq = Some(map.next_value()?),
                        "overdispersed_grad" => sigma_sq_grad = map.next_value()?,
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unexpected draw_method key \"{other}\""
                            )))
                        }
                    }
                }
                match sigma_sq {
                    Some(sigma_sq) => Ok(DrawMethod::Overdispersed { sigma_sq, sigma_sq_grad }),
                    None => Err(serde::de::Error::custom(
                        "overdispersed draw_method missing \"overdispersed\" key",
                    )),
                }
            }
        }
        d.deserialize_any(DrawMethodVisitor)
    }
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
    /// ∂rate/∂param for each estimated parameter, classified `Grad | Unsupported`
    /// (the obs analogue via [`crate::deriv::GradMap`]). Populated by the OCaml
    /// autodiff pass; empty (and omitted) if not computed, absent key ⇒ genuine
    /// zero. A live-but-omitted coefficient (Periodic/`lag`/non-const table index)
    /// serialises an `Unsupported` the fit-time gate refuses on — subsuming the
    /// old `coeff_guard` (gh#342). A structural coefficient is still an E600 at
    /// compile time (never reaches here).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub rate_grad:      crate::deriv::ParamGradMap,
    /// ∂rate/∂compartment for each compartment the rate depends on, classified
    /// `Grad | Unsupported` (`J_x`'s ingredient for the ODE forward sensitivities,
    /// gh#275). The `rate_grad` sibling, but keyed by **compartment** — hence the
    /// distinct [`crate::deriv::CompGradMap`] type, which cannot be resolved by the
    /// parameter resolver. Populated by the OCaml `WrtPop` autodiff pass; empty
    /// (and omitted) until then and for gradient-free backends, absent key ⇒
    /// genuine zero.
    #[serde(default, skip_serializing_if = "crate::deriv::CompGradMap::is_empty")]
    pub rate_state_grad: crate::deriv::CompGradMap,
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

#[cfg(test)]
mod draw_method_tests {
    use super::*;
    use crate::deriv::{DerivEntry, UnsupportedReason};
    use crate::expr::Expr;

    /// Empty `sigma_sq_grad` ⇒ the legacy byte-stable shape `{"overdispersed": <expr>}`
    /// (the σ² value stays the bare expression, so existing goldens don't move).
    #[test]
    fn overdispersed_empty_grad_is_byte_stable() {
        let dm = DrawMethod::Overdispersed {
            sigma_sq: Expr::param("sigma_se"),
            sigma_sq_grad: HashMap::new(),
        };
        assert_eq!(
            serde_json::to_string(&dm).unwrap(),
            r#"{"overdispersed":{"param":"sigma_se"}}"#
        );
        // Bare strings for the unit variants.
        assert_eq!(serde_json::to_string(&DrawMethod::Poisson).unwrap(), r#""poisson""#);
        assert_eq!(serde_json::to_string(&DrawMethod::Deterministic).unwrap(), r#""deterministic""#);
    }

    /// Non-empty grad ⇒ adjacent sibling key `overdispersed_grad`; the round-trip
    /// recovers both fields.
    #[test]
    fn overdispersed_with_grad_round_trips() {
        let mut grad = HashMap::new();
        grad.insert("k".to_string(), DerivEntry::Grad(Expr::const_(1.0)));
        let dm = DrawMethod::Overdispersed { sigma_sq: Expr::param("s"), sigma_sq_grad: grad };
        assert_eq!(
            serde_json::to_string(&dm).unwrap(),
            r#"{"overdispersed":{"param":"s"},"overdispersed_grad":{"k":{"grad":{"const":1.0}}}}"#
        );
        let back: DrawMethod = serde_json::from_str(&serde_json::to_string(&dm).unwrap()).unwrap();
        assert_eq!(dm, back);

        // An Unsupported entry survives too.
        let mut g2 = HashMap::new();
        g2.insert("p".to_string(), DerivEntry::Unsupported {
            node: "lag".into(), code: UnsupportedReason::Lag,
        });
        let dm2 = DrawMethod::Overdispersed { sigma_sq: Expr::param("s"), sigma_sq_grad: g2 };
        let back2: DrawMethod = serde_json::from_str(&serde_json::to_string(&dm2).unwrap()).unwrap();
        assert_eq!(dm2, back2);
    }

    #[test]
    fn draw_method_units_round_trip() {
        for dm in [DrawMethod::Poisson, DrawMethod::Deterministic] {
            let back: DrawMethod = serde_json::from_str(&serde_json::to_string(&dm).unwrap()).unwrap();
            assert_eq!(dm, back);
        }
    }
}
