use serde::{Deserialize, Serialize};

// ── Operators ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Min,
    Max,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Sin,    // gh#58
    Cos,    // gh#58
    Tanh,   // gh#58
}

// ── Inner structs for compound variants ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinOpExpr {
    pub op:    BinOp,
    pub left:  Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnOpExpr {
    pub op:  UnOp,
    pub arg: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CondExpr {
    pub pred: Box<Expr>,
    pub then: Box<Expr>,
    #[serde(rename = "else")]
    pub else_: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeFuncRef {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableLookupExpr {
    pub table:   String,
    pub indices: Vec<Expr>,
}

// ── Wrapper structs (each has one uniquely-named field → untagged works) ──────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstExpr {
    #[serde(rename = "const")]
    pub value: f64,
}

// Bitwise PartialEq/Eq so that `Expr::Const(NaN) == Expr::Const(NaN)` (when bit
// patterns match) and `Const(0.0) != Const(-0.0)`. Derived PartialEq would
// inherit IEEE 754 float semantics (NaN != NaN, 0.0 == -0.0), neither of which
// is correct for IR-equality purposes — two ASTs that differ only in NaN
// payload or zero sign should be observably distinct.
impl PartialEq for ConstExpr {
    fn eq(&self, other: &Self) -> bool {
        self.value.to_bits() == other.value.to_bits()
    }
}
impl Eq for ConstExpr {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamExpr {
    pub param: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopExpr {
    pub pop: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopSumExpr {
    pub pop_sum: Vec<String>,
}

/// `{"reduce": [<expr>, …]}` — n-ary sum over already-substituted terms (Fix D,
/// proposal 2026-05-29-shared-bindings-and-reduction). Replaces the deep
/// left-nested `BinOp(Add)` chain that `sum(...)` over a dimension lowered to,
/// which tripped serde's recursion limit past ~50 patches. Evaluated as a
/// left-fold to match the OCaml `List.fold_left (+)` order bit-for-bit. Sum
/// semantics only; product is deferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReduceWrap {
    pub reduce: Vec<Expr>,
}

/// `{"binding_ref": "<name>"}` — reference to a model-level `Binding` by name
/// (Fix B). Resolved to a slot at `CompiledModel::new` (like `Param`/`Pop`) and
/// evaluated on-demand from the binding's body. Hoisted FOI aggregates (N[l],
/// I_agg[l], spatial force F[l]) are defined once in `Model.bindings` instead of
/// being inlined into every (patch,age) rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingRefWrap {
    pub binding_ref: String,
}

/// `{"per_eval_ref": "<name>"}` — reference to a model-level `per_eval_binding`
/// by name (gh#272 LICM). Like `BindingRefWrap`, but the body is param/table-only
/// (loop-invariant within a trajectory) and may be param-carrying, so it is cached
/// once per θ-stable scope rather than per step. Produced by the LICM pass
/// (on by default); absent only when LICM is disabled (`--no-licm`) or the model
/// has nothing hoistable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerEvalRefWrap {
    pub per_eval_ref: String,
}

/// `{"time": null}` — unit value serialises to JSON null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeExpr {
    pub time: (),
}

/// `{"dt": null}` — runtime integrator step. Has dimension `T` (same
/// as `time`). Evaluator reads from `EvalCtx.dt` (populated from
/// `SMCConfig.dt` or backend `cfg.dt` at substep level). gh#54.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtExpr {
    pub dt: (),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinOpWrap {
    pub bin_op: BinOpExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnOpWrap {
    pub un_op: UnOpExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CondWrap {
    pub cond: CondExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeFuncWrap {
    pub time_func: TimeFuncRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableLookupWrap {
    pub table_lookup: TableLookupExpr,
}

/// `{"projected": null}` — used inside likelihood expressions to reference the
/// projection output.  Only valid in observation model likelihood fields; the
/// validator will flag it elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedExpr {
    pub projected: (),
}

/// `{"obs_column_ref": "<col>"}` — a per-observation auxiliary data column
/// referenced by name in a likelihood (e.g. binomial `n = tested`). Only valid
/// in observation-model likelihood fields; the Rust binder resolves the name
/// against the enclosing stream's bound aux columns and fills its value per
/// observation. 2026-06-10 observation data-entry §3, §6.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsColumnRefExpr {
    pub obs_column_ref: String,
}

/// Per-expression dimensional escape. Asserts that the wrapped
/// subexpression has dimension `(dim_p, dim_t)` without the
/// dim-checker verifying — the programmer takes responsibility.
/// Runtime semantics: identity — the evaluator unwraps `inner` and
/// returns its value. See
/// `docs/dev/proposals/notes/unchecked-dim-escape.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncheckedDimExpr {
    pub inner:  Box<Expr>,
    pub dim:    (i32, i32),
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncheckedDimWrap {
    pub unchecked_dim: UncheckedDimExpr,
}

// ── Expression ────────────────────────────────────────────────────────────────

/// Pure, total, first-order expression language.  Each variant serialises to
/// a JSON object whose sole key unambiguously identifies the variant.
///
/// **Serialization** uses `#[serde(untagged)]`, which for a newtype-variant
/// enum simply emits the inner wrapper — i.e. the single-key object
/// (`{"const": …}`, `{"bin_op": …}`).
///
/// **Deserialization** is hand-written below (see `impl Deserialize`), *not*
/// derived. Derived `untagged` deserialization buffers every node into an
/// owned `serde::private::de::Content` value and trial-deserializes each
/// variant in turn (clone + drop per node) — pathological for a deeply
/// recursive AST: profiling a 2 GB IR showed ~50% of `simulate` wall time in
/// `content_clone`/`Content` drop/malloc. The single map key already names the
/// variant, so the manual impl dispatches on it in one streaming pass with no
/// buffering. The emitted JSON is unchanged (golden round-trip tests pin it).
/// See docs/dev/notes/2026-05-29-foi-scaling-bench.md (Fix E).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Expr {
    Const(ConstExpr),
    Param(ParamExpr),
    Pop(PopExpr),
    PopSum(PopSumExpr),
    Time(TimeExpr),
    Dt(DtExpr),
    BinOp(BinOpWrap),
    UnOp(UnOpWrap),
    Cond(CondWrap),
    TimeFunc(TimeFuncWrap),
    TableLookup(TableLookupWrap),
    Projected(ProjectedExpr),
    UncheckedDim(UncheckedDimWrap),
    Reduce(ReduceWrap),
    BindingRef(BindingRefWrap),
    PerEvalRef(PerEvalRefWrap),
    ObsColumnRef(ObsColumnRefExpr),
}

// ── Convenience constructors ──────────────────────────────────────────────────

impl Expr {
    pub fn const_(v: f64) -> Self {
        Expr::Const(ConstExpr { value: v })
    }
    pub fn param(name: impl Into<String>) -> Self {
        Expr::Param(ParamExpr { param: name.into() })
    }
    pub fn pop(name: impl Into<String>) -> Self {
        Expr::Pop(PopExpr { pop: name.into() })
    }
    pub fn pop_sum(names: Vec<String>) -> Self {
        Expr::PopSum(PopSumExpr { pop_sum: names })
    }
    pub fn time() -> Self {
        Expr::Time(TimeExpr { time: () })
    }
    pub fn dt() -> Self {
        Expr::Dt(DtExpr { dt: () })
    }
    pub fn bin_op(op: BinOp, left: Expr, right: Expr) -> Self {
        Expr::BinOp(BinOpWrap {
            bin_op: BinOpExpr { op, left: Box::new(left), right: Box::new(right) },
        })
    }
    pub fn un_op(op: UnOp, arg: Expr) -> Self {
        Expr::UnOp(UnOpWrap {
            un_op: UnOpExpr { op, arg: Box::new(arg) },
        })
    }
    pub fn reduce(terms: Vec<Expr>) -> Self {
        Expr::Reduce(ReduceWrap { reduce: terms })
    }
    pub fn binding_ref(name: impl Into<String>) -> Self {
        Expr::BindingRef(BindingRefWrap { binding_ref: name.into() })
    }
    pub fn per_eval_ref(name: impl Into<String>) -> Self {
        Expr::PerEvalRef(PerEvalRefWrap { per_eval_ref: name.into() })
    }
    pub fn obs_column_ref(name: impl Into<String>) -> Self {
        Expr::ObsColumnRef(ObsColumnRefExpr { obs_column_ref: name.into() })
    }
}

// ── Hand-written Deserialize (single-pass, no Content buffering) ──────────────
//
// Replaces the derived `#[serde(untagged)]` deserialization. Every `Expr` node
// is a single-key object whose key names the variant, so we read that one key
// and dispatch — no per-node buffer-allocate/clone/drop. See the doc comment on
// `Expr` and docs/dev/notes/2026-05-29-foi-scaling-bench.md (Fix E).
//
// MAINTENANCE: adding or renaming an `Expr` variant now requires updating the
// match arm below *in addition to* the enum (Serialize is still derived). This
// is one extra same-file edit on top of the cross-language IR changes a new
// variant already needs (OCaml `ir.ml` + `ir/schema.json`). The omission is
// loud, not silent: a missing arm makes that node's key fall to the `other =>`
// hard error, and the `roundtrips_every_variant_*` test fails (Serialize emits
// a key the Deserialize then rejects). A future externally-tagged refactor
// (rename variants, drop the wrapper structs) would let the derive handle both
// directions again and remove this second location — see the note's Fix E.
impl<'de> Deserialize<'de> for Expr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct ExprVisitor;

        impl<'de> Visitor<'de> for ExprVisitor {
            type Value = Expr;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a camdl expression node: a single-key JSON object whose key names \
                     the node kind (e.g. \"const\", \"pop\", \"bin_op\")",
                )
            }

            fn visit_map<A>(self, mut map: A) -> Result<Expr, A::Error>
            where
                A: MapAccess<'de>,
            {
                let key: String = map.next_key()?.ok_or_else(|| {
                    de::Error::custom(
                        "expected a single-key expression object, found an empty object",
                    )
                })?;

                // `()`-valued variants (time/dt/projected) serialise as null;
                // deserialising into `()` enforces that the value is null.
                let expr = match key.as_str() {
                    "const" => Expr::Const(ConstExpr { value: map.next_value()? }),
                    "param" => Expr::Param(ParamExpr { param: map.next_value()? }),
                    "pop" => Expr::Pop(PopExpr { pop: map.next_value()? }),
                    "pop_sum" => Expr::PopSum(PopSumExpr { pop_sum: map.next_value()? }),
                    "time" => {
                        map.next_value::<()>()?;
                        Expr::Time(TimeExpr { time: () })
                    }
                    "dt" => {
                        map.next_value::<()>()?;
                        Expr::Dt(DtExpr { dt: () })
                    }
                    "bin_op" => Expr::BinOp(BinOpWrap { bin_op: map.next_value()? }),
                    "un_op" => Expr::UnOp(UnOpWrap { un_op: map.next_value()? }),
                    "cond" => Expr::Cond(CondWrap { cond: map.next_value()? }),
                    "time_func" => {
                        Expr::TimeFunc(TimeFuncWrap { time_func: map.next_value()? })
                    }
                    "table_lookup" => {
                        Expr::TableLookup(TableLookupWrap { table_lookup: map.next_value()? })
                    }
                    "projected" => {
                        map.next_value::<()>()?;
                        Expr::Projected(ProjectedExpr { projected: () })
                    }
                    "unchecked_dim" => {
                        Expr::UncheckedDim(UncheckedDimWrap { unchecked_dim: map.next_value()? })
                    }
                    "reduce" => Expr::Reduce(ReduceWrap { reduce: map.next_value()? }),
                    "binding_ref" => Expr::BindingRef(BindingRefWrap { binding_ref: map.next_value()? }),
                    "per_eval_ref" => Expr::PerEvalRef(PerEvalRefWrap { per_eval_ref: map.next_value()? }),
                    "obs_column_ref" => {
                        Expr::ObsColumnRef(ObsColumnRefExpr { obs_column_ref: map.next_value()? })
                    }
                    other => {
                        return Err(de::Error::custom(format!(
                            "unknown expression node kind '{other}' (expected one of: const, \
                             param, pop, pop_sum, time, dt, bin_op, un_op, cond, time_func, \
                             table_lookup, projected, unchecked_dim, reduce, binding_ref, \
                             per_eval_ref, obs_column_ref)"
                        )))
                    }
                };

                // Single-key invariant: a second key means malformed IR. Reject
                // rather than silently ignore it ("no loose semantics").
                if let Some(extra) = map.next_key::<String>()? {
                    return Err(de::Error::custom(format!(
                        "expression node '{key}' has an unexpected extra key '{extra}'; \
                         each expression object must have exactly one key"
                    )));
                }

                Ok(expr)
            }
        }

        deserializer.deserialize_map(ExprVisitor)
    }
}

#[cfg(test)]
mod deserialize_tests {
    use super::*;

    fn roundtrip(e: &Expr) {
        let json = serde_json::to_string(e).expect("serialize");
        let back: Expr = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*e, back, "round-trip changed value; json was {json}");
    }

    #[test]
    fn roundtrips_every_variant_and_a_deep_nesting() {
        // Each leaf variant.
        for e in [
            Expr::const_(1.5),
            Expr::const_(-0.0),
            Expr::param("beta"),
            Expr::pop("S_patch1_age_0_4"),
            Expr::pop_sum(vec!["S".into(), "E".into(), "I".into()]),
            Expr::time(),
            Expr::dt(),
            Expr::Projected(ProjectedExpr { projected: () }),
            Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: "school".into() } }),
            Expr::TableLookup(TableLookupWrap {
                table_lookup: TableLookupExpr { table: "W".into(), indices: vec![Expr::const_(3.0)] },
            }),
            Expr::reduce(vec![Expr::const_(1.0), Expr::param("kappa"), Expr::pop("I_p1")]),
            Expr::reduce(vec![]), // empty sum (= 0)
            Expr::binding_ref("N_patch1"),
            Expr::obs_column_ref("tested"),
            Expr::per_eval_ref("__licm_0"), // gh#272 LICM variant (gh#284: close the hole)
        ] {
            roundtrip(&e);
        }

        // A deep tree exercising every compound variant (BinOp/UnOp/Cond/
        // UncheckedDim) and recursion through Box<Expr>.
        let cond = Expr::Cond(CondWrap {
            cond: CondExpr {
                pred: Box::new(Expr::bin_op(BinOp::Gt, Expr::pop("I"), Expr::const_(0.0))),
                then: Box::new(Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I"))),
                else_: Box::new(Expr::const_(0.0)),
            },
        });
        let escaped = Expr::UncheckedDim(UncheckedDimWrap {
            unchecked_dim: UncheckedDimExpr {
                inner: Box::new(Expr::un_op(UnOp::Exp, Expr::time())),
                dim: (1, -1),
                reason: "test".into(),
            },
        });
        let tree = Expr::bin_op(BinOp::Add, cond, escaped);
        roundtrip(&tree);
    }

    #[test]
    fn deserializes_existing_json_shapes() {
        // The exact on-disk shapes the OCaml compiler emits.
        assert_eq!(
            serde_json::from_str::<Expr>(r#"{"const": 2.5}"#).unwrap(),
            Expr::const_(2.5)
        );
        assert_eq!(serde_json::from_str::<Expr>(r#"{"time": null}"#).unwrap(), Expr::time());
        assert_eq!(
            serde_json::from_str::<Expr>(r#"{"bin_op":{"op":"mul","left":{"param":"R0"},"right":{"pop":"S"}}}"#).unwrap(),
            Expr::bin_op(BinOp::Mul, Expr::param("R0"), Expr::pop("S"))
        );
    }

    #[test]
    fn rejects_malformed_nodes() {
        // Unknown kind, empty object, and the single-key-invariant violation
        // must all be hard errors (the derived untagged impl silently ignored
        // the extra key on multi-key objects; this is intentionally stricter).
        assert!(serde_json::from_str::<Expr>(r#"{"bogus": 1}"#).is_err(), "unknown key");
        assert!(serde_json::from_str::<Expr>("{}").is_err(), "empty object");
        assert!(
            serde_json::from_str::<Expr>(r#"{"const": 1.0, "param": "x"}"#).is_err(),
            "multi-key object"
        );
    }
}
