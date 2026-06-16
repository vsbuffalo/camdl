use serde::{Deserialize, Serialize};

// ── Prior distributions ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniformPrior   { pub lower: f64, pub upper: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalPrior    { pub mean: f64, pub sd: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogNormalPrior { pub mu: f64, pub sigma: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HalfNormalPrior { pub sigma: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BetaPrior      { pub alpha: f64, pub beta: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GammaPrior     { pub shape: f64, pub rate: f64 }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExponentialPrior { pub rate: f64 }
/// Uniform on the log scale: `log(X) ~ Uniform(log lower, log upper)`.
/// `lower, upper > 0`. The honest weakly-informative choice for a scale
/// parameter uncertain across orders of magnitude.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogUniformPrior { pub lower: f64, pub upper: f64 }
/// Normal(mean, sd) truncated to `[lower, upper]`. The truncation bounds
/// are the parameter's declared `in [lo, hi]` range (baked in by the
/// compiler so the IR stays self-contained).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruncatedNormalPrior { pub mean: f64, pub sd: f64, pub lower: f64, pub upper: f64 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorDist {
    Uniform(UniformPrior),
    Normal(NormalPrior),
    LogNormal(LogNormalPrior),
    HalfNormal(HalfNormalPrior),
    Beta(BetaPrior),
    Gamma(GammaPrior),
    Exponential(ExponentialPrior),
    LogUniform(LogUniformPrior),
    TruncatedNormal(TruncatedNormalPrior),
    Fixed(f64),
}

// ── Hierarchical priors ───────────────────────────────────────────────────────

/// Distribution family for a hierarchical (pooled) prior leaf.
///
/// Mirrors the variants of `PriorDist` except `Fixed` (which has no
/// meaning in a hierarchical context). Serializes to/from the same
/// snake_case strings used in the IR JSON ("uniform", "normal",
/// "log_normal", …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HierarchicalKind {
    Uniform,
    Normal,
    LogNormal,
    HalfNormal,
    Beta,
    Gamma,
    Exponential,
}

impl HierarchicalKind {
    /// Returns the snake_case string used in IR JSON serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Uniform     => "uniform",
            Self::Normal      => "normal",
            Self::LogNormal   => "log_normal",
            Self::HalfNormal  => "half_normal",
            Self::Beta        => "beta",
            Self::Gamma       => "gamma",
            Self::Exponential => "exponential",
        }
    }
}

impl std::fmt::Display for HierarchicalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Hierarchical prior for a leaf parameter in a pooled group (wave 2 /
/// malaria #3).
///
/// A "leaf" is a parameter whose prior references *other parameters*
/// (hyperparameters) rather than being a pure-constant distribution.
/// At inference time the hyperparameters carry their own priors and are
/// sampled jointly with the leaves; at each log-posterior evaluation
/// the `args` expressions are resolved against the current
/// hyperparameter values.
///
/// - `kind` names the distribution family. Typed enum — rejected at
///   IR deserialisation time, not at inference time.
/// - `args` are keyword → expression pairs (e.g. `"mu" → Param("mu_h")`,
///   `"sigma" → Param("sigma_h")`).
/// - `pool_over` names the dimension over which shrinkage is applied
///   (from the DSL `| age` clause). Empty string for scalar leaves
///   with hyperparent references but no pooling dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalPrior {
    pub kind:      HierarchicalKind,
    pub args:      std::collections::BTreeMap<String, crate::expr::Expr>,
    pub pool_over: String,
}

// ── Parameter transform ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Log,
    Logit,
    Identity,
}

// ── Parameter kind ──────────────────────────────────────────────────────────

/// DSL parameter-type keyword (the `param_kind` production in
/// `parser.mly`). Was `Option<String>`; the typed enum is rejected at IR
/// deserialisation rather than re-parsed at every consumer (the gh#191
/// stringly-typed surface). Each kind's dimensional meaning lives in the
/// OCaml `Dimcheck.param_dim_of_kind`; `Instant`/`Duration` are time-typed
/// (`[T]`) per the 2026-05-22 calendar-time proposal. Serialises to the same
/// snake_case strings the field has always used (`"rate"`, …), so the type
/// swap is byte-compatible with existing IR JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Rate,
    Probability,
    Count,
    Positive,
    Real,
    Instant,
    Duration,
}

impl ParamKind {
    /// The snake_case string used in IR JSON (mirrors the `serde` rename).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rate        => "rate",
            Self::Probability => "probability",
            Self::Count       => "count",
            Self::Positive    => "positive",
            Self::Real        => "real",
            Self::Instant     => "instant",
            Self::Duration    => "duration",
        }
    }
}

impl std::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Prior specification ───────────────────────────────────────────────────────

/// The prior on an *estimated* parameter. Collapses the former
/// `prior: Option<PriorDist>` + `hierarchical: Option<HierarchicalPrior>`
/// (which a comment declared "mutually exclusive") into a single slot, so
/// both-set (a leaf with two prior specs) is unrepresentable and the
/// previously-ambiguous both-`None` becomes the explicit `Flat`.
///
/// JSON: `Flat` → `"flat"`; `Dist` → `{"dist": <prior_dist>}`;
/// `Hierarchical` → `{"hierarchical": <hierarchical_prior>}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorSpec {
    /// No informative prior (flat / improper over the bounds).
    Flat,
    /// A single-level prior distribution.
    Dist(PriorDist),
    /// A hierarchical (pooled) prior leaf — its `args` reference
    /// hyperparameters resolved at inference time.
    Hierarchical(HierarchicalPrior),
}

// ── Parameter value ─────────────────────────────────────────────────────────

/// The three real meanings the former `value: Option<f64>` conflated
/// (gh#191). Inference configuration (`init`/`bounds`/`prior`/`transform`)
/// exists *only* on `Estimated`, so attaching a prior or bounds to a fixed
/// constant — or shipping a value-less parameter that no one will supply — is
/// unrepresentable.
///
/// JSON: internally tagged on `mode` —
/// `{"mode":"fixed","value":0.3}`,
/// `{"mode":"estimated","bounds":[0.01,2.0],"prior":"flat","transform":"log"}`,
/// `{"mode":"required"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ParamValue {
    /// Known at model-build time: a hand-crafted IR constant, a typed-const
    /// `let`, or an applied `--set`/`[fixed]` override. Carries no inference
    /// config.
    Fixed { value: f64 },
    /// Inference draws this. The optimiser's starting point (`init`), search
    /// box (`bounds`), `prior`, and `transform` live here and nowhere else.
    Estimated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        init: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bounds: Option<(f64, f64)>,
        prior: PriorSpec,
        transform: Transform,
    },
    /// Must be supplied at runtime (`--params`/`--set`); no default in the IR.
    /// This is the *only* state for which "parameter has no value" is correct.
    Required,
}

impl ParamValue {
    /// The number this parameter currently resolves to — the faithful drop-in
    /// for the former `value: Option<f64>`. `Fixed` → its constant;
    /// `Estimated` → its `init` (the optimiser start, filled by the fit layer
    /// or a placeholder gate — `None` until then); `Required` → `None`. A
    /// `None` here is exactly the former "parameter has no value yet".
    pub fn resolved_value(&self) -> Option<f64> {
        match self {
            ParamValue::Fixed { value } => Some(*value),
            ParamValue::Estimated { init, .. } => *init,
            ParamValue::Required => None,
        }
    }
    /// This value with a concrete number set, the drop-in for the former
    /// `p.value = Some(v)`. Used by every consumer that supplies/overrides a
    /// value: `--params`/`--set`/`[fixed]`/scenario overrides, the fit start
    /// fall-back, and the gh#191 capability gate.
    ///
    /// An `Estimated` parameter KEEPS its bounds/prior/transform (the number
    /// lands in `init`), so a supplied value is still bounds-checked against
    /// the author's declared range — collapsing it to a bare `Fixed` would
    /// drop the bounds and silently accept an out-of-range value. `Fixed` is
    /// replaced; `Required` becomes `Fixed` (it has no bounds to preserve).
    pub fn with_value(&self, v: f64) -> ParamValue {
        match self {
            ParamValue::Estimated { bounds, prior, transform, .. } => ParamValue::Estimated {
                init: Some(v),
                bounds: *bounds,
                prior: prior.clone(),
                transform: transform.clone(),
            },
            ParamValue::Fixed { .. } | ParamValue::Required => ParamValue::Fixed { value: v },
        }
    }
    /// The inference search box, if this is an `Estimated` parameter.
    pub fn bounds(&self) -> Option<(f64, f64)> {
        match self {
            ParamValue::Estimated { bounds, .. } => *bounds,
            _ => None,
        }
    }
    /// The optimiser's starting point (former `initial_value`), if `Estimated`.
    pub fn init(&self) -> Option<f64> {
        match self {
            ParamValue::Estimated { init, .. } => *init,
            _ => None,
        }
    }
    /// The transform, if `Estimated`.
    pub fn transform(&self) -> Option<Transform> {
        match self {
            ParamValue::Estimated { transform, .. } => Some(transform.clone()),
            _ => None,
        }
    }
    /// The prior spec, if `Estimated`.
    pub fn prior(&self) -> Option<&PriorSpec> {
        match self {
            ParamValue::Estimated { prior, .. } => Some(prior),
            _ => None,
        }
    }
}

// ── Parameter declaration ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name:          String,
    /// What kind of value this parameter has — see [`ParamValue`]. Inference
    /// config (bounds/prior/transform/init) lives inside `Estimated`.
    pub value:         ParamValue,
    /// DSL parameter-type keyword (typed enum; see [`ParamKind`]).
    /// Used by inference to choose the default transform. `None` = no
    /// annotation (dimension inferred by the compiler).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_kind:    Option<ParamKind>,
    /// Explicit dimension annotation from the DSL `[dim]` syntax.
    /// Two-element array: `[P_exponent, T_exponent]`.
    /// E.g., `[0, -1]` = per-capita rate (T⁻¹), `[1, -1]` = population rate (P·T⁻¹).
    /// `None` = no annotation (dimension inferred by compiler).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_dim:     Option<(i32, i32)>,
}

impl Parameter {
    /// The inference search box (former `bounds` field), if `Estimated`.
    pub fn bounds(&self) -> Option<(f64, f64)> {
        self.value.bounds()
    }
    /// The single-level prior (former `prior` field): `Some` only for an
    /// `Estimated` parameter whose `PriorSpec` is `Dist`.
    pub fn prior_dist(&self) -> Option<&PriorDist> {
        match &self.value {
            ParamValue::Estimated { prior: PriorSpec::Dist(d), .. } => Some(d),
            _ => None,
        }
    }
    /// The hierarchical prior (former `hierarchical` field): `Some` only for
    /// an `Estimated` parameter whose `PriorSpec` is `Hierarchical`.
    pub fn hierarchical(&self) -> Option<&HierarchicalPrior> {
        match &self.value {
            ParamValue::Estimated { prior: PriorSpec::Hierarchical(h), .. } => Some(h),
            _ => None,
        }
    }
    /// The optimiser's starting point (former `initial_value` field).
    pub fn initial_value(&self) -> Option<f64> {
        self.value.init()
    }
    /// The transform (former `transform` field), if `Estimated`.
    pub fn transform(&self) -> Option<Transform> {
        self.value.transform()
    }
}

