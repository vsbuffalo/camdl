use serde::{Deserialize, Serialize};
use crate::expr::Expr;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sinusoidal {
    pub amplitude: Expr,
    pub period:    Expr,
    pub phase:     Expr,
    pub baseline:  Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Piecewise {
    pub breakpoints: Vec<Expr>,
    pub values:      Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interpolated {
    pub times:  Vec<Expr>,
    pub values: Vec<Expr>,
    pub method: InterpMethod,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpMethod {
    Linear,
    Constant,
    Spline,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Periodic {
    pub period: Expr,
    pub values: Vec<Expr>,
}

/// gh#59: finite Fourier series with N estimable cos/sin pairs.
/// `harmonics[k]` is the (a_k, b_k) pair for harmonic k = 1, 2, …
/// (k=0 baseline is the caller's responsibility: `1 + sum_k a_k cos + b_k sin`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fourier {
    pub period:    Expr,
    pub harmonics: Vec<(Expr, Expr)>,
}

/// gh#59 v2 (2026-05-12): periodic B-spline with uniform knots.
///
/// Knots are implicit: `dx = period / n_basis`, knots at `k * dx`
/// for `k = -degree..n_basis+degree`. `coefs` has length `n_basis`.
/// Standard de Boor recurrence + periodic wrap-fold + (degree-1)/2
/// centering shift; algorithm from de Boor 1978 §X, Eilers & Marx
/// 1996, Wand & Ormerod 2008. See proposal at
/// `docs/dev/proposals/2026-05-12-periodic-bspline-algorithm.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeriodicSpline {
    pub period:  Expr,
    pub n_basis: u32,
    pub degree:  u32,
    pub coefs:   Vec<Expr>,
    /// How the basis sum becomes the forcing value. Omitted when identity, so
    /// a model without `link = log` serializes as before.
    #[serde(default, skip_serializing_if = "SplineLink::is_identity")]
    pub link:    SplineLink,
}

/// The map from a spline's basis sum g(t) = Σ cₖ Bₖ(t) to the forcing value:
/// `Identity` (f = g) or `Log` (f = exp g, positive by construction). Proposal
/// 2026-10-27-estimable-bsplines §4.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplineLink {
    #[default]
    Identity,
    Log,
}

impl SplineLink {
    pub fn is_identity(&self) -> bool {
        matches!(self, SplineLink::Identity)
    }
}

/// The knots of a clamped `bspline` forcing (proposal §4.3). `Uniform` keeps
/// its window ends as single expressions, because an end may be an
/// observation anchor the runtime substitutes; `n_intervals` equal intervals
/// span it. `Explicit` lists strictly increasing breakpoints, the first and
/// last being the window. Both are structural: a parameter there is refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BSplineKnots {
    Uniform { lower: Expr, upper: Expr, n_intervals: u32 },
    Explicit { breakpoints: Vec<Expr> },
}

/// A spline's coefficients: one live expression per basis function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplineCoefs {
    Explicit { coefs: Vec<Expr> },
}

/// A clamped B-spline of `degree` over a window, held at its end values
/// outside it (proposal §4.3). The basis has `n_intervals + degree`
/// functions, so that is the coefficient count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BSpline {
    pub knots:  BSplineKnots,
    pub degree: u32,
    pub coefs:  SplineCoefs,
    #[serde(default, skip_serializing_if = "SplineLink::is_identity")]
    pub link:   SplineLink,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeFuncKind {
    Sinusoidal(Sinusoidal),
    Piecewise(Piecewise),
    Interpolated(Interpolated),
    Periodic(Periodic),
    Fourier(Fourier),                  // gh#59
    PeriodicSpline(PeriodicSpline),    // gh#59
    // `rename_all` would spell this `b_spline`.
    #[serde(rename = "bspline")]
    BSpline(BSpline),                  // proposal 2026-10-27-estimable-bsplines
}

impl TimeFuncKind {
    /// The number of basis functions of a spline kind — the index range of its
    /// `forcing_basis` nodes — or `None` for a kind that has no basis.
    pub fn spline_basis_count(&self) -> Option<usize> {
        match self {
            TimeFuncKind::PeriodicSpline(ps) => Some(ps.n_basis as usize),
            TimeFuncKind::BSpline(b) => {
                let n_intervals = match &b.knots {
                    BSplineKnots::Uniform { n_intervals, .. } => *n_intervals as usize,
                    BSplineKnots::Explicit { breakpoints } => breakpoints.len().saturating_sub(1),
                };
                Some(n_intervals + b.degree as usize)
            }
            TimeFuncKind::Sinusoidal(_)
            | TimeFuncKind::Piecewise(_)
            | TimeFuncKind::Interpolated(_)
            | TimeFuncKind::Periodic(_)
            | TimeFuncKind::Fourier(_) => None,
        }
    }
}

/// Compile-time provenance for a forcing declared `data = "path"`.
///
/// The knots are read from the file and *inlined* into [`Interpolated`] by
/// `camdlc`, so the runtime never opens it — which is exactly why the file has
/// to be recorded. Without this, a completed fit's provenance is silently
/// incomplete on the one input most likely to change underneath it: the file
/// is not named in the IR, not named in the run record, and not recoverable
/// from the baked values.
///
/// # Not run identity
///
/// Neither field is folded into `Model::hash_into` (`runid::ir_hash`), and two
/// tests there pin that:
///
/// - `sha256` is redundant with the inlined knots, which *are* hashed. A file
///   edit that changes a value already re-keys through the values. Folding the
///   byte hash in as well would *additionally* re-key on edits that change no
///   compiled value — a comment line, a trailing newline, CRLF, a column
///   reorder, rows for a stratum this model does not read — invalidating the
///   cache for a model that is bit-for-bit the same model.
/// - `path` must not re-key: the same bytes read from a second location (a
///   copy, a checkout at a different relative prefix) are the same model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSource {
    /// The path **as written in the model**, not the resolved absolute one, so
    /// it stays portable and comparable across machines and checkouts.
    pub path: String,
    /// Lowercase 64-char SHA-256 of the file's raw bytes — reproducible with
    /// `shasum -a 256 <path>` from the model's directory.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeFunction {
    pub name: String,
    pub kind: TimeFuncKind,
    /// Declared dimension from the forcing's tier-3 unit literal
    /// (GH #8): `(P_exp, T_exp)`. E.g. `(0, -1)` for `'per_day`,
    /// `(1, 0)` for `'count`, `(0, 0)` for `'ratio`. Always present —
    /// the parser requires a unit literal on every forcing
    /// declaration, so the dim-checker can use this authoritatively
    /// without falling back on value-based inference.
    pub dim: (i32, i32),
    /// gh#314: optional evaluation-time shift. When `Some(lag)`, every
    /// forcing kind is evaluated at `t − lag` instead of `t` via one
    /// shared shift. `lag` is a duration expression already rescaled to
    /// the model's `time_unit` (a literal, or a `Param` reference — the
    /// lag-as-parameter case is a primary motivation). `None` ⇒ no shift,
    /// byte-identical to a forcing declared without `lag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag: Option<Expr>,
    /// The external file this forcing's knots were read from, when it declared
    /// `data = "path"`. `None` for every other forcing kind (and for the
    /// `table =` / inline `times`/`values` forms of `interpolated`), which is
    /// why it appends only when present (the `integrator` idiom): a model with
    /// no file-backed forcing keeps its exact pre-0.33 bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_source: Option<DataSource>,
}
