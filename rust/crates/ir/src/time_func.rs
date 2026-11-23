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

/// An `interpolated` forcing: its knots, from wherever they come, and how
/// the forcing passes between them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interpolated {
    #[serde(flatten)]
    pub source: InterpSource,
    pub method: InterpMethod,
}

impl Interpolated {
    /// A forcing through the knots `(times[i], values[i])`.
    pub fn knots(times: Vec<Expr>, values: Vec<Expr>, method: InterpMethod) -> Self {
        Interpolated { source: InterpSource::Knots { times, values }, method }
    }
}

/// Where an `interpolated` forcing's knots come from — the forcing
/// counterpart of [`crate::table::TableSource`], and serialized the same way
/// (untagged, flattened into the forcing), so a knot-backed forcing's JSON is
/// `{"times": [...], "values": [...], "method": ...}` as it always was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InterpSource {
    /// Knots known when the model compiled: written inline, sliced from a
    /// table, or read by `camdlc` from a `data = "path"` file.
    Knots { times: Vec<Expr>, values: Vec<Expr> },
    /// Knots read when the model is run, from the file bound to
    /// `external("NAME")`. The CLI replaces this with [`InterpSource::Knots`]
    /// before the model is built (`cli::external_inputs::bind`);
    /// `CompiledModel::new` refuses one that reaches it.
    External(ExternalSeries),
}

/// The rows of a bound file that an external forcing reads, and how its
/// values are scaled. The file is read exactly as `camdlc` reads a
/// `data = "path"` forcing: the first line that is neither blank nor a `#`
/// comment is the header, a time cell is a number in model time or an ISO
/// date resolved through the model's origin, and the rows are the knots in
/// file order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalSeries {
    /// The `NAME` of `data = external("NAME")`: the key a binding supplies a
    /// file under (`--table NAME=FILE`, or `[data.tables]` in a fit config).
    pub external:    String,
    pub time_col:    String,
    pub value_col:   String,
    /// The stratum filter of an indexed forcing: only rows whose `column`
    /// holds `level`. `None` for a forcing with no index, which reads every
    /// row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key:         Option<SeriesKey>,
    /// The factor converting a value in the forcing's declared unit to the
    /// model's time unit (1/7 for `'per_week` under `time_unit = 'days`, 1 for
    /// `'ratio`), computed by the compiler. A value `v` becomes the knot
    /// `v × value_scale`, the expression `camdlc` writes for a file it reads.
    pub value_scale: f64,
}

/// See [`ExternalSeries::key`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesKey {
    pub column: String,
    pub level:  String,
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

/// A spline's coefficients.
///
/// - `Explicit`: one live expression per basis function.
/// - `Smooth` (proposal 2026-10-27-estimable-bsplines §6.3–§6.4): the
///   whitened penalized curve w = level·1 + slope·(ξ − t_m·1) +
///   (scale/κ)·M·modes, with M built by the runtime from the resolved knots
///   (`sim::spline_penalty`) and κ = √(K − 2)·per^{3/2}. `modes` has K − 2
///   entries; `per` is the reference duration, structural.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplineCoefs {
    Explicit { coefs: Vec<Expr> },
    Smooth { level: Expr, slope: Expr, scale: Expr, modes: Vec<Expr>, per: Expr },
}

impl SplineCoefs {
    /// Every expression the coefficient form carries, `per` included.
    pub fn exprs(&self) -> Vec<&Expr> {
        match self {
            SplineCoefs::Explicit { coefs } => coefs.iter().collect(),
            SplineCoefs::Smooth { level, slope, scale, modes, per } => {
                [level, slope, scale, per].into_iter().chain(modes).collect()
            }
        }
    }

    pub fn exprs_mut(&mut self) -> Vec<&mut Expr> {
        match self {
            SplineCoefs::Explicit { coefs } => coefs.iter_mut().collect(),
            SplineCoefs::Smooth { level, slope, scale, modes, per } => {
                [level, slope, scale, per].into_iter().chain(modes.iter_mut()).collect()
            }
        }
    }
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

    /// The index range of a `forcing_basis` node in `space` on this forcing,
    /// or `None` if the forcing has no such basis: `Spline` on either spline
    /// kind; `Whitened` (K − 2 modes) and `Linear` (one) only on a `bspline`
    /// whose coefficients are `smooth`.
    pub fn basis_count(&self, space: crate::expr::BasisSpace) -> Option<usize> {
        use crate::expr::BasisSpace;
        let smooth = matches!(self, TimeFuncKind::BSpline(BSpline { coefs: SplineCoefs::Smooth { .. }, .. }));
        match space {
            BasisSpace::Spline => self.spline_basis_count(),
            BasisSpace::Whitened if smooth => self.spline_basis_count().map(|k| k.saturating_sub(2)),
            BasisSpace::Linear if smooth => Some(1),
            BasisSpace::Whitened | BasisSpace::Linear => None,
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
/// Neither field is folded into `Model::hash_into` (`model_hash`), and two
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire form of both knot sources, pinned to the exact strings the
    /// OCaml serializer is pinned to (`ocaml/test/test_ir_roundtrip.ml`,
    /// `external_series_serde_test` and `data_source_serde_test`): each parses
    /// to the source it names and writes back byte for byte.
    #[test]
    fn interp_source_wire_form_matches_the_compiler() {
        let cases = [
            r#"{"interpolated":{"times":[{"const":0.0},{"const":30.0}],"values":[{"const":1.4},{"const":1.3}],"method":"linear"}}"#,
            r#"{"interpolated":{"external":"school","time_col":"date","value_col":"in_term","value_scale":0.5,"method":"constant"}}"#,
            r#"{"interpolated":{"external":"school","time_col":"date","value_col":"in_term","key":{"column":"patch","level":"north"},"value_scale":0.5,"method":"constant"}}"#,
        ];
        for (i, wire) in cases.iter().enumerate() {
            let kind: TimeFuncKind = serde_json::from_str(wire)
                .unwrap_or_else(|e| panic!("case {i} does not parse: {e}"));
            let TimeFuncKind::Interpolated(interp) = &kind else { panic!("case {i}: not interpolated") };
            match (i, &interp.source) {
                (0, InterpSource::Knots { times, .. }) => assert_eq!(times.len(), 2),
                (1, InterpSource::External(e)) => assert_eq!(e.key, None),
                (2, InterpSource::External(e)) => assert_eq!(
                    e.key,
                    Some(SeriesKey { column: "patch".into(), level: "north".into() })
                ),
                (i, s) => panic!("case {i} parsed to the wrong source: {s:?}"),
            }
            assert_eq!(serde_json::to_string(&kind).unwrap(), *wire, "case {i} must write back unchanged");
        }
    }
}
