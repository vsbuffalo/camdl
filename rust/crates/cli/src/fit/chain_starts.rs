//! Phase-3 chain-start dispatcher for the CLI UX rev 2 init family.
//!
//! Sister surface to [`crate::fit::init::build_chain_starts`]. The
//! legacy entry point covers the per-stage init methods that ship in
//! fit.toml today (`single` / `uniform` / `lhs` / `survey_top_k`); this
//! module covers the four new warm-start variants introduced by the
//! 2026-05-25 CLI UX rev 2 proposal:
//!
//!   - [`InitMethod::FromPrior`] — per-chain draw from each parameter's
//!     `~ <dist>` declaration; fall back to bounds-uniform with a
//!     startup warning for parameters with no prior (Decision A).
//!   - [`InitMethod::FromPosterior`] — per-chain draw from a posterior
//!     draws TSV (uniformly with replacement; gh#83's default). An
//!     explicit source that can't bind every estimated parameter — a
//!     missing column or an unparseable cell — is a hard error, never a
//!     silent bounds-uniform fallback (gh#274).
//!   - [`InitMethod::FromMle`] — all chains at the MLE point from a
//!     prior fit; knows the fit-output TOML schema.
//!   - [`InitMethod::FromParams`] — all chains at a hand-written flat
//!     params TOML.
//!
//! The seam between Phase 2 (parameter resolution) and Phase 3 (chain
//! initialization) is the [`ChainStart::values`] map: it only contains
//! parameters in `resolved.estimate_set`. Every loader builds the map
//! by iterating that set — there is no public way to ask "what's the
//! starting value for `gamma`?" when `gamma` is in `Fixed`. This is
//! what guarantees `--fixed` always wins over `--init`.
//!
//! Provenance: every [`ChainStart`] carries an [`InitSource`] tag that
//! step 9 serializes into `run.json`'s `init_provenance.chains[i]`.
//!
//! See [`docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md`]
//! §"Init phase types" for the design rationale (verb-per-source
//! contract, fall-back behaviour, etc.).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sim::inference::prior::{Prior, Density};
use sim::rng::StatefulRng;

use crate::params_resolver::ResolvedParameters;
use crate::util::derive_chain_seed;

use super::init::{InitMethod, MleSource, PosteriorSource};

/// Per-chain provenance tag. Stored on each [`ChainStart`] and rendered
/// into `run.json`'s `init_provenance.chains[i][param].source` by
/// step 9.
///
/// Each variant carries enough information to identify the specific
/// draw: the seed for stochastic samplers, the row index + path for
/// file-based draws, the rank for survey ranking.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InitSource {
    /// All chains at the seeded base param values (`InitMethod::Single`,
    /// or the no-op fallback for `n_chains < 2`).
    SeededBase,
    /// Per-chain uniform random draw within parameter bounds (legacy
    /// `Uniform` mode).
    UniformDraw { seed: u64 },
    /// Per-chain Stan-style draw: i.i.d. `Uniform(-2, 2)` on the
    /// unconstrained scale, squashed and mapped into bounds
    /// (`InitMethod::UniformUnconstrained`).
    UnconstrainedDraw { seed: u64 },
    /// One stratum of an LHS layout.
    LhsCell { row: usize },
    /// Per-chain draw from a parameter's `~` prior declaration (or
    /// bounds-uniform fall-back when no prior).
    PriorDraw { seed: u64 },
    /// One row of a posterior draws TSV.
    PosteriorRow { row: usize, path: PathBuf },
    /// All chains at an MLE point loaded from a fit-output TOML.
    MlePoint { path: PathBuf },
    /// All chains at a hand-written flat params TOML.
    ParamsPoint { path: PathBuf },
    /// Top-K rank from a survey landscape (1-indexed).
    SurveyRank { rank: usize, path: PathBuf },
}

impl InitSource {
    /// Stable string tag for one-line / column-oriented provenance
    /// (e.g. `chain_starts.tsv`'s `source` column header).
    pub fn tag(&self) -> &'static str {
        match self {
            InitSource::SeededBase      => "seeded_base",
            InitSource::UniformDraw{..} => "uniform_draw",
            InitSource::UnconstrainedDraw{..} => "unconstrained_draw",
            InitSource::LhsCell{..}     => "lhs_cell",
            InitSource::PriorDraw{..}   => "prior_draw",
            InitSource::PosteriorRow{..}=> "posterior_row",
            InitSource::MlePoint{..}    => "mle_point",
            InitSource::ParamsPoint{..} => "params_point",
            InitSource::SurveyRank{..}  => "survey_rank",
        }
    }
}

/// One chain's starting point. Domain is restricted to
/// `resolved.estimate_set` — fixed parameters are not in this map.
#[derive(Debug, Clone)]
pub struct ChainStart {
    pub chain_id: usize,
    /// Parameter name → starting value. Keys = `resolved.estimate_set`
    /// exactly (loaders guarantee this).
    pub values:   HashMap<String, f64>,
    pub source:   InitSource,
}

/// The full set of chain starts produced by [`draw_chain_starts`].
#[derive(Debug, Clone)]
pub struct ChainStarts {
    /// Length = `n_chains` requested.
    pub starts: Vec<ChainStart>,
    /// The method that produced these starts. Echoed into
    /// `run.json`'s `init_provenance.method`.
    pub method: InitMethod,
}

impl ChainStarts {
    /// Adapt to the IF2-shaped `Vec<Vec<EstimatedParam>>` view that
    /// `runner::run_chains_with_per_chain_params` /
    /// `profile`'s per-cell init / `nlopt_stage::build_chain_param_vecs`
    /// already consume. Each chain's `EstimatedParam`s start from
    /// `base_specs` and have `.initial` overwritten from
    /// `ChainStart.values` for any name present in the HashMap.
    ///
    /// Names in `estimate_set` that the HashMap doesn't carry (e.g.
    /// when a loader fell back to bounds-uniform for a missing
    /// column) take `base_specs[i].initial`; the loader already
    /// emitted a startup warning so the silent fall-through is
    /// auditable.
    pub fn to_estimated_params(
        &self,
        base_specs: &[sim::inference::types::EstimatedParam],
    ) -> Vec<Vec<sim::inference::types::EstimatedParam>> {
        self.starts.iter().map(|cs| {
            base_specs.iter().map(|spec| {
                let initial = cs.values.get(&spec.name)
                    .copied()
                    .unwrap_or(spec.initial);
                sim::inference::types::EstimatedParam {
                    initial, ..spec.clone()
                }
            }).collect()
        }).collect()
    }
}

/// Errors specific to chain-start drawing. Returned by
/// [`draw_chain_starts`] and the per-variant loaders.
///
/// Missing parameters in a `from-mle` / `from-params` / `from-prior`
/// source are handled by those loaders via bounds-uniform fall-back +
/// a stderr warning (per proposal §"Init family"), not by a distinct
/// error variant. `from-posterior` is the exception: an explicit draws
/// source that can't bind an estimated parameter (missing column or
/// unparseable cell) is a hard [`InitError::SchemaMismatch`], not a
/// silent fall-back (gh#274).
#[derive(Debug)]
pub enum InitError {
    /// A path argument doesn't point at a readable file or directory.
    UnknownSource { path: PathBuf },
    /// A loader rejected its source because the file shape didn't
    /// match the variant's expected schema. The `expected` slice
    /// names the schema (e.g. `"flat params toml"`); `msg` is the
    /// loader's diagnostic.
    SchemaMismatch {
        path:     PathBuf,
        expected: &'static str,
        msg:      String,
    },
    /// `FromPrior` was requested but at least one parameter had no
    /// `~` declared and no bounds for the uniform fall-back. Lists
    /// the offending parameter names. The fall-back-to-bounds-uniform
    /// path of Decision A is the normal case; this variant fires
    /// only when neither prior nor bounds exist.
    NoPriorAndNoBounds { params: Vec<String> },
    /// I/O error reading a source file.
    Io { path: PathBuf, msg: String },
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::UnknownSource { path } => write!(
                f, "init source `{}` does not exist", path.display()),
            InitError::SchemaMismatch { path, expected, msg } => write!(
                f, "init source `{}` does not look like a {}: {}",
                path.display(), expected, msg),
            InitError::NoPriorAndNoBounds { params } => write!(
                f,
                "--init from-prior requires either a `~ <dist>` \
                 declaration or finite bounds on every estimated \
                 parameter; the following have neither: {}",
                params.join(", ")),
            InitError::Io { path, msg } => write!(
                f, "cannot read `{}`: {}", path.display(), msg),
        }
    }
}

impl std::error::Error for InitError {}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Build [`ChainStarts`] for the four step-6 warm-start init variants.
///
/// Loaders dispatch on `method`'s variant, then assemble per-chain
/// starts by iterating `resolved.estimate_set` and reading the
/// corresponding column / row / draw. Parameters absent from the
/// source file but present in `estimate_set` fall back to a uniform
/// draw within `model.parameters[*].bounds` with a stderr warning.
///
/// For the legacy `Single` / `Uniform` / `Lhs` / `SurveyTopK` variants
/// this function delegates to the existing
/// [`crate::fit::init::build_chain_starts`] path indirectly: it builds
/// `ChainStart` entries with [`InitSource::SeededBase`] /
/// [`InitSource::UniformDraw`] / [`InitSource::LhsCell`] /
/// [`InitSource::SurveyRank`] tags and the resolved-base values.
/// Step-7 wires this into the inference subcommands; for step 6 the
/// function is exercised primarily through unit tests.
pub fn draw_chain_starts(
    resolved: &ResolvedParameters,
    method:   &InitMethod,
    n_chains: usize,
    seed:     u64,
) -> Result<ChainStarts, InitError> {
    if n_chains == 0 {
        return Ok(ChainStarts { starts: Vec::new(), method: method.clone() });
    }
    let starts = match method {
        InitMethod::Single =>
            draw_single(resolved, n_chains),
        InitMethod::Uniform =>
            draw_uniform(resolved, n_chains, seed),
        InitMethod::Lhs =>
            draw_lhs(resolved, n_chains, seed),
        InitMethod::UniformUnconstrained =>
            draw_uniform_unconstrained(resolved, n_chains, seed),
        InitMethod::SurveyTopK => {
            // SurveyTopK requires a SurveyFitContext that's only built
            // at the stage callsite; the canonical entry point for
            // this variant remains `init::resolve_per_chain_starts_from_method`.
            // Reaching this branch via `draw_chain_starts` is a
            // mis-dispatch — surface it as a schema-mismatch error so
            // step-7 wiring catches the bug at test time.
            return Err(InitError::SchemaMismatch {
                path: PathBuf::from("<survey>"),
                expected: "draw_chain_starts(SurveyTopK) — use \
                    init::resolve_per_chain_starts_from_method instead",
                msg: "SurveyTopK needs a SurveyFitContext \
                    (model_identity, data_hashes, [fixed], estimate_names) \
                    that draw_chain_starts has no access to".into(),
            });
        }
        InitMethod::FromPrior =>
            draw_from_prior(resolved, n_chains, seed)?,
        InitMethod::FromPosterior { source } =>
            draw_from_posterior(resolved, source, n_chains, seed)?,
        InitMethod::FromMle { source } =>
            draw_from_mle(resolved, source, n_chains)?,
        InitMethod::FromParams { path } =>
            draw_from_params(resolved, path, n_chains)?,
    };
    Ok(ChainStarts { starts, method: method.clone() })
}

// ─── Legacy-mode adapters (Single / Uniform / Lhs) ──────────────────────────
//
// These exist to give the new entry point uniform output shape — every
// caller gets `Vec<ChainStart>` regardless of method, so step-9
// run.json provenance has one consistent serializer.

fn draw_single(
    resolved: &ResolvedParameters,
    n_chains: usize,
) -> Vec<ChainStart> {
    // Every chain starts at the resolved base values. The values map
    // is restricted to `estimate_set` per the Phase 2 → Phase 3
    // invariant.
    let base = estimate_values_from_resolved(resolved);
    (0..n_chains).map(|chain_id| ChainStart {
        chain_id,
        values: base.clone(),
        source: InitSource::SeededBase,
    }).collect()
}

fn draw_uniform(
    resolved: &ResolvedParameters,
    n_chains: usize,
    seed: u64,
) -> Vec<ChainStart> {
    // Chain 0 keeps the seeded base (reproducibility); chains 1..N
    // draw fresh uniform within bounds. Mirrors the legacy
    // `build_uniform_chain_starts` behaviour but on the
    // estimate_set-restricted value map.
    let bounds_map = bounds_map_for_estimate(resolved);
    let base = estimate_values_from_resolved(resolved);
    (0..n_chains).map(|chain_id| {
        let chain_seed = derive_chain_seed(seed, chain_id);
        let mut values = HashMap::with_capacity(resolved.estimate_set.len());
        if chain_id == 0 {
            values = base.clone();
        } else {
            let mut rng = StatefulRng::new(chain_seed);
            for name in &resolved.estimate_set {
                let val = match bounds_map.get(name) {
                    Some((lo, hi)) if lo.is_finite() && hi.is_finite() =>
                        lo + rng.uniform() * (hi - lo),
                    _ => base.get(name).copied().unwrap_or(0.0)
                        * (0.5 + rng.uniform()),
                };
                values.insert(name.clone(), val);
            }
        }
        ChainStart {
            chain_id,
            values,
            source: InitSource::UniformDraw { seed: chain_seed },
        }
    }).collect()
}

fn draw_lhs(
    resolved: &ResolvedParameters,
    n_chains: usize,
    seed: u64,
) -> Vec<ChainStart> {
    // LHS in [0, 1] across each estimate-set dim, then mapped linearly
    // to `[lo, hi]`. This is a simplified rendering — the full
    // Transform-aware mapping lives in `init::build_lhs_chain_starts`
    // (which operates on `EstimatedParam`, including `Transform`); for
    // `draw_chain_starts` the linear mapping is correct on `Identity`
    // and `Logit` transforms and is the floor for unwarranted-log
    // params. Step 7's wiring into IF2 / PGAS / PMMH will route
    // legacy LHS through the original surface; this version is here
    // primarily so `draw_chain_starts(Lhs)` returns sane values for
    // tests and for step-9 provenance round-trip.
    let n_params = resolved.estimate_set.len();
    if n_params == 0 || n_chains < 2 {
        // No estimate-set or only one chain: degenerate to Single.
        return draw_single(resolved, n_chains);
    }
    let bounds_map = bounds_map_for_estimate(resolved);
    let names: Vec<String> = resolved.estimate_set.iter().cloned().collect();
    let mut rng = StatefulRng::new(seed ^ 0x1f5_beef_u64);
    // u[chain_id][param_id] is the [0,1] LHS coord.
    let mut u: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n_chains];
    for d in 0..n_params {
        let mut perm: Vec<usize> = (0..n_chains).collect();
        for i in (1..n_chains).rev() {
            let j = (rng.uniform() * (i as f64 + 1.0)).floor() as usize;
            perm.swap(i, j.min(i));
        }
        for k in 0..n_chains {
            let jitter = rng.uniform();
            u[k][d] = (perm[k] as f64 + jitter) / n_chains as f64;
        }
    }
    let base = estimate_values_from_resolved(resolved);
    (0..n_chains).map(|chain_id| {
        let mut values = HashMap::with_capacity(n_params);
        for (d, name) in names.iter().enumerate() {
            let val = match bounds_map.get(name) {
                Some((lo, hi)) if lo.is_finite() && hi.is_finite() =>
                    lo + u[chain_id][d] * (hi - lo),
                _ => base.get(name).copied().unwrap_or(0.0)
                    * (0.5 + u[chain_id][d]),
            };
            values.insert(name.clone(), val);
        }
        ChainStart {
            chain_id,
            values,
            source: InitSource::LhsCell { row: chain_id },
        }
    }).collect()
}

fn draw_uniform_unconstrained(
    resolved: &ResolvedParameters,
    n_chains: usize,
    seed: u64,
) -> Vec<ChainStart> {
    // Stan-style: i.i.d. `z ~ U(-R, R)` per (chain, param) on the
    // unconstrained scale, squashed to `u = σ(z)` (a fixed interior band)
    // then mapped into `[lo, hi]`. Boundary-avoiding and scale-invariant.
    // Like `draw_lhs`, this surface uses the linear `[lo, hi]` mapping (it
    // has bounds, not the full `Transform`); the Transform-aware
    // production path for legacy modes lives in
    // `init::build_uniform_unconstrained_chain_starts`.
    let n_params = resolved.estimate_set.len();
    if n_params == 0 || n_chains < 2 {
        return draw_single(resolved, n_chains);
    }
    let radius = crate::fit::init::STAN_INIT_RADIUS;
    let bounds_map = bounds_map_for_estimate(resolved);
    let base = estimate_values_from_resolved(resolved);
    (0..n_chains).map(|chain_id| {
        let chain_seed = derive_chain_seed(seed, chain_id);
        let mut rng = StatefulRng::new(chain_seed);
        let mut values = HashMap::with_capacity(n_params);
        for name in &resolved.estimate_set {
            let z = (rng.uniform() * 2.0 - 1.0) * radius;
            let u = 1.0 / (1.0 + (-z).exp());
            let val = match bounds_map.get(name) {
                Some((lo, hi)) if lo.is_finite() && hi.is_finite() =>
                    lo + u * (hi - lo),
                _ => base.get(name).copied().unwrap_or(0.0) * (0.5 + u),
            };
            values.insert(name.clone(), val);
        }
        ChainStart {
            chain_id,
            values,
            source: InitSource::UnconstrainedDraw { seed: chain_seed },
        }
    }).collect()
}

// ─── Step 6 loaders ─────────────────────────────────────────────────────────

/// `--init from-prior`: per-chain draw from each parameter's `~`
/// declaration. Parameters with no prior fall back to a bounds-uniform
/// draw with a startup warning (Decision A).
fn draw_from_prior(
    resolved: &ResolvedParameters,
    n_chains: usize,
    seed: u64,
) -> Result<Vec<ChainStart>, InitError> {
    // Walk model.parameters once to classify each name as either
    // "has prior" or "fall-back bounds-uniform"; refuse if any name
    // has neither.
    let mut no_prior_names: Vec<String> = Vec::new();
    let mut no_prior_no_bounds: Vec<String> = Vec::new();
    let bounds_map = bounds_map_for_estimate(resolved);
    let priors_by_name: HashMap<String, Prior> = resolved.model.parameters.iter()
        .filter_map(|p| {
            if !resolved.estimate_set.contains(&p.name) { return None; }
            match p.prior_dist() {
                Some(pd) => Some((p.name.clone(), Prior::from_ir(pd))),
                None => {
                    no_prior_names.push(p.name.clone());
                    if bounds_map.get(&p.name)
                        .map(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
                        .unwrap_or(true)
                    {
                        no_prior_no_bounds.push(p.name.clone());
                    }
                    None
                }
            }
        })
        .collect();
    if !no_prior_no_bounds.is_empty() {
        return Err(InitError::NoPriorAndNoBounds {
            params: no_prior_no_bounds,
        });
    }
    if !no_prior_names.is_empty() {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m --init from-prior: no `~` \
             declared for {}; falling back to bounds-uniform for \
             those parameter(s). Add a `~ <dist>` clause in the \
             model or pass `--fixed {}=<value>` to silence this \
             warning.",
            no_prior_names.join(", "),
            no_prior_names.first().map(String::as_str).unwrap_or("<name>"),
        );
    }
    let base = estimate_values_from_resolved(resolved);
    let starts: Vec<ChainStart> = (0..n_chains).map(|chain_id| {
        let chain_seed = derive_chain_seed(seed, chain_id);
        let mut rng = StatefulRng::new(chain_seed);
        let mut values = HashMap::with_capacity(resolved.estimate_set.len());
        for name in &resolved.estimate_set {
            let val = if let Some(prior) = priors_by_name.get(name) {
                sample_prior_natural(prior, &mut rng, base.get(name).copied())
            } else if let Some(&(lo, hi)) = bounds_map.get(name) {
                lo + rng.uniform() * (hi - lo)
            } else {
                // Unreachable: NoPriorAndNoBounds would have caught
                // this above. Keep a defensive base-value fall-back.
                base.get(name).copied().unwrap_or(0.0)
            };
            values.insert(name.clone(), val);
        }
        ChainStart {
            chain_id,
            values,
            source: InitSource::PriorDraw { seed: chain_seed },
        }
    }).collect();
    Ok(starts)
}

/// Natural-scale prior sample. Implements the basics for each
/// [`Prior`] variant via the rejection-free transforms exposed by
/// [`StatefulRng`] (uniform, normal, gamma_multiplier).
///
/// `base` carries the resolver-set base value, used as a fall-back
/// when a draw would lie outside the support (e.g. log_normal on
/// non-positive numbers — should not happen by construction, but
/// kept defensive).
fn sample_prior_natural(prior: &Prior, rng: &mut StatefulRng, base: Option<f64>) -> f64 {
    use std::f64::consts::PI;
    match prior {
        Prior::Fixed(Density::Flat) => base.unwrap_or(0.0),
        Prior::Fixed(Density::Uniform { lower, upper }) => {
            lower + rng.uniform() * (upper - lower)
        }
        Prior::Fixed(Density::Normal { mean, sd }) => {
            mean + sd * rng.normal()
        }
        Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
            // Log-normal: Normal(mu, sigma) on the log scale.
            let z = mean + sd * rng.normal();
            z.exp()
        }
        Prior::Fixed(Density::HalfNormal { sigma }) => {
            // |Normal(0, sigma)|.
            (sigma * rng.normal()).abs()
        }
        Prior::Fixed(Density::Beta { alpha, beta }) => {
            // Beta(α, β) = X / (X + Y), X ~ Gamma(α, 1), Y ~ Gamma(β, 1).
            // StatefulRng.gamma_multiplier returns a Gamma(α, 1) factor
            // when called with (σ² = 1/α, dt = 1) — it's tuned for the
            // chain-binomial multiplier use case; here we need a plain
            // Gamma draw, so fall through to a simple Marsaglia &
            // Tsang trial via two Normal + one Uniform per accept.
            let x = sample_gamma_shape_rate(rng, *alpha, 1.0);
            let y = sample_gamma_shape_rate(rng, *beta,  1.0);
            x / (x + y)
        }
        Prior::Fixed(Density::Gamma { shape, rate }) => {
            sample_gamma_shape_rate(rng, *shape, *rate)
        }
        Prior::Fixed(Density::Exponential { rate }) => {
            // Inverse-CDF: -ln(U) / rate.
            -(1.0 - rng.uniform()).ln() / rate
        }
        Prior::Fixed(Density::LogUniform { lower, upper }) => {
            // Uniform on the log scale, exponentiated.
            let (ll, lu) = (lower.ln(), upper.ln());
            (ll + rng.uniform() * (lu - ll)).exp()
        }
        Prior::Fixed(Density::TruncatedNormal { mean, sd, lower, upper }) => {
            // Exact inverse-CDF draw inside [lower, upper] — no rejection.
            use sim::inference::{normal_cdf, normal_quantile};
            let a = normal_cdf((lower - mean) / sd);
            let b = normal_cdf((upper - mean) / sd);
            let q = a + rng.uniform() * (b - a);
            (mean + sd * normal_quantile(q)).clamp(*lower, *upper)
        }
        Prior::Hierarchical(_) => {
            // Hierarchical priors are evaluated against a ParamEnv that
            // we don't have here. Fall back to the base value with a
            // warning so the user notices.
            eprintln!("\x1b[33mwarning:\x1b[0m --init from-prior: \
                hierarchical prior cannot be sampled at chain-init \
                time (needs ParamEnv); using resolved base value.");
            base.unwrap_or({
                // sentinel — picks something visible if base is None.
                let _ = PI;
                1.0
            })
        }
    }
}

/// Marsaglia & Tsang's Gamma(shape, rate) sampler — accepts on a single
/// Normal² rejection step in the shape ≥ 1 branch; uses a recursion
/// `Gamma(s, r) = U^(1/s) · Gamma(s+1, r)` for shape < 1.
fn sample_gamma_shape_rate(rng: &mut StatefulRng, shape: f64, rate: f64) -> f64 {
    if shape < 1.0 {
        // Recursive boost: Gamma(α, r) = U^(1/α) · Gamma(α+1, r).
        let u: f64 = {
            let mut x = rng.uniform();
            while x <= 0.0 { x = rng.uniform(); }
            x
        };
        return u.powf(1.0 / shape) * sample_gamma_shape_rate(rng, shape + 1.0, rate);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = rng.normal();
        let v_inner = 1.0 + c * z;
        if v_inner <= 0.0 { continue; }
        let v = v_inner * v_inner * v_inner;
        let u: f64 = {
            let mut x = rng.uniform();
            while x <= 0.0 { x = rng.uniform(); }
            x
        };
        // Squeeze test (cheap reject path).
        if u < 1.0 - 0.0331 * z * z * z * z {
            return d * v / rate;
        }
        // Full check.
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

/// `--init from-posterior`: per-chain row draw (uniform with
/// replacement) from a posterior draws TSV.
///
/// The source is explicitly requested, so it must bind every estimated
/// parameter: a missing column, or a cell that won't parse as a number,
/// is a hard [`InitError::SchemaMismatch`] — not a silent bounds-uniform
/// or base-value substitution (gh#274). Cells are parsed up front so a
/// bad value is caught deterministically, independent of which rows the
/// per-chain sampler happens to draw.
fn draw_from_posterior(
    resolved: &ResolvedParameters,
    source: &PosteriorSource,
    n_chains: usize,
    seed: u64,
) -> Result<Vec<ChainStart>, InitError> {
    let path = match source {
        PosteriorSource::DrawsTsv(p) => p.clone(),
        PosteriorSource::FitDir(dir) => {
            let candidate = dir.join("draws.tsv");
            if !candidate.is_file() {
                return Err(InitError::UnknownSource { path: candidate });
            }
            candidate
        }
    };
    let (header, rows) = read_tsv(&path)?;
    if rows.is_empty() {
        return Err(InitError::SchemaMismatch {
            path: path.clone(),
            expected: "posterior draws TSV with at least one row",
            msg: "file has a header but no data rows".into(),
        });
    }
    // Map header → column index for each estimate-set name.
    let col_for: HashMap<String, usize> = header.iter().enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    // Every estimated parameter must have a matching column. An
    // explicitly-requested from-posterior source that cannot bind the
    // parameters we asked for is a HARD ERROR, never a silent
    // bounds-uniform substitution — silently starting a stiff model at
    // extreme uniform draws is the gh#274 failure mode (chains blow up
    // at the first emit, with nothing logged).
    let missing: Vec<String> = resolved.estimate_set.iter()
        .filter(|n| !col_for.contains_key(n.as_str()))
        .cloned()
        .collect();
    if !missing.is_empty() {
        let expected_cols: Vec<&str> =
            resolved.estimate_set.iter().map(String::as_str).collect();
        return Err(InitError::SchemaMismatch {
            path: path.clone(),
            expected: "posterior draws TSV",
            msg: format!(
                "missing column(s): {} — the draws TSV must have one \
                 column per estimated parameter ({}). Present columns: {}.",
                missing.join(", "),
                expected_cols.join(", "),
                header.join(", ")),
        });
    }
    // Parse the estimate-set columns for every row up front, so an
    // unparseable cell in an explicitly-requested source is a
    // deterministic hard error (independent of which rows get sampled)
    // rather than a silent base-value substitution (gh#274). After this
    // the sampling loop only indexes into already-parsed values.
    let names: Vec<&String> = resolved.estimate_set.iter().collect();
    let mut parsed: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
    for (r, row) in rows.iter().enumerate() {
        let mut vals = Vec::with_capacity(names.len());
        for name in &names {
            let col = col_for[name.as_str()]; // present: checked above
            let cell = row.get(col).map(String::as_str).unwrap_or("");
            let v = cell.parse::<f64>().map_err(|_| InitError::SchemaMismatch {
                path: path.clone(),
                expected: "posterior draws TSV",
                msg: format!(
                    "column `{}` row {} has value `{}` that does not \
                     parse as a number",
                    name, r + 1, cell),
            })?;
            vals.push(v);
        }
        parsed.push(vals);
    }
    let mut rng = StatefulRng::new(seed ^ 0xb05_e_05u64);
    let starts: Vec<ChainStart> = (0..n_chains).map(|chain_id| {
        let row_idx = (rng.uniform() * rows.len() as f64).floor() as usize;
        let row_idx = row_idx.min(rows.len() - 1);
        let mut values = HashMap::with_capacity(names.len());
        for (k, name) in names.iter().enumerate() {
            values.insert((*name).clone(), parsed[row_idx][k]);
        }
        ChainStart {
            chain_id,
            values,
            source: InitSource::PosteriorRow { row: row_idx, path: path.clone() },
        }
    }).collect();
    Ok(starts)
}

/// `--init from-mle`: all chains at the MLE point from a fit-output
/// TOML. Knows the fit-output schema — skips `[provenance]` /
/// `[focal]` / scalar metadata and reads values from either an `[mle]`
/// section or top-level scalars.
fn draw_from_mle(
    resolved: &ResolvedParameters,
    source: &MleSource,
    n_chains: usize,
) -> Result<Vec<ChainStart>, InitError> {
    let path = match source {
        MleSource::File(p) => p.clone(),
        MleSource::FitDir(dir) => {
            let mle_path = dir.join("mle.toml");
            if mle_path.is_file() {
                mle_path
            } else {
                let final_path = dir.join("final_params.toml");
                if final_path.is_file() {
                    final_path
                } else {
                    return Err(InitError::UnknownSource {
                        path: dir.join("mle.toml-or-final_params.toml"),
                    });
                }
            }
        }
    };
    let values_in_file = load_mle_toml(&path)?;
    apply_point_to_all_chains(resolved, &path, &values_in_file, n_chains,
        |path| InitSource::MlePoint { path })
}

/// `--init from-params`: all chains at a hand-written flat params TOML.
fn draw_from_params(
    resolved: &ResolvedParameters,
    path: &Path,
    n_chains: usize,
) -> Result<Vec<ChainStart>, InitError> {
    // Reject files that look like fit-output (have `[focal]` or
    // `[mle]` sections, or a `final_loglik` scalar) — the actionable
    // hint redirects the user to `--init from-mle`.
    let path_buf: PathBuf = path.to_path_buf();
    let raw = std::fs::read_to_string(path).map_err(|e| InitError::Io {
        path: path_buf.clone(), msg: e.to_string(),
    })?;
    let table: toml::Table = raw.parse().map_err(|e: toml::de::Error| {
        InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "flat params TOML (top-level keys = parameter names)",
            msg: e.to_string(),
        }
    })?;
    let has_focal = table.contains_key("focal");
    let has_mle_section = table.get("mle")
        .map(|v| matches!(v, toml::Value::Table(_))).unwrap_or(false);
    let has_final_loglik = table.contains_key("final_loglik");
    if has_focal || has_mle_section || has_final_loglik {
        return Err(InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "flat params TOML (top-level keys = parameter names)",
            msg: "this file has `[focal]` / `[mle]` / `final_loglik` \
                  scalars — it looks like an mle.toml. Use \
                  `--init from-mle --mle <path>` for fit-output \
                  TOMLs.".into(),
        });
    }
    let values_in_file = crate::util::load_params_toml(
        &path_buf.to_string_lossy())
        .map_err(|msg| InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "flat params TOML",
            msg,
        })?;
    apply_point_to_all_chains(resolved, &path_buf, &values_in_file, n_chains,
        |path| InitSource::ParamsPoint { path })
}

/// Shared logic for `FromMle` and `FromParams`: a single point loaded
/// from a TOML, replicated across all chains. Missing parameters
/// (i.e. names in `estimate_set` not present in the file) fall back
/// to bounds-uniform with a startup warning naming them.
fn apply_point_to_all_chains<F>(
    resolved: &ResolvedParameters,
    path: &Path,
    values_in_file: &HashMap<String, f64>,
    n_chains: usize,
    make_source: F,
) -> Result<Vec<ChainStart>, InitError>
where F: Fn(PathBuf) -> InitSource,
{
    let bounds_map = bounds_map_for_estimate(resolved);
    let base = estimate_values_from_resolved(resolved);
    let needed: HashSet<&str> = resolved.estimate_set.iter()
        .map(|s| s.as_str()).collect();
    let missing: Vec<String> = needed.iter()
        .filter(|n| !values_in_file.contains_key(**n))
        .map(|s| (*s).to_string())
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m --init source `{}` is missing \
             parameter(s): {}. Falling back to bounds-uniform for \
             those parameter(s).",
            path.display(), missing.join(", "));
    }
    let path_buf: PathBuf = path.to_path_buf();
    // Build the per-chain values once — all chains share the point;
    // the per-chain RNG only fires for the bounds-uniform fall-back
    // on missing names, and we want that draw to vary across chains.
    let starts: Vec<ChainStart> = (0..n_chains).map(|chain_id| {
        let mut values = HashMap::with_capacity(resolved.estimate_set.len());
        let mut rng = StatefulRng::new(
            derive_chain_seed(0xfa11_bac0u64, chain_id));
        for name in &resolved.estimate_set {
            let val = match values_in_file.get(name) {
                Some(&v) => v,
                None => match bounds_map.get(name) {
                    Some(&(lo, hi)) if lo.is_finite() && hi.is_finite() =>
                        lo + rng.uniform() * (hi - lo),
                    _ => base.get(name).copied().unwrap_or(0.0),
                },
            };
            values.insert(name.clone(), val);
        }
        ChainStart {
            chain_id,
            values,
            source: make_source(path_buf.clone()),
        }
    }).collect();
    Ok(starts)
}

/// Load an `mle.toml` / `final_params.toml`-shape file. Skips
/// `[provenance]` / `[focal]` sections, reads parameter values from
/// either top-level scalars or an `[mle]` section. Mirrors what the
/// `from-mle` documentation promises and what current fit-output
/// emits.
pub fn load_mle_toml(path: &Path) -> Result<HashMap<String, f64>, InitError> {
    let path_buf: PathBuf = path.to_path_buf();
    let raw = std::fs::read_to_string(path).map_err(|e| InitError::Io {
        path: path_buf.clone(), msg: e.to_string(),
    })?;
    let table: toml::Table = raw.parse().map_err(|e: toml::de::Error| {
        InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "fit-output TOML (mle.toml / final_params.toml)",
            msg: e.to_string(),
        }
    })?;
    let mut out: HashMap<String, f64> = HashMap::new();
    // First: top-level scalar entries (final_params.toml uses this
    // shape; `[provenance]`, `[focal]` are sections that get skipped).
    for (key, val) in &table {
        // Skip section names (handled below) + metadata scalars like
        // `final_loglik` that share the top-level namespace but
        // aren't parameters.
        if key == "provenance" || key == "focal" || key == "mle"
            || key == "final_loglik" { continue; }
        match val {
            toml::Value::Float(f)   => { out.insert(key.clone(), *f); }
            toml::Value::Integer(i) => { out.insert(key.clone(), *i as f64); }
            // skip non-scalar metadata silently (`final_loglik`, etc.
            // are floats — they'll be captured above; any unexpected
            // section is ignored).
            _ => {}
        }
    }
    // Second: `[mle]` section overrides top-level (`mle.toml` shape).
    if let Some(toml::Value::Table(mle_section)) = table.get("mle") {
        for (key, val) in mle_section {
            match val {
                toml::Value::Float(f)   => { out.insert(key.clone(), *f); }
                toml::Value::Integer(i) => { out.insert(key.clone(), *i as f64); }
                _ => {}
            }
        }
    }
    Ok(out)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Bounds map restricted to `estimate_set`. Missing bounds → omitted.
///
/// Reads the same search box the transform clamps to
/// (`params_resolver::resolved_bounds`), so a `probability` that declared no `in
/// [lo, hi]` gets random / LHS starts drawn from `[0, 1]` rather than being
/// omitted from the map entirely (gh#763).
fn bounds_map_for_estimate(resolved: &ResolvedParameters) -> HashMap<String, (f64, f64)> {
    resolved.model.parameters.iter()
        .filter(|p| resolved.estimate_set.contains(&p.name))
        .filter_map(|p| crate::params_resolver::resolved_bounds(p).map(|b| (p.name.clone(), b)))
        .collect()
}

/// Resolved base-value map restricted to `estimate_set`. Built from
/// `resolved.params` so the per-parameter `value` is always present.
fn estimate_values_from_resolved(resolved: &ResolvedParameters) -> HashMap<String, f64> {
    resolved.params.iter()
        .filter(|p| resolved.estimate_set.contains(&p.name))
        .map(|p| (p.name.clone(), p.value))
        .collect()
}

/// Minimal TSV reader: header line + data rows. Returns (header,
/// rows) where `header` is the column names and each row is a vec of
/// the line's tab-separated cells (unparsed strings — caller parses
/// per-column).
fn read_tsv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), InitError> {
    let path_buf: PathBuf = path.to_path_buf();
    let raw = std::fs::read_to_string(path).map_err(|e| InitError::Io {
        path: path_buf.clone(), msg: e.to_string(),
    })?;
    let mut lines = raw.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header_line = lines.next().ok_or_else(|| InitError::SchemaMismatch {
        path: path_buf.clone(),
        expected: "TSV with header + at least one data row",
        msg: "file is empty or has only comments".into(),
    })?;
    let header: Vec<String> = header_line.split('\t').map(|s| s.to_string()).collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.is_empty() { continue; }
        let cells: Vec<String> = line.split('\t').map(|s| s.to_string()).collect();
        rows.push(cells);
    }
    Ok((header, rows))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params_resolver::{ParameterRole, ResolvedParameter, ValueSource};

    use indexmap::IndexSet;

    fn mk_param(name: &str, value: f64, prior: Option<ir::parameter::PriorDist>,
                bounds: Option<(f64, f64)>) -> ir::parameter::Parameter {
        // A concrete `value` plus optional inference config: carry the value as
        // the estimated `init` when bounds/prior are present (can't be both a
        // Fixed constant and carry bounds), else a plain Fixed.
        let pv = if bounds.is_some() || prior.is_some() {
            ir::parameter::ParamValue::Estimated {
                init: Some(value),
                bounds,
                prior: match prior {
                    Some(pd) => ir::parameter::PriorSpec::Dist(pd),
                    None => ir::parameter::PriorSpec::Flat,
                },
                transform: ir::parameter::Transform::Identity,
            }
        } else {
            ir::parameter::ParamValue::Fixed { value }
        };
        ir::parameter::Parameter {
            name: name.into(),
            value: pv,
            param_kind: None,
            param_dim: None,
        }
    }

    fn mk_resolved(parameters: Vec<ir::parameter::Parameter>,
                   estimate: &[&str]) -> ResolvedParameters {
        let estimate_set: IndexSet<String> = estimate.iter()
            .map(|s| s.to_string()).collect();
        let params: Vec<ResolvedParameter> = parameters.iter().map(|p| {
            ResolvedParameter {
                name: p.name.clone(),
                value: p.value.resolved_value().unwrap(),
                source: ValueSource::ModelDefault,
                role: if estimate_set.contains(&p.name) {
                    ParameterRole::Estimated
                } else {
                    ParameterRole::Fixed {
                        reason: crate::params_resolver::FixReason::NotInEstimate,
                    }
                },
                overrode_scenario: None,
            }
        }).collect();
        let model = ir::Model {
            ic_grad: Default::default(),
            name: "test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters,
            initial_conditions: ir::model::InitialConditions::default(),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0, t_end: 1.0,
                time_semantics: "continuous".into(),
                dt: None, rng_seed: None,
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        ResolvedParameters {
            params, estimate_set, model, warnings: vec![],
        }
    }

    fn write_tmp(name: &str, contents: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "camdl_init_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::write(&p, contents).unwrap();
        p
    }

    // ─── `from-params` ────────────────────────────────────────────────

    #[test]
    fn from_params_loads_flat_toml_and_assigns_to_estimate_set_only() {
        // Resolver: beta + gamma estimated; N0 fixed. The TOML has all
        // three names. Only the two estimated names should appear in
        // ChainStart.values.
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
                mk_param("N0", 1000.0, None, None),
            ],
            &["beta", "gamma"],
        );
        let path = write_tmp("from_params_flat",
            "beta = 0.42\ngamma = 0.12\nN0 = 999\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromParams { path: path.clone() },
            3, 42,
        ).unwrap();
        assert_eq!(starts.starts.len(), 3);
        for cs in &starts.starts {
            // domain restricted to estimate_set.
            let keys: HashSet<&str> = cs.values.keys().map(String::as_str).collect();
            assert_eq!(keys, ["beta", "gamma"].iter().copied().collect::<HashSet<_>>(),
                "ChainStart.values must equal estimate_set, got {:?}", keys);
            // file's beta/gamma applied.
            assert!((cs.values["beta"]  - 0.42).abs() < 1e-12);
            assert!((cs.values["gamma"] - 0.12).abs() < 1e-12);
            // source carries the path.
            match &cs.source {
                InitSource::ParamsPoint { path: p } => assert_eq!(p, &path),
                other => panic!("unexpected InitSource: {:?}", other),
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_params_errors_on_mle_toml_shape_with_actionable_hint() {
        // A file with `[focal]` or `[mle]` section is mle.toml-shaped
        // — `from_params` must refuse and point at `--init from-mle`.
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_params_mle_shape",
            "[focal]\nname = \"beta\"\n\n[mle]\nbeta = 0.42\n");
        let err = draw_chain_starts(
            &resolved,
            &InitMethod::FromParams { path: path.clone() },
            1, 42,
        ).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("from-mle"),
            "error must hint at --init from-mle: {}", msg);
        assert!(msg.contains("mle.toml"),
            "error must explain the file looks like mle.toml: {}", msg);
        std::fs::remove_file(&path).ok();
    }

    // ─── `from-mle` ───────────────────────────────────────────────────

    #[test]
    fn from_mle_resolves_fitdir_to_mle_toml_first_then_final_params() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
            ],
            &["beta", "gamma"],
        );
        let dir = std::env::temp_dir().join(format!(
            "camdl_from_mle_fitdir_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        // mle.toml + final_params.toml both present — mle.toml wins.
        std::fs::write(dir.join("mle.toml"),
            "[mle]\nbeta = 0.55\ngamma = 0.22\n").unwrap();
        std::fs::write(dir.join("final_params.toml"),
            "beta = 9.99\ngamma = 9.99\n").unwrap();
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromMle { source: MleSource::FitDir(dir.clone()) },
            2, 0,
        ).unwrap();
        for cs in &starts.starts {
            assert!((cs.values["beta"]  - 0.55).abs() < 1e-12,
                "expected mle.toml to win, got {}", cs.values["beta"]);
            assert!((cs.values["gamma"] - 0.22).abs() < 1e-12);
            match &cs.source {
                InitSource::MlePoint { path } =>
                    assert_eq!(path, &dir.join("mle.toml")),
                other => panic!("unexpected: {:?}", other),
            }
        }
        // Remove mle.toml: final_params.toml is the fallback.
        std::fs::remove_file(dir.join("mle.toml")).unwrap();
        let starts2 = draw_chain_starts(
            &resolved,
            &InitMethod::FromMle { source: MleSource::FitDir(dir.clone()) },
            1, 0,
        ).unwrap();
        assert!((starts2.starts[0].values["beta"] - 9.99).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_mle_handles_explicit_mle_toml_file() {
        // File path → File variant. Skips [provenance].
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_mle_explicit",
            "[provenance]\nbackend = \"chain_binomial\"\n\n\
             [mle]\nbeta = 0.77\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromMle { source: MleSource::File(path.clone()) },
            2, 0,
        ).unwrap();
        for cs in &starts.starts {
            assert!((cs.values["beta"] - 0.77).abs() < 1e-12);
        }
        std::fs::remove_file(&path).ok();
    }

    // ─── `from-posterior` ──────────────────────────────────────────────

    #[test]
    fn from_posterior_samples_uniformly_with_replacement() {
        // Tiny TSV with 4 rows; draw 50 chains and verify all rows
        // get used (with replacement → expect ≥3 of 4 distinct).
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_post_sampling",
            "beta\n0.10\n0.20\n0.30\n0.40\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(path.clone()),
            },
            50, 42,
        ).unwrap();
        // All values should be from {0.10, 0.20, 0.30, 0.40}.
        let allowed: [f64; 4] = [0.10, 0.20, 0.30, 0.40];
        let used: HashSet<i64> = starts.starts.iter()
            .map(|cs| (cs.values["beta"] * 100.0).round() as i64)
            .collect();
        for u in &used {
            assert!(allowed.iter().any(|a| ((*a * 100.0).round() as i64) == *u),
                "value {} not in allowed {:?}", u, allowed);
        }
        // 50 draws from 4 rows uniformly → very likely all 4 hit at
        // least once. Bound at ≥ 3 to avoid flakiness, but in
        // practice the test seed makes all 4 used.
        assert!(used.len() >= 3,
            "expected ≥ 3 distinct rows used in 50 draws, got {}: {:?}",
            used.len(), used);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_posterior_resolves_fitdir_to_draws_tsv() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let dir = std::env::temp_dir().join(format!(
            "camdl_from_post_fitdir_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("draws.tsv"), "beta\n0.55\n0.66\n").unwrap();
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::FitDir(dir.clone()),
            },
            10, 1,
        ).unwrap();
        assert_eq!(starts.starts.len(), 10);
        for cs in &starts.starts {
            let v = cs.values["beta"];
            assert!((v - 0.55).abs() < 1e-9 || (v - 0.66).abs() < 1e-9,
                "value {} not in {{0.55, 0.66}}", v);
            match &cs.source {
                InitSource::PosteriorRow { path, .. } =>
                    assert_eq!(path, &dir.join("draws.tsv")),
                other => panic!("unexpected: {:?}", other),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_posterior_missing_column_is_hard_error() {
        // gh#274: an explicitly-requested from_posterior source whose
        // draws TSV lacks a column for an estimated parameter must be a
        // HARD ERROR — never a silent bounds-uniform fallback (which on a
        // stiff model starts the chains at extreme uniform draws and
        // crashes at the first emit). `gamma` is estimated but absent
        // from the file.
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
            ],
            &["beta", "gamma"],
        );
        let path = write_tmp("from_post_missing_col",
            "beta\n0.10\n0.20\n0.30\n");
        let err = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(path.clone()),
            },
            4, 42,
        ).unwrap_err();
        let msg = err.to_string();
        // names the file, the missing column, and hints the fix.
        assert!(msg.contains(&path.display().to_string()),
            "error must name the file: {}", msg);
        assert!(msg.contains("gamma"),
            "error must name the missing column: {}", msg);
        assert!(msg.contains("column"),
            "error must hint the one-column-per-parameter fix: {}", msg);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_posterior_unparseable_cell_is_hard_error() {
        // gh#274: a matched column whose value won't parse as f64 is a
        // hard error for an explicit source, not a silent base-value
        // substitution. The check is deterministic (all rows validated
        // up front), so it fires regardless of which rows get sampled.
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_post_bad_cell",
            "beta\n0.10\nnotanumber\n0.30\n");
        let err = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(path.clone()),
            },
            4, 42,
        ).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&path.display().to_string()),
            "error must name the file: {}", msg);
        assert!(msg.contains("beta"),
            "error must name the offending column: {}", msg);
        assert!(msg.contains("notanumber"),
            "error must name the offending value: {}", msg);
        std::fs::remove_file(&path).ok();
    }

    // ─── `from-prior` ──────────────────────────────────────────────────

    #[test]
    fn from_prior_falls_back_to_bounds_uniform_with_warning_for_no_tilde_params() {
        // beta has a prior, gamma does not. Both are bounded — the
        // fallback uniform-on-bounds path must engage for gamma.
        // Decision A: warn, don't error.
        let resolved = mk_resolved(
            vec![
                mk_param("beta", 0.3,
                    Some(ir::parameter::PriorDist::Uniform(
                        ir::parameter::UniformPrior { lower: 0.0, upper: 1.0 })),
                    Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
            ],
            &["beta", "gamma"],
        );
        let starts = draw_chain_starts(
            &resolved, &InitMethod::FromPrior, 4, 42,
        ).unwrap();
        for cs in &starts.starts {
            // Both names present (estimate_set restriction).
            assert!(cs.values.contains_key("beta"));
            assert!(cs.values.contains_key("gamma"));
            // Both within their bounds.
            let b = cs.values["beta"];
            let g = cs.values["gamma"];
            assert!(b >= 0.0 && b <= 1.0);
            assert!(g >= 0.0 && g <= 1.0);
            // Source = PriorDraw with a chain-specific seed.
            match &cs.source {
                InitSource::PriorDraw { .. } => {}
                other => panic!("unexpected source: {:?}", other),
            }
        }
    }

    #[test]
    fn from_prior_uses_declared_dist_when_tilde_present() {
        // beta has Uniform(0.4, 0.5) — narrow box. Many draws must
        // all land inside.
        let resolved = mk_resolved(
            vec![
                mk_param("beta", 0.42,
                    Some(ir::parameter::PriorDist::Uniform(
                        ir::parameter::UniformPrior { lower: 0.4, upper: 0.5 })),
                    Some((0.0, 1.0))),
            ],
            &["beta"],
        );
        let starts = draw_chain_starts(
            &resolved, &InitMethod::FromPrior, 32, 7,
        ).unwrap();
        for cs in &starts.starts {
            let v = cs.values["beta"];
            assert!(v >= 0.4 - 1e-9 && v <= 0.5 + 1e-9,
                "draw {} outside declared prior [0.4, 0.5]", v);
        }
    }

    // ─── Estimate-set domain invariant (per-variant) ──────────────────

    #[test]
    fn from_params_chainstart_values_restricted_to_estimate_set() {
        // Already validated above. Repeat here for the audit-required
        // per-variant coverage. The TOML carries an extra `extra_param`
        // not in estimate_set — must be ignored silently.
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
                mk_param("N0", 1000.0, None, None),
            ],
            &["beta"],
        );
        let path = write_tmp("from_params_extra",
            "beta = 0.42\ngamma = 0.12\nN0 = 999\nextra_param = 1.0\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromParams { path: path.clone() },
            1, 0,
        ).unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_mle_chainstart_values_restricted_to_estimate_set() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("rho",   0.1, None, Some((0.0, 1.0))),
            ],
            &["beta"],
        );
        let path = write_tmp("from_mle_restricted",
            "[mle]\nbeta = 0.66\nrho = 0.05\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromMle { source: MleSource::File(path.clone()) },
            1, 0,
        ).unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_posterior_chainstart_values_restricted_to_estimate_set() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("rho",   0.1, None, Some((0.0, 1.0))),
            ],
            &["beta"],
        );
        let path = write_tmp("from_post_restricted",
            "beta\trho\n0.42\t0.05\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(path.clone()),
            },
            1, 0,
        ).unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_prior_chainstart_values_restricted_to_estimate_set() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3,
                    Some(ir::parameter::PriorDist::Uniform(
                        ir::parameter::UniformPrior { lower: 0.0, upper: 1.0 })),
                    Some((0.0, 1.0))),
                mk_param("rho",   0.1, None, Some((0.0, 1.0))),
            ],
            &["beta"],
        );
        let starts = draw_chain_starts(
            &resolved, &InitMethod::FromPrior, 1, 0,
        ).unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
    }

    // ─── Per-variant provenance tag check ─────────────────────────────

    #[test]
    fn from_params_chainstart_source_records_provenance_with_correct_tag() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_params_tag", "beta = 0.42\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromParams { path: path.clone() },
            1, 0,
        ).unwrap();
        assert_eq!(starts.starts[0].source.tag(), "params_point");
        assert!(matches!(starts.starts[0].source,
            InitSource::ParamsPoint { .. }));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_mle_chainstart_source_records_provenance_with_correct_tag() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_mle_tag", "[mle]\nbeta = 0.42\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromMle { source: MleSource::File(path.clone()) },
            1, 0,
        ).unwrap();
        assert_eq!(starts.starts[0].source.tag(), "mle_point");
        assert!(matches!(starts.starts[0].source,
            InitSource::MlePoint { .. }));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_posterior_chainstart_source_records_provenance_with_correct_tag() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_post_tag", "beta\n0.42\n");
        let starts = draw_chain_starts(
            &resolved,
            &InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(path.clone()),
            },
            1, 0,
        ).unwrap();
        assert_eq!(starts.starts[0].source.tag(), "posterior_row");
        assert!(matches!(starts.starts[0].source,
            InitSource::PosteriorRow { .. }));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_prior_chainstart_source_records_provenance_with_correct_tag() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta", 0.3,
                    Some(ir::parameter::PriorDist::Uniform(
                        ir::parameter::UniformPrior { lower: 0.0, upper: 1.0 })),
                    Some((0.0, 1.0))),
            ],
            &["beta"],
        );
        let starts = draw_chain_starts(
            &resolved, &InitMethod::FromPrior, 1, 42,
        ).unwrap();
        assert_eq!(starts.starts[0].source.tag(), "prior_draw");
        assert!(matches!(starts.starts[0].source,
            InitSource::PriorDraw { .. }));
    }

    // ─── Legacy variant dispatch through draw_chain_starts ─────────────

    #[test]
    fn legacy_single_returns_seeded_base() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let starts = draw_chain_starts(
            &resolved, &InitMethod::Single, 3, 0,
        ).unwrap();
        for cs in &starts.starts {
            assert!((cs.values["beta"] - 0.3).abs() < 1e-12);
            assert!(matches!(cs.source, InitSource::SeededBase));
        }
    }
}
