//! Chain / per-cell init strategies.
//!
//! Three modes today, dispatched via the `init_method` field on
//! `[stages.X]` (and the `--init` CLI override):
//!
//! - **Single** — every chain starts at `config.estimated_params[*].initial`
//!   (i.e. `[estimate].start =` or its fallback). Chains differ only by
//!   IF2's per-chain RNG. Useful for refine stages, single-chain runs,
//!   reproducibility-critical tests, and deterministic NLopt at a known
//!   seed point.
//! - **Uniform** — per-chain uniform random draw within natural-scale
//!   bounds. Legacy mode; equivalent to `Lhs` for `Logit`/`None`
//!   parameters but worse for `Log`-typed parameters at low chain count
//!   (clumps in linear space). Kept for reproducibility of pre-LHS results.
//! - **Lhs** — Latin-hypercube stratified sampling, **scale-aware via
//!   `Transform`**. For Log-typed params (rates, positive quantities) LHS
//!   spans `[ln(lo), ln(hi)]` and exponentiates back, so a single LHS pass
//!   covers orders of magnitude rather than concentrating mass near `hi`.
//!   For Logit-typed params (probabilities) LHS spans `[lo, hi]` linearly.
//!   For untransformed params LHS spans `[lo, hi]`. **This is the default**
//!   across IF2 / PGAS / PMMH / NLopt multi-chain stages.
//!
//! Filed as gh#42. Motivation: downstream typhoid SIRC fit found
//! 30 LHS-drawn chains at chain_binomial backend reach a basin
//! 80,542 nats better than 8 uniform-random-start chains, holding
//! everything else equal. Single-point starts (and clumpy uniform
//! starts at low N) miss basins; LHS gives stratified coverage at
//! the same chain count.

use std::path::PathBuf;

use sim::inference::types::{EstimatedParam, Transform};
use sim::rng::StatefulRng;

use crate::util::derive_chain_seed;

/// How chain (or per-cell) starting points are drawn.
///
/// Default is `Lhs` — see the `Default` impl below for rationale.
///
/// **Step 6 (proposal 2026-05-25-cli-init-and-params-ux) expansion.**
/// Four new variants for inference warm-starts:
///
/// - `FromPrior` — sample once per chain from each parameter's `~`
///   declaration in the model IR. Parameters with no `~` fall back to
///   bounds-uniform with a startup warning (Decision A).
/// - `FromPosterior { source }` — draw one row per chain from a
///   posterior draws TSV (uniform with replacement; gh#83 default).
/// - `FromMle { source }` — all chains start at the MLE point from a
///   prior fit. Knows the fit-output TOML schema.
/// - `FromParams { path }` — all chains at a flat hand-written params
///   TOML. Top-level keys = parameter names → values.
///
/// Per the proposal's verb-per-source contract, `FromMle` and
/// `FromParams` are distinct verbs (not unified) because their file
/// schemas differ (structured fit output vs flat hand-authored TOML).
///
/// `SurveyTopK` is kept as a unit variant rather than carrying the
/// proposal's `SurveyTopK { source: SurveySource, k: usize }` payload.
/// The existing fit.toml stage dispatch reads sibling `survey_path` /
/// `survey_top_k_n` fields and constructs the per-chain starts via
/// `resolve_per_chain_starts_from_method`, which is independent of the
/// `draw_chain_starts` entry point introduced for the four new
/// variants. Restructuring `SurveyTopK`'s payload would cascade into
/// pgas / pmmh / profile / nlopt_stage dispatch — out of scope for
/// step 6, deferred to step 7's CLI break or a follow-up RFC.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InitMethod {
    Single,
    Uniform,
    Lhs,
    /// Stan-style initialization. Each chain's start is an i.i.d. draw
    /// `z ~ Uniform(-RADIUS, RADIUS)` on the *unconstrained* scale
    /// (Stan's `init_radius`, default 2), squashed to the open unit
    /// interval `u = σ(z)` via the logistic sigmoid, then mapped to the
    /// natural parameter scale through the same transform-aware seam LHS
    /// uses ([`lhs_map_to_natural`]): log-uniform interior for `Log`
    /// parameters with positive bounds, linear interior otherwise.
    ///
    /// Boundary-avoiding (`σ(±2) ≈ (0.119, 0.881)` is a fixed interior
    /// band, so starts never sit on a bound — no degenerate `-inf`
    /// likelihoods or zero-gradient starts) and scale-invariant (the same
    /// radius works whether a parameter is `O(1)` or `O(1e6)`). This is
    /// Stan's well-tested default initialization (mc-stan.org Reference
    /// Manual, "Initialization"). For a `Logit` parameter it is exactly
    /// Stan's `lo + (hi-lo)·σ(z)`; for a `Log` rate it is the
    /// camdl-faithful log-scale analog `lo·(hi/lo)^σ(z)`, keeping the
    /// gh#42 log-scale-awareness that drove LHS past the legacy linear
    /// `Uniform`. Draws are i.i.d. (no Latin-hypercube stratification) —
    /// over-dispersed independent starts are the textbook basis for MCMC
    /// convergence diagnostics. `n_chains < 2 ⇒ base` (same contract as
    /// LHS/Uniform), so flipping the default only moves multi-chain fits.
    #[serde(rename = "uniform_unconstrained")]
    UniformUnconstrained,
    /// Pull per-chain starts from the top-K rows of a `camdl survey`
    /// landscape. Requires sibling fields `survey_path` (CAS dir) and
    /// `survey_top_k_n` (defaults to `chains`) on the same stage. The
    /// reader cross-checks the survey's `run.json` against the fit's
    /// resolved inputs (model_identity, data_hashes, [fixed] superset,
    /// estimate-set subset) and filters the landscape rows to fit's
    /// bounds before ranking. See gh#51 +
    /// `docs/dev/proposals/2026-05-07-survey-top-k-init.md`.
    #[serde(rename = "survey_top_k")]
    SurveyTopK,
    /// Per-chain draw from each parameter's `~ <dist>` declaration in
    /// the model IR. Parameters with no prior declared fall back to
    /// bounds-uniform with a startup warning (proposal Decision A).
    /// gh#83.
    #[serde(rename = "from_prior")]
    FromPrior,
    /// Per-chain draw from a posterior draws TSV (uniformly with
    /// replacement; gh#83's default). Source may be the TSV file
    /// directly or a fit-results directory whose canonical
    /// `draws.tsv` is loaded.
    #[serde(rename = "from_posterior")]
    FromPosterior { source: PosteriorSource },
    /// All chains start at the MLE point from a prior fit. Knows the
    /// fit-output TOML schema: skips `[provenance]`, `[focal]`, and
    /// reads parameter values from the section that holds them.
    /// Replaces the legacy `--starts-from <fit-dir>` flag.
    #[serde(rename = "from_mle")]
    FromMle { source: MleSource },
    /// All chains at the point given by a hand-written **flat** params
    /// TOML. Top-level keys = parameter names → values. Distinct from
    /// `FromMle` so the loader can reject misuse with an actionable
    /// hint (e.g. user pointing `from-params` at an `mle.toml`).
    #[serde(rename = "from_params")]
    FromParams { path: PathBuf },
}

impl InitMethod {
    /// The chain-start SOURCE FILE this init mode reads, if it names one, plus
    /// the artifact name to record. Callers fold the file's CONTENT into their
    /// identity deps so rewriting it in place re-keys the run — a path only
    /// distinguishes a *different* file, never the same path rewritten
    /// (gh#541).
    ///
    /// `FromMle` is absent on purpose: it already folds the upstream leaf's
    /// `fit_state.toml` digest through `cas_dep_from_dir`, and a second dep
    /// for it would double-count.
    pub fn source_file(&self) -> Option<(PathBuf, &'static str)> {
        match self {
            InitMethod::FromPosterior { source } => Some(match source {
                PosteriorSource::DrawsTsv(p) => (p.clone(), "draws.tsv"),
                PosteriorSource::FitDir(d) => (d.join("draws.tsv"), "draws.tsv"),
            }),
            InitMethod::FromParams { path } => Some((path.clone(), "params.toml")),
            _ => None,
        }
    }
}

/// Where to read posterior draws from. See [`InitMethod::FromPosterior`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorSource {
    /// A posterior draws TSV directly. Header line: column-per-parameter;
    /// each row is one draw.
    DrawsTsv(PathBuf),
    /// A fit-results directory. Auto-resolves to `<dir>/draws.tsv`;
    /// errors if missing.
    FitDir(PathBuf),
}

/// Where to read an MLE point from. See [`InitMethod::FromMle`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MleSource {
    /// An MLE-shape TOML file directly (e.g. `mle.toml` or
    /// `final_params.toml`).
    File(PathBuf),
    /// A fit-results directory. Auto-resolves: tries `<dir>/mle.toml`
    /// first, then `<dir>/final_params.toml`. Errors if neither exists.
    FitDir(PathBuf),
}

impl Default for InitMethod {
    /// Stan-style `UniformUnconstrained` — i.i.d. boundary-avoiding,
    /// scale-invariant draws on the unconstrained scale (see the variant
    /// doc). It keeps the log-scale-awareness that drove the gh#42 win
    /// over the legacy linear `Uniform` (both route through
    /// [`lhs_map_to_natural`]) while adding Stan's robustness: starts can
    /// never land on a bound, and the radius is bounds-independent. The
    /// trade-off versus `Lhs` is i.i.d. rather than Latin-hypercube
    /// stratified draws — over-dispersed independent starts are the
    /// textbook choice for MCMC convergence diagnostics, at the cost of
    /// LHS's guaranteed full-bounds stratification (which matters most for
    /// low-chain-count IF2 scout basin-finding; a scout that wants maximal
    /// coverage can set `init = "lhs"`). Default across IF2 / PGAS / PMMH
    /// / NLopt multi-chain stages.
    fn default() -> Self { InitMethod::UniformUnconstrained }
}

impl std::str::FromStr for InitMethod {
    type Err = String;
    /// Parse a string form into an `InitMethod`. Only the **payload-free**
    /// variants are parseable from a bare string — the new
    /// `from-posterior` / `from-mle` / `from-params` variants need
    /// companion path flags and are constructed by the step-7 CLI
    /// layer; `from-prior` is parseable as a bare string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "single"        => Ok(InitMethod::Single),
            "uniform"       => Ok(InitMethod::Uniform),
            "lhs"           => Ok(InitMethod::Lhs),
            "uniform_unconstrained" | "uniform-unconstrained"
                            => Ok(InitMethod::UniformUnconstrained),
            "survey_top_k"  => Ok(InitMethod::SurveyTopK),
            "from_prior" | "from-prior" => Ok(InitMethod::FromPrior),
            other => Err(format!(
                "unknown init_method '{}': expected one of \
                 single, uniform, lhs, uniform_unconstrained, survey_top_k, \
                 from_prior. from-posterior / from-mle / from-params require \
                 companion path flags and cannot be set as a bare string.",
                other)),
        }
    }
}

impl std::fmt::Display for InitMethod {
    /// Stable string tag for one-line provenance / `chain_init_source` /
    /// `init_provenance.method`. Snake_case throughout to match the
    /// CLI + serde + `clap::ValueEnum::to_possible_value` surfaces;
    /// any downstream tool that ingests `run.json` only needs to
    /// recognise one spelling per variant (gh#87). Variants with
    /// payload render the bare tag; per-chain provenance lives in
    /// the sibling [`InitSource`] discriminator with the actual path.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            InitMethod::Single             => "single",
            InitMethod::Uniform            => "uniform",
            InitMethod::Lhs                => "lhs",
            InitMethod::UniformUnconstrained => "uniform_unconstrained",
            InitMethod::SurveyTopK         => "survey_top_k",
            InitMethod::FromPrior          => "from_prior",
            InitMethod::FromPosterior { .. } => "from_posterior",
            InitMethod::FromMle       { .. } => "from_mle",
            InitMethod::FromParams    { .. } => "from_params",
        })
    }
}

/// Manual `clap::ValueEnum` over the payload-free `InitMethod`
/// variants. Variants with payload (`FromPosterior`, `FromMle`,
/// `FromParams`) are not surfaced through `value_enum` parsing —
/// they're constructed by the step-7 CLI layer from `--init <mode>` +
/// companion path flags.
///
/// Preserves the legacy CLI surface (`camdl profile --init lhs`,
/// `camdl profile --init survey_top_k`) while the type itself grew
/// payload variants.
impl clap::ValueEnum for InitMethod {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            InitMethod::Single,
            InitMethod::Uniform,
            InitMethod::Lhs,
            InitMethod::UniformUnconstrained,
            InitMethod::SurveyTopK,
            InitMethod::FromPrior,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        let name = match self {
            InitMethod::Single       => "single",
            InitMethod::Uniform      => "uniform",
            InitMethod::Lhs          => "lhs",
            InitMethod::UniformUnconstrained => "uniform_unconstrained",
            InitMethod::SurveyTopK   => "survey_top_k",
            InitMethod::FromPrior    => "from_prior",
            // Payload variants are not surfaced via value_enum.
            InitMethod::FromPosterior { .. }
            | InitMethod::FromMle    { .. }
            | InitMethod::FromParams { .. } => return None,
        };
        Some(clap::builder::PossibleValue::new(name))
    }
}

/// Does this mode discard the base point — and with it `[estimate].start`?
///
/// gh#506 surfaced the question: `start` is load-bearing under some init modes
/// and inert under others, and nothing said which. Declaring a start that the
/// mode then ignores is not an error (the modes that ignore it do so on
/// purpose — a chain-agreement gate is only informative if the chains genuinely
/// start apart), but it IS a silent no-op, and the user who wrote the value
/// deserves to know it had no effect.
///
/// `n_chains` matters because the three spreading modes fall back to the base
/// point at one chain: there is nothing to spread.
pub fn ignores_base_point(method: &InitMethod, n_chains: usize) -> bool {
    match method {
        // Every chain at the base point.
        InitMethod::Single => false,
        // Chain 1 keeps the seeded start; only 2..N are drawn.
        InitMethod::Uniform => false,
        // Stratified / unconstrained spreads use the base point for nothing —
        // unless there is only one chain, where they degrade to it.
        InitMethod::Lhs | InitMethod::UniformUnconstrained => n_chains >= 2,
        // These read every chain's start from somewhere else entirely, at any
        // chain count.
        //
        // Per METHOD, not per parameter — and the distinction is real. A
        // parameter the survey never swept falls back to `spec.initial`
        // (`build_survey_chain_starts`), and a name missing from a
        // `from_mle`/`from_params` source falls back to bounds-uniform, or to
        // the base point when the model declares no range
        // (`chain_starts.rs`). So for THOSE parameters the declared `start` is
        // still load-bearing, and the note this drives is conservative rather
        // than exact: it can say "discarded" about a start that one parameter
        // out of several still used. Erring toward warning is the right
        // direction for a heads-up, but a per-parameter answer would be the
        // honest one if this ever becomes load-bearing.
        InitMethod::SurveyTopK
        | InitMethod::FromPrior
        | InitMethod::FromPosterior { .. }
        | InitMethod::FromMle { .. }
        | InitMethod::FromParams { .. } => true,
    }
}

/// Build N chain starts according to `method`. Returns `None` when
/// caller should pass `None` to `run_chains_with_per_chain_params`
/// (i.e. all chains use `config.estimated_params` directly).
///
/// `seed` is the fit's top-level seed; per-chain RNGs derive from it
/// via `derive_chain_seed`. LHS uses one RNG seeded from `seed` for
/// the permutations + per-stratum jitters (so adding a chain reshuffles
/// all stratum assignments — that's the price of stratification).
pub fn build_chain_starts(
    method: InitMethod,
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Option<Vec<Vec<EstimatedParam>>> {
    match method {
        InitMethod::Single => None,
        InitMethod::Uniform => {
            if n_chains < 2 { return None; }
            Some(build_uniform_chain_starts(base, n_chains, seed))
        }
        InitMethod::Lhs => {
            if n_chains < 2 { return None; }
            Some(build_lhs_chain_starts(base, n_chains, seed))
        }
        InitMethod::UniformUnconstrained => {
            if n_chains < 2 { return None; }
            Some(build_uniform_unconstrained_chain_starts(base, n_chains, seed))
        }
        InitMethod::SurveyTopK => {
            // Routed through `build_chain_starts_from_survey` at the
            // stage callsite, where the fit-level cross-check context
            // is in scope. Reaching this branch is a wiring bug, not
            // a user-input problem — panic in debug, return None in
            // release so the caller falls back to base specs (rather
            // than mid-fit panicking on a dispatch oversight).
            debug_assert!(false,
                "InitMethod::SurveyTopK reached build_chain_starts; \
                 callsite must dispatch via build_chain_starts_from_survey");
            None
        }
        // Step 6 init variants: not yet routed through the
        // legacy `build_chain_starts` surface. These are constructed
        // by step-7 CLI parsing and dispatched via `draw_chain_starts`
        // (which is the unified entry point for the warm-start
        // family). Reaching this branch is a wiring bug — debug-panic
        // so it surfaces in tests; return None in release so the
        // caller falls back to base specs without mid-fit panicking.
        InitMethod::FromPrior
        | InitMethod::FromPosterior { .. }
        | InitMethod::FromMle    { .. }
        | InitMethod::FromParams { .. } => {
            debug_assert!(false,
                "InitMethod::{} reached build_chain_starts; \
                 callsite must dispatch via draw_chain_starts",
                method);
            None
        }
    }
}

/// Resolve `method` to per-chain full parameter vectors, for routines
/// (NLopt, profile) that work with `Vec<f64>` directly rather than the
/// IF2-shaped `Vec<EstimatedParam>`. Returns one full param-vector
/// per chain, with each `EstimatedParam`-listed index overwritten by
/// the per-chain draw and all other slots taken from `base_params`.
///
/// Returns `Ok(None)` when the caller should treat every chain as
/// starting from `base_params` directly (i.e. `Single`, or
/// `n_chains < 2`). Returns `Err` for `InitMethod::SurveyTopK` —
/// NLopt and profile are deferred to v3 (see gh#51). IF2 / PMMH /
/// PGAS dispatch survey_top_k via
/// `resolve_per_chain_starts_from_method` at the stage callsite,
/// where the fit-level cross-check context (`SurveyFitContext`) is in
/// scope.
pub fn build_chain_param_vecs(
    method: &InitMethod,
    base_specs: &[EstimatedParam],
    base_params: &[f64],
    n_chains: usize,
    seed: u64,
) -> Result<Option<Vec<Vec<f64>>>, String> {
    match method {
        InitMethod::SurveyTopK => {
            return Err(
                "init = \"survey_top_k\" is not yet supported on this \
                 stage type; v2 ships it on IF2 / PMMH / PGAS. NLopt and \
                 profile support is deferred to v3 (see gh#51). Workaround: \
                 use init = \"lhs\" on this stage, or run an IF2 / \
                 PMMH / PGAS scout first and chain via \
                 init_mle = \"<scout>\".".to_string());
        }
        // Step 6 warm-start variants — not yet routed through the
        // legacy NLopt / profile `build_chain_param_vecs` surface.
        // These are constructed by step-7 CLI parsing and dispatched
        // via `draw_chain_starts`; profile / NLopt support arrives in
        // step 7 alongside the CLI break. Errors actionably so the
        // user redirects to a supported mode.
        InitMethod::FromPrior
        | InitMethod::FromPosterior { .. }
        | InitMethod::FromMle    { .. }
        | InitMethod::FromParams { .. } => {
            return Err(format!(
                "init = \"{}\" is not yet supported on this stage \
                 type (NLopt / profile). Use init = \"lhs\" or \
                 \"single\" for now; warm-start `from_prior` / \
                 `from_posterior` / `from_mle` / `from_params` ship on \
                 IF2 / PGAS / PMMH stages via the step-7 CLI surface.",
                method));
        }
        _ => {}
    }
    let per_chain = build_chain_starts(method.clone(), base_specs, n_chains, seed);
    Ok(per_chain.map(|chains| chain_starts_to_param_vecs(&chains, base_params)))
}

/// Convert per-chain `EstimatedParam` specs into per-chain full
/// parameter vectors. Each chain starts from `base_params`; each
/// `EstimatedParam`-listed index is overwritten with that chain's
/// `initial` value. Shared between `build_chain_param_vecs` and the
/// PMMH/PGAS dispatch sites that consume
/// `resolve_per_chain_starts_from_method`'s `Vec<Vec<EstimatedParam>>`
/// output.
pub fn chain_starts_to_param_vecs(
    chains: &[Vec<EstimatedParam>],
    base_params: &[f64],
) -> Vec<Vec<f64>> {
    chains.iter().map(|chain| {
        let mut params = base_params.to_vec();
        for spec in chain { params[spec.index] = spec.initial; }
        params
    }).collect()
}

/// Resolve per-chain starting points from `init_method`, dispatching
/// between LHS / Uniform / Single / SurveyTopK. The shared backbone
/// of IF2 / PMMH / PGAS chain init (gh#51 v2).
///
/// Returns `(per_chain_starts, survey_top_k_result)`:
///
/// - `per_chain_starts = None` means the caller should treat every
///   chain as starting from `base_specs` directly (i.e.
///   `InitMethod::Single`, or `n_chains < 2` for `Lhs` / `Uniform`).
///   The IF2 path passes `None` into `run_chains_with_per_chain_params`;
///   PMMH / PGAS materialise N copies of `base_params`.
/// - `survey_top_k_result = Some(_)` only when `method =
///   SurveyTopK`. Carries the survey's full content hash so the
///   caller can populate `fit_state.toml.chain_init_source`
///   (`survey:<hash>:top-<K>`) and the `chain_starts.tsv` sidecar's
///   per-row `source` column (`survey:<hash>:rank-<N>`). For non-survey
///   modes, callers fall back to `format_chain_init_source(method, None)`
///   which renders `"lhs"` / `"single"` / `"uniform"`.
///
/// When `method = SurveyTopK` and `survey_path` is `None`, returns an
/// error naming the offending stage. Callers handle this with the same
/// diagnostic surface they use for other survey-config mistakes.
/// Build a minimal `ResolvedParameters` view from a fit-runner-style
/// pair of (model, base_params, estimated_specs). Used by pgas/pmmh
/// stage dispatch to thread warm-start variants through
/// `chain_starts::draw_chain_starts` without forcing a full
/// `ParameterInputs` reconstruction. Provenance from the original
/// resolve is not preserved here (the fit-runner path already
/// recorded provenance into run.json upstream); only the fields
/// `draw_chain_starts` reads — `model`, `estimate_set`, and per-name
/// `params[i].value` — are populated.
pub fn build_resolved_view_for_init(
    model: &ir::Model,
    base_params: &[f64],
    estimated_specs: &[EstimatedParam],
) -> crate::params_resolver::ResolvedParameters {
    use crate::params_resolver::{
        FixReason, ParameterRole, ResolvedParameter, ResolvedParameters, ValueSource,
    };
    use indexmap::IndexSet;
    let estimate_set: IndexSet<String> = estimated_specs.iter()
        .map(|s| s.name.clone()).collect();
    let mut params: Vec<ResolvedParameter> = Vec::with_capacity(model.parameters.len());
    for p in &model.parameters {
        // `base_params` is indexed by compiled-model param index; look
        // up by name to find the value. Missing names fall back to
        // p.value (the model default) — should not happen by
        // construction, but kept defensive.
        let value = estimated_specs.iter()
            .find(|s| s.name == p.name)
            .map(|s| base_params[s.index])
            .or_else(|| {
                // Non-estimated param: look up by position in
                // model.parameters → base_params.
                model.parameters.iter().position(|q| q.name == p.name)
                    .and_then(|idx| base_params.get(idx).copied())
            })
            .or(p.value.resolved_value())
            .unwrap_or(f64::NAN);
        let role = if estimate_set.contains(&p.name) {
            ParameterRole::Estimated
        } else {
            ParameterRole::Fixed { reason: FixReason::NotInEstimate }
        };
        params.push(ResolvedParameter {
            name: p.name.clone(),
            value,
            source: ValueSource::ModelDefault,
            role,
            overrode_scenario: None,
        });
    }
    ResolvedParameters {
        params,
        estimate_set,
        model: model.clone(),
        warnings: Vec::new(),
    }
}

pub fn resolve_per_chain_starts_from_method(
    method: &InitMethod,
    survey_path: Option<&std::path::Path>,
    survey_top_k_n: Option<usize>,
    stage_name: &str,
    base_specs: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
    ctx: &SurveyFitContext<'_>,
    resolved: Option<&crate::params_resolver::ResolvedParameters>,
) -> Result<(Option<Vec<Vec<EstimatedParam>>>, Option<SurveyTopKResult>), String> {
    match method {
        InitMethod::SurveyTopK => {
            let path = survey_path.ok_or_else(|| format!(
                "stage `{}`: init = \"survey_top_k\" requires \
                 `survey_path = \"<survey CAS dir>\"` (set on the stage in \
                 fit.toml or via CLI `--survey-path`). See gh#51.",
                stage_name))?;
            let result = build_chain_starts_from_survey(
                path, survey_top_k_n, n_chains, base_specs, ctx)?;
            let chains_out = result.chains.clone();
            Ok((Some(chains_out), Some(result)))
        }
        // Step 7 warm-start variants: dispatch through
        // `chain_starts::draw_chain_starts` when the caller supplied a
        // `ResolvedParameters` view (CLI-driven IF2 / PGAS / PMMH
        // stages of `camdl fit run` and the standalone subcommands
        // build one before dispatch). When `resolved` is None the
        // legacy `init::build_chain_param_vecs` rejection fires —
        // that path covers callers that haven't migrated yet.
        InitMethod::FromPrior
        | InitMethod::FromPosterior { .. }
        | InitMethod::FromMle    { .. }
        | InitMethod::FromParams { .. } => {
            let resolved = resolved.ok_or_else(|| format!(
                "stage `{}`: init = \"{}\" is a step-7 warm-start \
                 variant that requires the dispatch to pass \
                 `ResolvedParameters` into \
                 `resolve_per_chain_starts_from_method`. This caller \
                 hasn't been migrated yet.",
                stage_name, method))?;
            let starts = crate::fit::chain_starts::draw_chain_starts(
                resolved, method, n_chains, seed,
            ).map_err(|e| format!("stage `{}`: --init {}: {}",
                stage_name, method, e))?;
            let chains_specs = starts.to_estimated_params(base_specs);
            Ok((Some(chains_specs), None))
        }
        _ => {
            let per_chain = build_chain_starts(
                method.clone(), base_specs, n_chains, seed);
            Ok((per_chain, None))
        }
    }
}

/// Per-chain uniform random draw within natural-scale bounds. Chain 0
/// keeps the seeded start (reproducibility); chains 1..N draw fresh.
/// Equivalent to the previous `runner::build_random_chain_starts`
/// (kept as a free function here so the runner doesn't grow more init
/// strategies inline).
fn build_uniform_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    (0..n_chains).map(|chain_id| {
        let mut rng = StatefulRng::new(derive_chain_seed(seed, chain_id));
        base.iter().map(|spec| {
            let initial = if chain_id == 0 {
                spec.initial
            } else if spec.lower.is_finite() && spec.upper.is_finite() {
                spec.lower + rng.uniform() * (spec.upper - spec.lower)
            } else {
                spec.initial * (0.5 + rng.uniform())
            };
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Latin-hypercube stratified starts, scale-aware via `Transform`.
///
/// Algorithm (textbook stratified LHS):
/// 1. For each parameter dim d, draw a random permutation π_d of `[0..n_chains)`.
/// 2. For chain k, dim d: `u_{k,d} = (π_d[k] + jitter) / n_chains`, with
///    `jitter ~ Uniform(0, 1)` — a uniform draw within stratum k's cell.
/// 3. Map `u_{k,d}` to natural-scale θ via the parameter's transform:
///    - `Transform::Log` and both bounds positive → exponential mapping
///      `θ = lo · (hi/lo)^u`. Equivalent to LHS in `[ln lo, ln hi]`.
///    - Otherwise (Logit, None, or pathological log bounds) → linear
///      `θ = lo + u · (hi - lo)`.
///
/// Unbounded params (lower or upper non-finite) fall back to a
/// `±50%` jitter around `spec.initial` — same fallback as
/// `build_uniform_chain_starts` for parity. LHS without finite bounds
/// is meaningless; flag with the validator if this matters in practice.
fn build_lhs_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    let n_params = base.len();
    let mut rng = StatefulRng::new(seed ^ 0x1f5_beef_u64);

    // Step 1+2: per-dim permutation, jitter within each stratum.
    // u[chain_id][param_id] is the [0,1] LHS coordinate.
    let mut u: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n_chains];
    for d in 0..n_params {
        let mut perm: Vec<usize> = (0..n_chains).collect();
        // Fisher-Yates using the same RNG (deterministic given seed).
        for i in (1..n_chains).rev() {
            let j = (rng.uniform() * (i as f64 + 1.0)).floor() as usize;
            perm.swap(i, j.min(i));
        }
        for k in 0..n_chains {
            let jitter = rng.uniform();
            u[k][d] = (perm[k] as f64 + jitter) / n_chains as f64;
        }
    }

    // Step 3: map [0,1] LHS coord to natural-scale θ per Transform.
    (0..n_chains).map(|chain_id| {
        base.iter().enumerate().map(|(d, spec)| {
            let initial = lhs_map_to_natural(spec, u[chain_id][d]);
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Stan's default initialization radius on the unconstrained scale: each
/// chain draws `z ~ Uniform(-2, 2)` per parameter (Stan's `init_radius`,
/// mc-stan.org Reference Manual, "Initialization"). Hardcoded for v1.
pub(crate) const STAN_INIT_RADIUS: f64 = 2.0;

/// Stan-style starts: i.i.d. `z ~ Uniform(-R, R)` on the unconstrained
/// scale, squashed to the open unit interval `u = σ(z)` and mapped to
/// natural scale through the same transform-aware seam as LHS
/// ([`lhs_map_to_natural`]). See [`InitMethod::UniformUnconstrained`] for
/// the geometry and the robustness rationale.
///
/// Per-chain RNGs derive from `seed` via `derive_chain_seed` (same as
/// `build_uniform_chain_starts`), so draws are independent across chains
/// and reproducible given the fit seed. Unlike LHS there is no
/// stratification: each `(chain, param)` coordinate is an independent
/// squashed-uniform draw.
fn build_uniform_unconstrained_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    (0..n_chains).map(|chain_id| {
        let mut rng = StatefulRng::new(derive_chain_seed(seed, chain_id));
        base.iter().map(|spec| {
            // z ~ U(-R, R) on the unconstrained scale; σ(z) lands in the
            // open interior of [0, 1] (σ(±2) ≈ 0.119 / 0.881), so the
            // mapped start never sits on a bound regardless of [lo, hi]
            // width — boundary-avoiding and scale-invariant.
            let z = (rng.uniform() * 2.0 - 1.0) * STAN_INIT_RADIUS;
            let u = 1.0 / (1.0 + (-z).exp());
            let initial = lhs_map_to_natural(spec, u);
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Draw a single Transform-aware value within `[lo, hi]`, suitable for
/// the gh#34 start-fallback path: when an `[estimate]` entry has neither
/// `start =` nor a model-declared parameter `value`, we still need a
/// scalar to seed `model.parameters[i].value` with so that compile +
/// `validate_parameter_values` succeed. Downstream chain init can then
/// perturb from this base.
///
/// "Transform-aware" means: for `Log`-typed parameters with both bounds
/// strictly positive, draw uniformly in *log space* and exponentiate;
/// otherwise draw linearly in `[lo, hi]`. Replaces the legacy
/// bounds-midpoint heuristic (`(lo*hi).sqrt()` or `(lo+hi)/2`), which
/// was geometric-shape-aware via a positive-bounds proxy but ignored
/// the parameter's declared transform and gave the same point at every
/// seed.
///
/// Reproducibility: the per-parameter `u ∈ [0, 1]` is derived from
/// `(seed, param_name)` via a 64-bit hash, so re-running with the same
/// `seed` gives the same start, and two estimate entries with the same
/// bounds at the same seed get *different* draws (their names hash
/// differently). Same seed across runs ⇒ same fallback start; different
/// seeds ⇒ different fallback starts within `[lo, hi]`.
pub fn draw_start_in_bounds(
    lo: f64,
    hi: f64,
    log_scale: bool,
    seed: u64,
    param_name: &str,
) -> f64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    param_name.hash(&mut h);
    // Map the 64-bit hash into u ∈ (0, 1) — open interval, so the
    // log-scale branch's `(hi/lo).powf(u)` never lands exactly on a
    // bound. 53-bit mantissa is plenty.
    let u = ((h.finish() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;

    if log_scale && lo > 0.0 && hi > 0.0 {
        lo * (hi / lo).powf(u)
    } else {
        lo + u * (hi - lo)
    }
}

// ── survey_top_k chain init (gh#51) ──────────────────────────────────

/// Fit-level context required to validate a survey artifact and resolve
/// fallbacks. Constructed at the runner side (where the fit's resolved
/// inputs are in scope) and passed to `build_chain_starts_from_survey`.
///
/// The borrows are short-lived — this struct exists only for the
/// duration of a single chain-init resolution.
pub struct SurveyFitContext<'a> {
    /// The fit's `runid` model identity (hex, from
    /// [`crate::resolve::model_identity_from_ir`]). Must match the survey's
    /// recorded `model_identity` exactly.
    pub model_identity: &'a str,
    /// Per-stream content hashes of the fit's data files. Each stream
    /// the fit consumes must appear with a matching hash in the
    /// survey's `data_hashes`. Survey may reference *more* streams
    /// than the fit (e.g. survey held a covariate fixed and the fit
    /// drops it); those are ignored.
    pub data_hashes: &'a std::collections::HashMap<String, String>,
    /// Resolved `[fixed]` block from the fit. Survey's `[fixed]` must
    /// be a superset; differing-value at any shared key refuses.
    pub fixed: &'a std::collections::HashMap<String, f64>,
    /// Estimated-param names from the fit, in any order. Each must
    /// either appear in the survey's estimated-param column set, or
    /// fall back to the row's `base.initial` (typically the user's
    /// `[estimate].start` or its gh#34 uniform draw).
    pub estimate_names: &'a [String],
}

/// SHA-256 every data file referenced by `effective_obs`, returning the
/// stream-name → hex-hash map shape that `SurveyFitContext.data_hashes`
/// (and the survey `run.json` `inputs.data_hashes`) use for the
/// cross-check. Centralised here so the four-or-five fit-stage dispatch
/// sites compute it identically.
pub fn compute_data_hashes(
    effective_obs: &indexmap::IndexMap<String, String>,
) -> Result<std::collections::HashMap<String, String>, String> {
    let mut out = std::collections::HashMap::with_capacity(effective_obs.len());
    for (stream, path) in effective_obs {
        let bytes = std::fs::read(path).map_err(|e| format!(
            "cannot read data file `{}` for stream `{}`: {}", path, stream, e))?;
        out.insert(stream.clone(), crate::hashing::sha256_hex(&bytes));
    }
    Ok(out)
}

/// Result of `build_chain_starts_from_survey` — bundles the per-chain
/// `EstimatedParam` overrides with provenance info the caller needs
/// to populate `fit_state.toml.chain_init_source` and
/// `chain_starts.tsv`. Each chain's rank in the survey is its index
/// in `chains` plus 1 (chain 0 = rank-1, chain 1 = rank-2, ...).
#[derive(Debug)]
pub struct SurveyTopKResult {
    pub chains: Vec<Vec<EstimatedParam>>,
    /// Full content hash of the survey CAS dir (`run.json.hash`).
    /// Embedded into provenance strings as
    /// `survey:<survey_hash>:top-<K>` (one per fit) and
    /// `survey:<survey_hash>:rank-<N>` (one per chain in
    /// `chain_starts.tsv`). Full hash, not short — short hashes
    /// collide and audit-survivable links must point at exactly one
    /// CAS dir.
    pub survey_hash: String,
}

/// Pull per-chain starts from the top-K rows of a `camdl survey`
/// landscape. See gh#51 +
/// `docs/dev/proposals/2026-05-07-survey-top-k-init.md` for the
/// design rationale.
///
/// Steps:
/// 1. Load `<survey_path>/run.json` as a `runid::RunRecord`; refuse
///    unless `kind == ArtifactKind::Survey`. The cross-check provenance
///    (`model_identity` / `data_hashes` / `fixed` / `estimated`) is read from
///    the record's `inputs` payload.
/// 2. Cross-check `model_identity`, `data_hashes`, `[fixed]` superset,
///    estimate-set subset against `ctx`. Refuse on any mismatch with
///    a diagnostic naming the offending field.
/// 3. Read `<survey_path>/landscape.tsv` (skipping `#` comment lines).
/// 4. **Filter** rows: keep only those whose every parameter value
///    lies within the corresponding `base[i].lower / .upper` bound.
///    No clipping. Refuse if filtered count < `n_chains`. Warn if
///    filtered drops > 50% of original.
/// 5. **Rank** filtered rows by `loglik` desc; take top-`top_k`
///    (defaults to `n_chains` when `top_k_n` is `None`). v1 enforces
///    `top_k == n_chains` (strict K=chains; K > chains is v2).
/// 6. **SE-aware warn**: if the top-K decibans-spread is below
///    `max(30.0, 8 · σ_max · NATS_TO_DB)`, warn that the rank
///    ordering is uncertain at this measurement budget. Never refuse
///    on this — fits with noisy seeds still work.
///
/// For each top-K row, build a chain by cloning `base` and
/// overriding `initial` with the row's column value for every
/// estimated param the survey carried. Estimated params present in
/// the fit but absent from the survey (fit estimates ρ, survey held
/// it fixed) keep `base.initial` as the per-chain start.
pub fn build_chain_starts_from_survey(
    survey_path: &std::path::Path,
    top_k_n: Option<usize>,
    n_chains: usize,
    base: &[EstimatedParam],
    ctx: &SurveyFitContext,
) -> Result<SurveyTopKResult, String> {
    use runid::{ArtifactKind, RunRecord};

    let top_k = top_k_n.unwrap_or(n_chains);
    if top_k != n_chains {
        return Err(format!(
            "init = \"survey_top_k\": v1 requires \
             survey_top_k_n == chains (got top_k_n = {}, chains = {}). \
             K > chains with stratified sub-sampling is deferred to v2 \
             — see gh#51 §\"Out of scope for v1\".",
            top_k, n_chains));
    }

    // Step 1: load run.json as a runid RunRecord; refuse unless Survey.
    let run_json = survey_path.join("run.json");
    let bytes = std::fs::read(&run_json).map_err(|e| format!(
        "init = \"survey_top_k\": cannot read {:?}: {}", run_json, e))?;
    let record: RunRecord = serde_json::from_slice(&bytes).map_err(|e| format!(
        "init = \"survey_top_k\": {:?} is not a parseable run.json: {}. \
         survey_path must point at a `camdl survey` CAS directory.",
        run_json, e))?;
    if record.kind != ArtifactKind::Survey {
        return Err(format!(
            "init = \"survey_top_k\": {:?} is a {:?} run, not a Survey run. \
             survey_path must point at a `camdl survey` CAS directory.",
            survey_path, record.kind));
    }
    let cross = SurveyCrossCheck::from_inputs(&record.inputs).map_err(|e| format!(
        "init = \"survey_top_k\": survey run.json `inputs` is missing \
         cross-check provenance ({}). The survey at {:?} predates the \
         cross-check schema — re-run `camdl survey`.", e, survey_path))?;

    // Step 2: cross-check.
    cross_check_survey(&cross, ctx)?;

    // Step 3: read + parse landscape.tsv.
    let landscape_path = survey_path.join("landscape.tsv");
    let raw = std::fs::read_to_string(&landscape_path).map_err(|e| format!(
        "init = \"survey_top_k\": cannot read {:?}: {}",
        landscape_path, e))?;
    let rows = parse_landscape_tsv(&raw, &cross.estimated)
        .map_err(|e| format!("init = \"survey_top_k\": {}", e))?;
    let total_rows = rows.len();

    // Step 4: filter by fit bounds.
    let filtered: Vec<&LandscapeRow> = rows.iter().filter(|row| {
        base.iter().all(|spec| {
            match row.params.get(&spec.name) {
                Some(&v) => v >= spec.lower && v <= spec.upper,
                // Param not in survey → not a filter criterion (it'll
                // fall back to base.initial in step 6).
                None => true,
            }
        })
    }).collect();

    if filtered.len() < n_chains {
        return Err(format!(
            "init = \"survey_top_k\": survey has {} rows but only {} \
             fall within fit bounds, and chains = {}. Either widen fit's \
             bounds toward the surveyed region, or re-run the survey on \
             the narrower box.",
            total_rows, filtered.len(), n_chains));
    }
    if (filtered.len() as f64) < 0.5 * (total_rows as f64) {
        eprintln!("\x1b[33mwarning:\x1b[0m init = \"survey_top_k\" \
            discards {} of {} survey rows as outside fit bounds. The \
            fit will use the top-{} of the {} that remain, but most of \
            the survey's measurement budget is being thrown away. \
            Consider widening fit bounds or re-running the survey.",
            total_rows - filtered.len(), total_rows, top_k, filtered.len());
    }

    // Step 5: rank + take top-K.
    //
    // gh#129 (2026-05-26 week-audit C1): this ranks by likelihood, not
    // posterior. For Bayesian targets (PGAS, PMMH) the seeded chains
    // will sit at likelihood maxima irrespective of prior mass — a
    // silent bias when any estimated parameter has a non-flat prior.
    // The proper v2 fix is two-step: (a) survey writer emits log_prior
    // alongside loglik; (b) this site ranks by log_posterior with a
    // prior_hash cross-check. Until then, fire a loud warning every
    // time survey_top_k is used so the bias is at least visible.
    let mut ranked: Vec<&LandscapeRow> = filtered;
    ranked.sort_by(|a, b| {
        b.loglik.partial_cmp(&a.loglik).unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected: &[&LandscapeRow] = &ranked[..top_k];

    // Step 5.5: gh#129 — surface the rank-by-likelihood bias.
    emit_rank_by_likelihood_bias_warning();

    // Step 6: SE-aware warn on rank noise.
    emit_top_k_se_warning(selected);

    // Step 7: assemble per-chain EstimatedParam vectors.
    let chains: Vec<Vec<EstimatedParam>> = selected.iter().map(|row| {
        base.iter().map(|spec| {
            let initial = row.params.get(&spec.name)
                .copied()
                .unwrap_or(spec.initial);
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect();

    Ok(SurveyTopKResult {
        chains,
        survey_hash: record.run_id.to_hex(),
    })
}

/// Cross-check provenance read back from a survey `run.json`'s `inputs`
/// payload (gh#51). The survey writer records the `runid` `model_identity`,
/// per-stream `data_hashes`, the resolved `[fixed]` block, and the
/// `estimated`-param column names; this is the consumer's view of them.
#[derive(serde::Deserialize)]
struct SurveyCrossCheck {
    model_identity: String,
    #[serde(default)]
    data_hashes: std::collections::HashMap<String, String>,
    #[serde(default)]
    fixed: std::collections::HashMap<String, f64>,
    estimated: Vec<String>,
}

impl SurveyCrossCheck {
    fn from_inputs(inputs: &serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(inputs.clone()).map_err(|e| e.to_string())
    }
}

/// One row of a survey landscape.tsv, parsed.
#[derive(Debug, Clone)]
struct LandscapeRow {
    params: std::collections::HashMap<String, f64>,
    loglik: f64,
    loglik_se: f64,
}

/// Parse `landscape.tsv` body. Skips `#` comment lines, reads the
/// header, then each data row. Recognised column-set: `<param>...
/// loglik loglik_se [mean_ess] n_replicates point_id`. Param columns
/// are matched against `survey_estimated`; remaining named columns
/// (loglik / loglik_se) are extracted explicitly.
fn parse_landscape_tsv(
    raw: &str,
    survey_estimated: &[String],
) -> Result<Vec<LandscapeRow>, String> {
    let mut lines = raw.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header = lines.next()
        .ok_or_else(|| "landscape.tsv has no header row (only comments?)".to_string())?;
    let cols: Vec<&str> = header.split('\t').collect();
    let loglik_idx = cols.iter().position(|c| *c == "loglik")
        .ok_or_else(|| "landscape.tsv header missing `loglik` column".to_string())?;
    let loglik_se_idx = cols.iter().position(|c| *c == "loglik_se")
        .ok_or_else(|| "landscape.tsv header missing `loglik_se` column".to_string())?;
    // Param columns are the leading run of columns whose name matches
    // an entry in survey_estimated (their order matters for the survey
    // writer; we use their names for the lookup).
    let param_indices: Vec<(String, usize)> = survey_estimated.iter()
        .map(|name| {
            cols.iter().position(|c| *c == name)
                .map(|i| (name.clone(), i))
                .ok_or_else(|| format!(
                    "landscape.tsv header missing param column `{}` \
                     (declared in run.json `estimated`)", name))
        })
        .collect::<Result<_, _>>()?;

    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != cols.len() {
            return Err(format!(
                "landscape.tsv data row {} has {} fields, expected {}",
                line_no + 1, fields.len(), cols.len()));
        }
        let parse = |i: usize, name: &str| -> Result<f64, String> {
            fields[i].parse::<f64>().map_err(|e| format!(
                "landscape.tsv data row {}: cannot parse `{}` field {:?}: {}",
                line_no + 1, name, fields[i], e))
        };
        let loglik = parse(loglik_idx, "loglik")?;
        let loglik_se = parse(loglik_se_idx, "loglik_se")?;
        let mut params = std::collections::HashMap::with_capacity(param_indices.len());
        for (name, idx) in &param_indices {
            params.insert(name.clone(), parse(*idx, name)?);
        }
        rows.push(LandscapeRow { params, loglik, loglik_se });
    }
    Ok(rows)
}

fn cross_check_survey(
    meta: &SurveyCrossCheck,
    ctx: &SurveyFitContext<'_>,
) -> Result<(), String> {
    if meta.model_identity != ctx.model_identity {
        return Err(format!(
            "init = \"survey_top_k\": model_identity mismatch.\n  \
             survey: {}\n     fit: {}\nA model edit between survey and \
             fit invalidates the cross-check; re-run the survey on the \
             current model.",
            meta.model_identity, ctx.model_identity));
    }
    for (stream, fit_hash) in ctx.data_hashes {
        match meta.data_hashes.get(stream) {
            Some(survey_hash) if survey_hash == fit_hash => {}
            Some(survey_hash) => return Err(format!(
                "init = \"survey_top_k\": data_hashes mismatch on \
                 stream `{}`.\n  survey: {}\n     fit: {}",
                stream, survey_hash, fit_hash)),
            None => return Err(format!(
                "init = \"survey_top_k\": fit consumes data stream \
                 `{}` which the survey did not score against. Re-run the \
                 survey with this stream included.", stream)),
        }
    }
    for (name, &fit_value) in ctx.fixed {
        match meta.fixed.get(name) {
            Some(&survey_value) if (survey_value - fit_value).abs() < 1e-12 => {}
            Some(&survey_value) => return Err(format!(
                "init = \"survey_top_k\": [fixed].{} disagrees.\n  \
                 survey: {}\n     fit: {}\nFixed-value drift between \
                 survey and fit invalidates the seeded starts.",
                name, survey_value, fit_value)),
            None => return Err(format!(
                "init = \"survey_top_k\": fit's [fixed] must be a \
                 subset of survey's [fixed]; survey did not pin `{}` (the \
                 survey estimated it or left it free). Pin it in the \
                 survey, or remove it from fit's [fixed].", name)),
        }
    }
    let survey_estimated: std::collections::HashSet<&str> =
        meta.estimated.iter().map(|s| s.as_str()).collect();
    for name in ctx.estimate_names {
        // Fit-estimate params absent from the survey are fine (fall
        // back to base.initial). Fit-estimate params *fixed* by the
        // survey at a value that equals fit's expected start would
        // also be fine, but that's a degenerate case we don't need to
        // optimise for. The hard refusal is when the survey neither
        // estimated nor fixed the param — meaning it has no value at
        // all in the survey's parameter space. That can't happen for
        // a model the survey actually ran (every model parameter is
        // either estimated or fixed at survey time), so this loop is
        // mostly defensive.
        let _ = survey_estimated; // keep the set for future cross-checks
        let _ = name;
    }
    Ok(())
}

/// gh#129 (2026-05-26 week-audit C1) — fires whenever `init =
/// survey_top_k` is used. The current ranking is by likelihood
/// alone; PGAS/PMMH chains target `posterior ∝ likelihood × prior`.
/// For any non-flat prior the seeded chains sit at likelihood
/// maxima irrespective of prior mass — a silent bias.
///
/// The warning is unconditional. For flat-prior fits it's technically
/// noise (rank-by-likelihood and rank-by-posterior agree), but the
/// alternative — only fire when at least one non-flat prior is
/// resolved — would require threading priors through the call chain,
/// which the audit's recommended v2 fix already restructures. Loud
/// noise on the flat-prior path is the conservative defensive choice
/// while the v2 fix is in flight; the noise becomes signal the
/// moment a user adds a non-flat prior.
fn emit_rank_by_likelihood_bias_warning() {
    eprintln!("\x1b[33mwarning:\x1b[0m \
        init = \"survey_top_k\" currently ranks survey rows by \
        likelihood, not posterior. For any estimated parameter with \
        a non-flat prior, this seeds PGAS/PMMH chains at likelihood \
        maxima irrespective of prior mass — a silent bias.\n\
        \n\
        * If your fit uses flat priors only: no impact; this warning \
          is decorative.\n\
        * If your fit uses any non-flat prior (model `~` syntax or \
          fit toml `[estimate.<param>.prior]`): the seeded chain inits \
          may sit in regions the prior excludes; expect long burn-in \
          or chains failing to mix.\n\
        \n\
        Workarounds until proper posterior-ranking ships:\n\
        * Switch to `init = lhs` (`--init lhs`) — unbiased stratified \
          coverage of the bound box, ignores the survey.\n\
        * Or verify post-hoc that your `chain_starts.tsv` inits sit \
          within the priors' high-density regions.\n\
        \n\
        Tracked: docs/dev/reviews/2026-05-26-week-audit-findings.md C1.");
}

fn emit_top_k_se_warning(top_k: &[&LandscapeRow]) {
    use crate::evidence::NATS_TO_DB;
    if top_k.len() < 2 { return; }
    let logliks: Vec<f64> = top_k.iter().map(|r| r.loglik).collect();
    let ses: Vec<f64> = top_k.iter().map(|r| r.loglik_se).collect();
    let hi = logliks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = logliks.iter().copied().fold(f64::INFINITY, f64::min);
    let delta_db = (hi - lo) * NATS_TO_DB;
    let sigma_max = ses.iter().copied().fold(0.0_f64, f64::max);
    // Mirror the IF2 convergence-gate floor: rank ordering is
    // meaningful only when the spread exceeds the SE-aware threshold.
    // 30 dB matches `GateConfig::default().decibans_thresh`.
    let threshold_db = (30.0_f64).max(8.0 * sigma_max * NATS_TO_DB);
    if delta_db < threshold_db {
        eprintln!("\x1b[33mwarning:\x1b[0m init = \"survey_top_k\": \
            top-{} loglik spread = {:.1} dB is below the SE-aware threshold \
            ({:.1} dB; σ_max = {:.2} nats). Rank ordering is uncertain at \
            this measurement budget — chains seeded from rank-1 vs rank-{} \
            may not be in genuinely-different basins. Consider re-running \
            the survey with higher --eval-replicates.",
            top_k.len(), delta_db, threshold_db, sigma_max, top_k.len());
    }
}

/// Format `chain_init_source` for `fit_state.toml` — one line of
/// provenance describing where this stage's chain starts came from.
/// `lhs` / `single` / `uniform` for the in-process samplers,
/// `survey:<full-hash>:top-<K>` for the survey reader.
pub fn format_chain_init_source(
    method: &InitMethod,
    survey_top_k: Option<&SurveyTopKResult>,
) -> String {
    if let Some(res) = survey_top_k {
        return format!("survey:{}:top-{}", res.survey_hash, res.chains.len());
    }
    match method {
        InitMethod::Single => "single".into(),
        InitMethod::Uniform => "uniform".into(),
        InitMethod::Lhs => "lhs".into(),
        InitMethod::UniformUnconstrained => "uniform_unconstrained".into(),
        InitMethod::SurveyTopK => {
            // SurveyTopKResult should have been provided. Defensive
            // fallback so a wiring bug doesn't write a corrupt
            // provenance string into fit_state.toml.
            "survey:<missing-result>:top-?".into()
        }
        // Step 6 warm-start variants render the bare kebab-case tag
        // here. Per-chain provenance lives in `InitSource` on each
        // `ChainStart`, which step-9 serialises into
        // `init_provenance.chains[i]` of `run.json`.
        InitMethod::FromPrior          => "from-prior".into(),
        InitMethod::FromPosterior { .. } => "from-posterior".into(),
        InitMethod::FromMle       { .. } => "from-mle".into(),
        InitMethod::FromParams    { .. } => "from-params".into(),
    }
}

/// Write `chain_starts.tsv` — sidecar audit-only artifact recording
/// the per-chain starting parameter vector and its provenance source
/// (e.g. `survey:<hash>:rank-1`, `lhs:chain-0`). Lives next to
/// `chain_evaluations.tsv` at the stage root. Emitted for every
/// init mode so an auditor can re-derive any chain's exact start
/// from a single TSV without reading inference-engine internals.
///
/// `per_chain_starts` is `None` for `InitMethod::Single` (every
/// chain at `base`) — that case writes one row per chain with the
/// base values and `source = "single"`. For ranked-survey mode the
/// caller supplies `survey_top_k` so each chain's source carries
/// `:rank-N` (1-indexed).
pub fn write_chain_starts_tsv(
    stage_dir: &std::path::Path,
    base: &[EstimatedParam],
    per_chain_starts: Option<&[Vec<EstimatedParam>]>,
    n_chains: usize,
    method: &InitMethod,
    survey_top_k: Option<&SurveyTopKResult>,
) -> std::io::Result<()> {
    use std::io::Write as _;
    let path = stage_dir.join("chain_starts.tsv");
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        // Comment header — stable, machine-parseable.
        writeln!(f, "# camdl chain_starts; method={}; chains={}",
            method, n_chains)?;
        if let Some(res) = survey_top_k {
            writeln!(f, "# survey_hash={}", res.survey_hash)?;
        }
        // Header row.
        let mut cols = vec!["chain_id".to_string(), "source".to_string()];
        for spec in base { cols.push(spec.name.clone()); }
        writeln!(f, "{}", cols.join("\t"))?;
        for chain_id in 0..n_chains {
            let source = match (method, survey_top_k) {
                (InitMethod::SurveyTopK, Some(res)) =>
                    format!("survey:{}:rank-{}", res.survey_hash, chain_id + 1),
                _ => format!("{}:chain-{}", method, chain_id),
            };
            let mut fields = vec![chain_id.to_string(), source];
            for (i, spec) in base.iter().enumerate() {
                let initial = per_chain_starts
                    .and_then(|chains| chains.get(chain_id))
                    .and_then(|c| c.get(i))
                    .map(|s| s.initial)
                    .unwrap_or(spec.initial);
                fields.push(format_float_for_tsv(initial));
            }
            writeln!(f, "{}", fields.join("\t"))?;
        }
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn format_float_for_tsv(v: f64) -> String {
    if v.is_nan() { "NaN".into() }
    else if v == f64::INFINITY  { "Inf".into() }
    else if v == f64::NEG_INFINITY { "-Inf".into() }
    else { format!("{}", v) }
}

/// Map an LHS coordinate `u ∈ [0, 1]` to the natural-scale parameter
/// value, respecting the parameter's transform.
fn lhs_map_to_natural(spec: &EstimatedParam, u: f64) -> f64 {
    if !spec.lower.is_finite() || !spec.upper.is_finite() {
        // Unbounded: ±50% jitter around the seeded start. LHS is meaningless
        // here but we don't want to fail — the upstream validator should
        // refuse fits with unbounded estimated params; until that lands,
        // fall back gracefully.
        return spec.initial * (0.5 + u);
    }
    match &spec.transform {
        Transform::Log { .. } if spec.lower > 0.0 && spec.upper > 0.0 => {
            // LHS in log space: θ = lo · (hi/lo)^u
            spec.lower * (spec.upper / spec.lower).powf(u)
        }
        _ => {
            // Linear LHS in [lo, hi]
            spec.lower + u * (spec.upper - spec.lower)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::types::Transform;

    /// gh#506 follow-up. `ignores_base_point` decides whether a declared
    /// `[estimate].start` has any effect, so it must agree exactly with what
    /// `build_chain_starts` does — a disagreement means either a note that
    /// contradicts the run, or silence where the start really was discarded.
    #[test]
    fn ignores_base_point_agrees_with_build_chain_starts() {
        // The three spreading modes fall back to the base point at one chain
        // (`build_chain_starts` returns None below n_chains = 2) and discard
        // it above.
        for m in [InitMethod::Uniform, InitMethod::Lhs,
                  InitMethod::UniformUnconstrained] {
            assert!(build_chain_starts(m.clone(), &[], 1, 1).is_none(),
                "{m} must degrade to the base point at one chain");
            assert!(!ignores_base_point(&m, 1),
                "{m} at one chain uses the base point");
        }
        // Above one chain, `uniform` keeps chain 1 at the seeded start; the
        // other two do not use it at all.
        assert!(!ignores_base_point(&InitMethod::Uniform, 4),
            "uniform's chain 1 keeps the seeded start");
        assert!(ignores_base_point(&InitMethod::Lhs, 4));
        assert!(ignores_base_point(&InitMethod::UniformUnconstrained, 4));

        // `single` is the mode whose entire contract is the base point.
        assert!(build_chain_starts(InitMethod::Single, &[], 8, 1).is_none());
        assert!(!ignores_base_point(&InitMethod::Single, 8));
        assert!(!ignores_base_point(&InitMethod::Single, 1));

        // The source-reading modes read every chain from elsewhere at any
        // chain count — they do NOT degrade to the base point at one chain.
        assert!(ignores_base_point(&InitMethod::FromPrior, 1));
        assert!(ignores_base_point(&InitMethod::SurveyTopK, 1));
        assert!(ignores_base_point(
            &InitMethod::FromParams { path: "p.toml".into() }, 1));
    }

    fn ep(name: &str, lower: f64, upper: f64, transform: Transform, initial: f64) -> EstimatedParam {
        EstimatedParam {
            name: name.into(),
            index: 0,
            initial,
            rw_sd: 0.1,
            transform,
            lower,
            upper,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }
    }

    #[test]
    fn init_method_default_is_uniform_unconstrained() {
        // Stan-style UniformUnconstrained by default for all multi-chain
        // stages — boundary-avoiding + scale-invariant; keeps the gh#42
        // log-scale awareness, adds Stan's robustness over Uniform/LHS.
        assert_eq!(InitMethod::default(), InitMethod::UniformUnconstrained);
    }

    #[test]
    fn init_method_from_str_round_trip() {
        for m in [
            InitMethod::Single,
            InitMethod::Uniform,
            InitMethod::Lhs,
            InitMethod::UniformUnconstrained,
            InitMethod::SurveyTopK,
            InitMethod::FromPrior,
        ] {
            let s = m.to_string();
            let parsed: InitMethod = s.parse().unwrap();
            assert_eq!(parsed, m,
                "Display ↔ FromStr round-trip must succeed for {:?}; \
                 got tag {:?}", m, s);
        }
        assert!("unknown".parse::<InitMethod>().is_err());
        // The TOML-on-the-wire form is the snake_case variant, not
        // hyphenated — survey_top_k, not survey-top-k.
        assert!("survey-top-k".parse::<InitMethod>().is_err());
    }

    /// gh#87: Display must emit **snake_case** for every variant, to
    /// match the CLI / serde / `clap::ValueEnum::to_possible_value`
    /// surfaces. Pre-fix, the payload-bearing variants rendered as
    /// kebab-case (`from-prior`, `from-posterior`, …) while the
    /// payload-free ones used snake_case (`survey_top_k`) —
    /// inconsistent JSON tags in `run.json`'s
    /// `init_provenance.method` field, and a downstream tool
    /// ingesting that field had to handle both spellings.
    #[test]
    fn init_method_display_is_snake_case_for_every_variant() {
        use std::path::PathBuf;
        let cases: Vec<(InitMethod, &str)> = vec![
            (InitMethod::Single,                              "single"),
            (InitMethod::Uniform,                             "uniform"),
            (InitMethod::Lhs,                                 "lhs"),
            (InitMethod::UniformUnconstrained,                "uniform_unconstrained"),
            (InitMethod::SurveyTopK,                          "survey_top_k"),
            (InitMethod::FromPrior,                           "from_prior"),
            (InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv(PathBuf::from("/x")),
            },                                                "from_posterior"),
            (InitMethod::FromMle {
                source: MleSource::File(PathBuf::from("/y")),
            },                                                "from_mle"),
            (InitMethod::FromParams {
                path: PathBuf::from("/z"),
            },                                                "from_params"),
        ];
        for (m, expected_tag) in cases {
            let got = m.to_string();
            assert_eq!(got, expected_tag,
                "InitMethod::{:?} Display must emit {:?} (snake_case); \
                 got {:?}. gh#87 pins this — downstream tools \
                 ingest run.json's init_provenance.method without \
                 having to disambiguate kebab-/snake-case forms.",
                m, expected_tag, got);
            // No `-` in any variant's Display tag.
            assert!(!got.contains('-'),
                "InitMethod::{:?} Display contains '-': {}", m, got);
        }
    }

    #[test]
    fn single_returns_none_so_caller_uses_base_params() {
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let out = build_chain_starts(InitMethod::Single, &base, 8, 42);
        assert!(out.is_none());
    }

    #[test]
    fn uniform_n1_returns_none() {
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        assert!(build_chain_starts(InitMethod::Uniform, &base, 1, 42).is_none());
        assert!(build_chain_starts(InitMethod::Lhs, &base, 1, 42).is_none());
    }

    #[test]
    fn lhs_strata_cover_range_uniformly() {
        // 100 chains × 1 param ∈ [0, 1] linear: every decile should
        // contain ~10 starts (LHS guarantee at this resolution).
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let starts = build_chain_starts(InitMethod::Lhs, &base, 100, 42).unwrap();
        let values: Vec<f64> = starts.iter().map(|c| c[0].initial).collect();

        let mut bin_counts = vec![0usize; 10];
        for &v in &values {
            let bin = ((v * 10.0) as usize).min(9);
            bin_counts[bin] += 1;
        }
        // LHS guarantees exactly one sample per stratum at the dim level.
        // With 100 chains and 10 bins, each stratum aligns 10:1 with bins.
        for &c in &bin_counts {
            assert!(c >= 8 && c <= 12,
                "LHS strata uneven: counts = {:?}", bin_counts);
        }
    }

    #[test]
    fn lhs_log_param_spans_orders_of_magnitude() {
        // Log-typed param with bounds [1e-5, 1e-2] should LHS in log space.
        // The geomean of all draws should be near sqrt(1e-5 * 1e-2) = 1e-3.5
        // and the spread should be the full range — not concentrated near 1e-2.
        let base = vec![ep("rate", 1e-5, 1e-2, Transform::Log { lo: 1e-5, hi: 1e-2 }, 1e-3)];
        let starts = build_chain_starts(InitMethod::Lhs, &base, 50, 42).unwrap();
        let values: Vec<f64> = starts.iter().map(|c| c[0].initial).collect();

        // Distribute roughly evenly across each decade.
        let log_vals: Vec<f64> = values.iter().map(|v| v.log10()).collect();
        let mean = log_vals.iter().sum::<f64>() / log_vals.len() as f64;
        // log10(1e-5) = -5, log10(1e-2) = -2, midpoint = -3.5
        assert!((mean - (-3.5)).abs() < 0.3,
            "log-LHS mean = {} (expected ~−3.5)", mean);

        let lo_count = values.iter().filter(|&&v| v < 1e-4).count();
        let hi_count = values.iter().filter(|&&v| v > 1e-3).count();
        // With LHS in log space, mass spreads across decades; uniform
        // (linear) sampling would cluster near 1e-2 with very few < 1e-4.
        assert!(lo_count >= 5 && hi_count >= 5,
            "log-LHS clusters: lo<1e-4={} hi>1e-3={} (linear sampling would skew here)",
            lo_count, hi_count);
    }

    #[test]
    fn lhs_deterministic_given_seed() {
        let base = vec![
            ep("a", 0.0, 1.0, Transform::None, 0.5),
            ep("b", 1e-3, 1.0, Transform::Log { lo: 1e-3, hi: 1.0 }, 0.1),
        ];
        let s1 = build_chain_starts(InitMethod::Lhs, &base, 16, 42).unwrap();
        let s2 = build_chain_starts(InitMethod::Lhs, &base, 16, 42).unwrap();
        for (c1, c2) in s1.iter().zip(s2.iter()) {
            for (p1, p2) in c1.iter().zip(c2.iter()) {
                assert_eq!(p1.initial, p2.initial);
            }
        }
    }

    #[test]
    fn uniform_unconstrained_n1_returns_none() {
        // Single chain ⇒ base spec — same `n_chains < 2` contract as
        // LHS/Uniform, so flipping the default only moves multi-chain fits.
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        assert!(build_chain_starts(InitMethod::UniformUnconstrained, &base, 1, 42).is_none());
    }

    #[test]
    fn uniform_unconstrained_is_boundary_avoiding() {
        // z ~ U(-2,2) ⇒ σ(z) ∈ (σ(-2), σ(2)) ≈ (0.1192, 0.8808), a fixed
        // interior band independent of the bounds. For a linear param on
        // [lo,hi] every start falls strictly inside the band — never on a
        // bound.
        let (lo, hi) = (0.0_f64, 1.0_f64);
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let band_lo = lo + (hi - lo) * sig(-STAN_INIT_RADIUS);
        let band_hi = lo + (hi - lo) * sig(STAN_INIT_RADIUS);
        let base = vec![ep("p", lo, hi, Transform::None, 0.5)];
        let starts = build_chain_starts(InitMethod::UniformUnconstrained, &base, 200, 7).unwrap();
        for c in &starts {
            let v = c[0].initial;
            assert!(v > band_lo - 1e-12 && v < band_hi + 1e-12,
                "start {v} escaped the σ(±2) interior band [{band_lo}, {band_hi}]");
            assert!(v > lo && v < hi, "start {v} on/outside bound [{lo}, {hi}]");
        }
    }

    #[test]
    fn uniform_unconstrained_log_param_stays_in_log_interior() {
        // Log param [1e-5, 1e-2]: θ = lo·(hi/lo)^σ(z). log10-interior band
        // is [-5 + 3·σ(-2), -5 + 3·σ(2)] = [-4.64, -2.36] — strictly inside
        // the declared decades.
        let base = vec![ep("rate", 1e-5, 1e-2, Transform::Log { lo: 1e-5, hi: 1e-2 }, 1e-3)];
        let starts = build_chain_starts(InitMethod::UniformUnconstrained, &base, 200, 7).unwrap();
        for c in &starts {
            let l = c[0].initial.log10();
            assert!(l > -4.65 && l < -2.35,
                "log10(start) = {l} escaped the σ(±2) log-interior [-4.64, -2.36]");
        }
    }

    #[test]
    fn uniform_unconstrained_deterministic_and_iid() {
        let base = vec![
            ep("a", 0.0, 1.0, Transform::None, 0.5),
            ep("b", 1e-3, 1.0, Transform::Log { lo: 1e-3, hi: 1.0 }, 0.1),
        ];
        let s1 = build_chain_starts(InitMethod::UniformUnconstrained, &base, 16, 42).unwrap();
        let s2 = build_chain_starts(InitMethod::UniformUnconstrained, &base, 16, 42).unwrap();
        for (c1, c2) in s1.iter().zip(s2.iter()) {
            for (p1, p2) in c1.iter().zip(c2.iter()) {
                assert_eq!(p1.initial, p2.initial);
            }
        }
        // i.i.d. across chains: chain 0 and chain 1 differ on both params.
        assert_ne!(s1[0][0].initial, s1[1][0].initial);
        assert_ne!(s1[0][1].initial, s1[1][1].initial);
    }

    #[test]
    fn lhs_different_seed_gives_different_draws() {
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let s1 = build_chain_starts(InitMethod::Lhs, &base, 16, 42).unwrap();
        let s2 = build_chain_starts(InitMethod::Lhs, &base, 16, 43).unwrap();
        let differs = s1.iter().zip(s2.iter())
            .any(|(c1, c2)| c1[0].initial != c2[0].initial);
        assert!(differs, "LHS with different seeds returned identical draws");
    }

    #[test]
    fn lhs_within_bounds() {
        let base = vec![
            ep("rate",  1e-5, 1.0, Transform::Log   { lo: 1e-5, hi: 1.0 }, 0.01),
            ep("prob",  0.05, 0.95, Transform::Logit { lo: 0.05, hi: 0.95 }, 0.5),
            ep("real", -10.0, 10.0, Transform::None,                          0.0),
        ];
        let starts = build_chain_starts(InitMethod::Lhs, &base, 32, 7).unwrap();
        for chain in &starts {
            for spec in chain {
                assert!(spec.initial >= spec.lower && spec.initial <= spec.upper,
                    "{} out of bounds: {} not in [{}, {}]",
                    spec.name, spec.initial, spec.lower, spec.upper);
            }
        }
    }

    // ── draw_start_in_bounds (gh#34 fallback) ────────────────────────

    #[test]
    fn draw_start_log_scale_lands_inside_positive_bounds() {
        // Log-scale draw across six orders of magnitude: result must
        // be strictly inside (lo, hi) and stay positive.
        let v = draw_start_in_bounds(1e-6, 1.0, true, 42, "beta");
        assert!(v > 1e-6 && v < 1.0, "{} not in (1e-6, 1.0)", v);
        assert!(v.is_finite() && v > 0.0);
    }

    #[test]
    fn draw_start_linear_scale_lands_inside_bounds() {
        // Linear draw on a real-valued parameter (Logit/None analogue):
        // negative-to-positive bounds, no log-scale possible.
        let v = draw_start_in_bounds(-10.0, 10.0, false, 42, "drift");
        assert!(v > -10.0 && v < 10.0, "{} not in (-10, 10)", v);
    }

    #[test]
    fn draw_start_log_falls_back_to_linear_when_lo_nonpositive() {
        // log_scale=true but lo=0 — helper must NOT call powf on zero
        // (would yield 0 always or NaN); falls back to linear.
        let v = draw_start_in_bounds(0.0, 1.0, true, 42, "p");
        assert!(v > 0.0 && v < 1.0, "{} not in (0, 1)", v);
    }

    #[test]
    fn draw_start_deterministic_per_seed_and_name() {
        // Same (seed, name) ⇒ same draw.
        let a = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        assert_eq!(a, b);
    }

    #[test]
    fn draw_start_different_names_give_different_draws() {
        // Two parameters with identical bounds at the same seed must
        // not collide (would defeat the point of the per-name hash).
        let a = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 7, "gamma");
        assert_ne!(a, b);
    }

    #[test]
    fn draw_start_different_seeds_give_different_draws() {
        // Reseeding the run shifts the fallback (so users get spread
        // across seed sweeps, unlike the old midpoint heuristic which
        // gave the same point at every seed).
        let a = draw_start_in_bounds(1e-3, 1.0, true, 1, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 2, "beta");
        assert_ne!(a, b);
    }

    #[test]
    fn draw_start_log_scale_spans_orders_of_magnitude() {
        // Across many seeds, log-scale draws on (1e-6, 1.0) should
        // populate at least three different decade buckets — the prior
        // midpoint would have given 1e-3 at every seed.
        use std::collections::HashSet;
        let mut decades: HashSet<i32> = HashSet::new();
        for seed in 0..64u64 {
            let v = draw_start_in_bounds(1e-6, 1.0, true, seed, "beta");
            decades.insert(v.log10().floor() as i32);
        }
        assert!(decades.len() >= 3,
            "expected ≥3 decades populated across 64 seeds, got {}: {:?}",
            decades.len(), decades);
    }

    // ── parse_landscape_tsv (gh#51) ──────────────────────────────────

    #[test]
    fn parse_landscape_tsv_pfilter_columns() {
        // Real shape: comments, header, 3 data rows. Param order in
        // header matches survey_estimated. Pfilter eval includes
        // mean_ess column (we ignore it but the parser must tolerate
        // the wider row).
        let raw = "\
# camdl survey landscape; run_hash=abc; version=0.1\n\
# eval=pfilter; n_points=3\n\
beta\tgamma\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id\n\
0.3\t0.1\t-100.5\t1.2\t0.8\t8\t0\n\
0.4\t0.2\t-95.0\t0.9\t0.85\t8\t1\n\
0.5\t0.15\t-110.2\t2.0\t0.75\t8\t2\n";
        let estimated = vec!["beta".to_string(), "gamma".to_string()];
        let rows = parse_landscape_tsv(raw, &estimated).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].params.get("beta"), Some(&0.3));
        assert_eq!(rows[0].params.get("gamma"), Some(&0.1));
        assert_eq!(rows[0].loglik, -100.5);
        assert_eq!(rows[0].loglik_se, 1.2);
        // Best-loglik row is index 1 (loglik = -95.0).
        let best = rows.iter().max_by(|a, b|
            a.loglik.partial_cmp(&b.loglik).unwrap()).unwrap();
        assert_eq!(best.loglik, -95.0);
    }

    #[test]
    fn parse_landscape_tsv_simulate_columns() {
        // Simulate eval omits mean_ess column.
        let raw = "\
# survey\n\
beta\tgamma\tloglik\tloglik_se\tn_replicates\tpoint_id\n\
0.3\t0.1\t-100.5\t1.2\t1\t0\n";
        let estimated = vec!["beta".to_string(), "gamma".to_string()];
        let rows = parse_landscape_tsv(raw, &estimated).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].params.get("beta"), Some(&0.3));
        assert_eq!(rows[0].loglik, -100.5);
    }

    #[test]
    fn parse_landscape_tsv_missing_param_column_errors() {
        // Survey claims `beta` is estimated but the header doesn't
        // have a `beta` column. Should error with a clear message
        // naming the missing param.
        let raw = "\
gamma\tloglik\tloglik_se\tn_replicates\tpoint_id\n\
0.1\t-100.5\t1.2\t1\t0\n";
        let estimated = vec!["beta".to_string(), "gamma".to_string()];
        let err = parse_landscape_tsv(raw, &estimated).unwrap_err();
        assert!(err.contains("beta"), "error should name missing param: {}", err);
    }

    #[test]
    fn parse_landscape_tsv_missing_loglik_errors() {
        let raw = "\
beta\tgamma\tn_replicates\tpoint_id\n\
0.3\t0.1\t1\t0\n";
        let estimated = vec!["beta".to_string(), "gamma".to_string()];
        let err = parse_landscape_tsv(raw, &estimated).unwrap_err();
        assert!(err.contains("loglik"));
    }

    // ── cross_check_survey (gh#51) ───────────────────────────────────

    fn make_survey_meta(
        model_identity: &str,
        data_hashes: &[(&str, &str)],
        fixed: &[(&str, f64)],
        estimated: &[&str],
    ) -> SurveyCrossCheck {
        SurveyCrossCheck {
            model_identity: model_identity.into(),
            data_hashes: data_hashes.iter()
                .map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            fixed: fixed.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            estimated: estimated.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Write a new-format (`runid::RunRecord`) survey `run.json` to `dir`
    /// carrying the cross-check provenance the consumer reads back.
    fn write_survey_record(
        dir: &std::path::Path,
        run_id_hex: &str,
        model_identity: &str,
        data_hashes: &[(&str, &str)],
        fixed: &[(&str, f64)],
        estimated: &[&str],
    ) {
        let dh: std::collections::HashMap<String, String> = data_hashes.iter()
            .map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let fx: std::collections::HashMap<String, f64> = fixed.iter()
            .map(|(k, v)| (k.to_string(), *v)).collect();
        let est: Vec<String> = estimated.iter().map(|s| s.to_string()).collect();
        let record = runid::RunRecord {
            format_version: runid::FORMAT_VERSION,
            kind: runid::ArtifactKind::Survey,
            run_id: runid::ContentHash::from_hex(run_id_hex).unwrap(),
            hash_version: runid::HASH_VERSION,
            ir_version: "0.7".into(),
            engine_version: "test".into(),
            levels: Vec::new(),
            deps: Vec::new(),
            status: runid::RunStatus::Completed,
            artifacts: Default::default(),
            output_schema: Default::default(),
            children: Default::default(),
            inputs: serde_json::json!({
                "model_identity": model_identity,
                "data_hashes": dh,
                "fixed": fx,
                "estimated": est,
            }),
            provenance: Default::default(),
        };
        std::fs::write(dir.join("run.json"),
            serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    #[test]
    fn cross_check_refuses_model_identity_mismatch() {
        let meta = make_survey_meta("aaa", &[("cases", "h1")], &[], &["beta"]);
        let dh: std::collections::HashMap<String, String> =
            [("cases".to_string(), "h1".to_string())].into_iter().collect();
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "bbb",  // ← mismatch
            data_hashes: &dh,
            fixed: &fx,
            estimate_names: &names,
        };
        let err = cross_check_survey(&meta, &ctx).unwrap_err();
        assert!(err.contains("model_identity"));
        assert!(err.contains("aaa") && err.contains("bbb"),
            "diagnostic should print both hashes: {}", err);
    }

    #[test]
    fn cross_check_refuses_data_hash_mismatch_on_fit_stream() {
        let meta = make_survey_meta("aaa", &[("cases", "h1")], &[], &["beta"]);
        let dh: std::collections::HashMap<String, String> =
            [("cases".to_string(), "h2".to_string())].into_iter().collect();  // ← differs
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "aaa", data_hashes: &dh, fixed: &fx, estimate_names: &names,
        };
        let err = cross_check_survey(&meta, &ctx).unwrap_err();
        assert!(err.contains("data_hashes"));
        assert!(err.contains("cases"));
    }

    #[test]
    fn cross_check_refuses_fit_stream_absent_from_survey() {
        // Fit consumes a stream the survey didn't score.
        let meta = make_survey_meta("aaa", &[("cases", "h1")], &[], &["beta"]);
        let dh: std::collections::HashMap<String, String> =
            [("deaths".to_string(), "h3".to_string())].into_iter().collect();
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "aaa", data_hashes: &dh, fixed: &fx, estimate_names: &names,
        };
        let err = cross_check_survey(&meta, &ctx).unwrap_err();
        assert!(err.contains("deaths"));
    }

    #[test]
    fn cross_check_refuses_fixed_disagreement() {
        let meta = make_survey_meta("aaa", &[], &[("rho", 0.5)], &["beta"]);
        let dh: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let fx: std::collections::HashMap<String, f64> =
            [("rho".to_string(), 0.7)].into_iter().collect();   // ← differs
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "aaa", data_hashes: &dh, fixed: &fx, estimate_names: &names,
        };
        let err = cross_check_survey(&meta, &ctx).unwrap_err();
        assert!(err.contains("rho"));
        assert!(err.contains("[fixed]"));
    }

    #[test]
    fn cross_check_refuses_fit_fixed_absent_from_survey() {
        // Fit pins `rho`; survey didn't (estimated or free).
        let meta = make_survey_meta("aaa", &[], &[], &["beta", "rho"]);
        let dh: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let fx: std::collections::HashMap<String, f64> =
            [("rho".to_string(), 0.5)].into_iter().collect();
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "aaa", data_hashes: &dh, fixed: &fx, estimate_names: &names,
        };
        let err = cross_check_survey(&meta, &ctx).unwrap_err();
        assert!(err.contains("rho"));
        assert!(err.contains("subset"));
    }

    #[test]
    fn cross_check_passes_when_survey_pins_more_than_fit() {
        // Survey pinned `extra_param` at 1.0; fit doesn't fix or
        // estimate it. The "[fixed] superset" rule says survey ⊇ fit,
        // so survey-extras are fine.
        let meta = make_survey_meta(
            "aaa", &[("cases", "h1")], &[("rho", 0.5), ("extra_param", 1.0)],
            &["beta"],
        );
        let dh: std::collections::HashMap<String, String> =
            [("cases".to_string(), "h1".to_string())].into_iter().collect();
        let fx: std::collections::HashMap<String, f64> =
            [("rho".to_string(), 0.5)].into_iter().collect();
        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "aaa", data_hashes: &dh, fixed: &fx, estimate_names: &names,
        };
        cross_check_survey(&meta, &ctx).expect("survey-pins-more should pass");
    }

    // ── format_chain_init_source (gh#51) ────────────────────────────

    #[test]
    fn chain_init_source_format_for_each_method() {
        assert_eq!(format_chain_init_source(&InitMethod::Single, None), "single");
        assert_eq!(format_chain_init_source(&InitMethod::Uniform, None), "uniform");
        assert_eq!(format_chain_init_source(&InitMethod::Lhs, None), "lhs");

        let result = SurveyTopKResult {
            chains: vec![Vec::new(); 20],   // 20 chains
            survey_hash: "deadbeefcafe1234deadbeefcafe1234deadbeefcafe1234deadbeefcafe1234".into(),
        };
        let s = format_chain_init_source(&InitMethod::SurveyTopK, Some(&result));
        // Full hash, not short — the audit-survivability invariant.
        assert!(s.starts_with("survey:deadbeefcafe1234deadbeefcafe1234"),
            "should embed full hash: {}", s);
        assert!(s.ends_with(":top-20"),
            "should include K from chain count: {}", s);
    }

    #[test]
    fn chain_init_source_falls_back_when_survey_result_missing() {
        // Defensive: should never happen in practice, but a wiring
        // bug must not write a corrupt provenance string.
        let s = format_chain_init_source(&InitMethod::SurveyTopK, None);
        assert!(s.contains("missing"),
            "fallback string should flag the wiring bug: {}", s);
    }

    // ── build_chain_starts_from_survey end-to-end (gh#51) ────────────

    /// Write a minimal but real survey CAS dir to disk, point
    /// `build_chain_starts_from_survey` at it, and verify the round-
    /// trip: top-K rows by loglik come back as per-chain
    /// EstimatedParam vectors with the right `initial` values pulled
    /// from the landscape TSV.
    #[test]
    fn build_chain_starts_from_survey_end_to_end_happy_path() {
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "camdl_survey_top_k_e2e_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();

        let model_identity = "model-hash-aaa";
        let mut data_hashes = HashMap::new();
        data_hashes.insert("cases".to_string(), "data-hash-bbb".to_string());

        let run_id_hex = "cafef00d".repeat(8);   // 64-hex
        write_survey_record(
            &dir, &run_id_hex, model_identity,
            &[("cases", "data-hash-bbb")], &[], &["beta", "gamma"]);

        // Landscape: 5 rows, loglik increasing from -100 to -90 in the
        // sorted-desc order rank-1 = -90 (best). Within bounds
        // beta∈[0.1, 1.0], gamma∈[0.05, 0.5].
        let landscape = "\
# camdl survey landscape\n\
# eval=pfilter\n\
beta\tgamma\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id\n\
0.30\t0.10\t-100.0\t1.0\t0.8\t4\t0\n\
0.40\t0.15\t-95.0\t1.0\t0.8\t4\t1\n\
0.50\t0.20\t-90.0\t1.0\t0.8\t4\t2\n\
0.60\t0.25\t-99.0\t1.0\t0.8\t4\t3\n\
0.70\t0.30\t-98.0\t1.0\t0.8\t4\t4\n";
        std::fs::write(dir.join("landscape.tsv"), landscape).unwrap();

        let names = vec!["beta".to_string(), "gamma".to_string()];
        let ctx = SurveyFitContext {
            model_identity, data_hashes: &data_hashes,
            fixed: &HashMap::new(),
            estimate_names: &names,
        };
        let base = vec![
            ep("beta",  0.1, 1.0,  Transform::Log { lo: 0.1, hi: 1.0 }, 0.5),
            ep("gamma", 0.05, 0.5, Transform::Log { lo: 0.05, hi: 0.5 }, 0.2),
        ];

        let result = build_chain_starts_from_survey(&dir, Some(3), 3, &base, &ctx)
            .expect("happy path should succeed");
        assert_eq!(result.chains.len(), 3);
        assert_eq!(result.survey_hash, run_id_hex);

        // rank-1 by loglik (best = -90) is row 2: beta=0.50, gamma=0.20.
        assert!((result.chains[0][0].initial - 0.50).abs() < 1e-9);
        assert!((result.chains[0][1].initial - 0.20).abs() < 1e-9);
        // rank-2 (loglik=-95) is row 1: beta=0.40, gamma=0.15.
        assert!((result.chains[1][0].initial - 0.40).abs() < 1e-9);
        assert!((result.chains[1][1].initial - 0.15).abs() < 1e-9);
        // rank-3 (loglik=-98) is row 4: beta=0.70, gamma=0.30.
        assert!((result.chains[2][0].initial - 0.70).abs() < 1e-9);
        assert!((result.chains[2][1].initial - 0.30).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// When fit's bounds exclude rows that the survey scored, those
    /// rows are filtered (no clipping). Refusal fires when the
    /// filtered count drops below `n_chains`.
    #[test]
    fn build_chain_starts_from_survey_filter_then_rank_refuses() {
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "camdl_survey_top_k_filter_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();

        write_survey_record(
            &dir, &"aa".repeat(32), "h", &[], &[], &["beta"]);

        // Survey ranged over beta∈[0.0, 1.0]. Fit narrows to [0.6, 0.8].
        // Only one row (beta=0.7) falls within fit's bounds.
        let landscape = "\
beta\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id\n\
0.10\t-100.0\t1.0\t0.8\t1\t0\n\
0.30\t-95.0\t1.0\t0.8\t1\t1\n\
0.50\t-90.0\t1.0\t0.8\t1\t2\n\
0.70\t-92.0\t1.0\t0.8\t1\t3\n\
0.90\t-99.0\t1.0\t0.8\t1\t4\n";
        std::fs::write(dir.join("landscape.tsv"), landscape).unwrap();

        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "h", data_hashes: &HashMap::new(),
            fixed: &HashMap::new(), estimate_names: &names,
        };
        // Tight fit bounds: only row beta=0.70 survives.
        let base = vec![ep("beta", 0.6, 0.8, Transform::None, 0.7)];

        // Asking for 3 chains when only 1 row survives → refuse.
        let err = build_chain_starts_from_survey(&dir, Some(3), 3, &base, &ctx)
            .unwrap_err();
        assert!(err.contains("only 1") || err.contains("survey has"),
            "diagnostic should name the filtered count: {}", err);

        // Asking for 1 chain (matches filtered count) → succeeds with
        // the surviving row.
        let res = build_chain_starts_from_survey(&dir, Some(1), 1, &base, &ctx)
            .expect("single-chain happy path");
        assert_eq!(res.chains.len(), 1);
        assert!((res.chains[0][0].initial - 0.70).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_chain_starts_from_survey_v1_strict_k_equals_chains() {
        // top_k_n must equal n_chains in v1; K > chains is deferred.
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "camdl_survey_top_k_strict_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        write_survey_record(
            &dir, &"bb".repeat(32), "h", &[], &[], &["beta"]);
        std::fs::write(dir.join("landscape.tsv"),
            "beta\tloglik\tloglik_se\tn_replicates\tpoint_id\n\
             0.5\t-100.0\t1.0\t1\t0\n").unwrap();

        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "h", data_hashes: &HashMap::new(),
            fixed: &HashMap::new(), estimate_names: &names,
        };
        let base = vec![ep("beta", 0.0, 1.0, Transform::None, 0.5)];

        // top_k=2, n_chains=1 → refuse with v1-scope diagnostic.
        let err = build_chain_starts_from_survey(&dir, Some(2), 1, &base, &ctx)
            .unwrap_err();
        assert!(err.contains("v1") && err.contains("top_k"),
            "v1 strict-K diagnostic should mention v1: {}", err);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── resolve_per_chain_starts_from_method (gh#51 v2) ──────────────

    #[test]
    fn resolve_per_chain_starts_lhs_path_no_survey() {
        // Non-survey methods route through build_chain_starts and
        // return (per_chain, None).
        let base = vec![
            ep("a", 0.0, 1.0, Transform::None, 0.5),
            ep("b", 1e-3, 1.0, Transform::Log { lo: 1e-3, hi: 1.0 }, 0.1),
        ];
        let dh: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names: Vec<String> = vec![];
        let ctx = SurveyFitContext {
            model_identity: "h", data_hashes: &dh, fixed: &fx,
            estimate_names: &names,
        };
        let (per_chain, survey) = resolve_per_chain_starts_from_method(
            &InitMethod::Lhs, None, None, "scout",
            &base, 8, 42, &ctx, None,
        ).expect("non-survey LHS path must succeed");
        assert!(per_chain.is_some(), "Lhs with n_chains=8 should produce chains");
        assert_eq!(per_chain.as_ref().unwrap().len(), 8);
        assert!(survey.is_none(), "non-survey method must produce no SurveyTopKResult");
    }

    #[test]
    fn resolve_per_chain_starts_single_returns_none() {
        // InitMethod::Single → (None, None): caller uses base_specs.
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let dh: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names: Vec<String> = vec![];
        let ctx = SurveyFitContext {
            model_identity: "h", data_hashes: &dh, fixed: &fx,
            estimate_names: &names,
        };
        let (per_chain, survey) = resolve_per_chain_starts_from_method(
            &InitMethod::Single, None, None, "refine",
            &base, 4, 42, &ctx, None,
        ).unwrap();
        assert!(per_chain.is_none(), "Single → None (caller uses base directly)");
        assert!(survey.is_none());
    }

    #[test]
    fn resolve_per_chain_starts_survey_top_k_without_path_errors() {
        // SurveyTopK with survey_path = None → error naming the stage.
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let dh: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let fx: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let names = vec!["a".to_string()];
        let ctx = SurveyFitContext {
            model_identity: "h", data_hashes: &dh, fixed: &fx,
            estimate_names: &names,
        };
        let err = resolve_per_chain_starts_from_method(
            &InitMethod::SurveyTopK, None, None, "scout",
            &base, 4, 42, &ctx, None,
        ).unwrap_err();
        assert!(err.contains("survey_path"),
            "diagnostic should name survey_path: {}", err);
        assert!(err.contains("scout"),
            "diagnostic should name the offending stage: {}", err);
        assert!(err.contains("gh#51"),
            "diagnostic should reference gh#51: {}", err);
    }

    #[test]
    fn resolve_per_chain_starts_survey_top_k_end_to_end() {
        // Happy path: SurveyTopK with a valid survey dir →
        // returns (Some(chains), Some(result)). Same fixture shape
        // as the build_chain_starts_from_survey happy-path test.
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "camdl_resolve_e2e_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();

        let model_identity = "model-hash-aaa";
        let mut data_hashes = HashMap::new();
        data_hashes.insert("cases".to_string(), "data-hash-bbb".to_string());

        let run_id_hex =
            "deadbeefcafe1234deadbeefcafe1234deadbeefcafe1234deadbeefcafe1234";
        write_survey_record(
            &dir, run_id_hex, model_identity,
            &[("cases", "data-hash-bbb")], &[], &["beta"]);
        std::fs::write(dir.join("landscape.tsv"),
            "beta\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id\n\
             0.3\t-100.0\t1.0\t0.8\t1\t0\n\
             0.5\t-90.0\t1.0\t0.8\t1\t1\n").unwrap();

        let names = vec!["beta".to_string()];
        let ctx = SurveyFitContext {
            model_identity, data_hashes: &data_hashes,
            fixed: &HashMap::new(), estimate_names: &names,
        };
        let base = vec![ep("beta", 0.1, 1.0, Transform::Log { lo: 0.1, hi: 1.0 }, 0.5)];

        let (per_chain, survey) = resolve_per_chain_starts_from_method(
            &InitMethod::SurveyTopK, Some(&dir), Some(2), "scout",
            &base, 2, 42, &ctx, None,
        ).expect("SurveyTopK happy path must succeed");
        let chains = per_chain.expect("SurveyTopK must produce chains");
        let result = survey.expect("SurveyTopK must produce a SurveyTopKResult");
        // rank-1 by loglik is beta=0.5 (loglik=-90); rank-2 is beta=0.3 (-100).
        assert_eq!(chains.len(), 2);
        assert!((chains[0][0].initial - 0.5).abs() < 1e-9);
        assert!((chains[1][0].initial - 0.3).abs() < 1e-9);
        assert_eq!(result.survey_hash, run_id_hex);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chain_starts_to_param_vecs_overwrites_estimated_indices() {
        // Verify the spec→f64 conversion that PMMH/PGAS will perform on
        // resolve_per_chain_starts_from_method's output.
        let base_specs = vec![
            ep_with_idx("beta",  0, 0.0, 1.0, Transform::None, 0.5),
            ep_with_idx("gamma", 2, 0.0, 1.0, Transform::None, 0.3),
        ];
        // Two chains, two estimated indices (0 and 2 of a 4-slot vector).
        let chains = vec![
            vec![
                EstimatedParam { initial: 0.1, ..base_specs[0].clone() },
                EstimatedParam { initial: 0.2, ..base_specs[1].clone() },
            ],
            vec![
                EstimatedParam { initial: 0.7, ..base_specs[0].clone() },
                EstimatedParam { initial: 0.8, ..base_specs[1].clone() },
            ],
        ];
        let base_params = vec![999.0, 11.0, 999.0, 22.0];
        let out = chain_starts_to_param_vecs(&chains, &base_params);
        assert_eq!(out.len(), 2);
        // Chain 0: positions 0/2 overwritten, 1/3 untouched.
        assert_eq!(out[0], vec![0.1, 11.0, 0.2, 22.0]);
        assert_eq!(out[1], vec![0.7, 11.0, 0.8, 22.0]);
    }

    fn ep_with_idx(
        name: &str, index: usize, lower: f64, upper: f64,
        transform: Transform, initial: f64,
    ) -> EstimatedParam {
        EstimatedParam {
            name: name.into(),
            index,
            initial,
            rw_sd: 0.1,
            transform,
            lower,
            upper,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }
    }
}
