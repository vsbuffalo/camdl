//! The one seam every runner draws its chain starts through.
//!
//! [`draw_chain_starts`] dispatches on a [`ChainStarts`] rule: the bare
//! spread rules go to the transform-aware builders in [`crate::fit::init`],
//! the sourced rules read their file, and every rule returns the same
//! [`DrawnStarts`] shape — one [`ChainStart`] per chain, each carrying an
//! [`InitSource`] tag saying which draw produced it. A sourced rule's handle
//! is resolved once, before the run's identity is taken, by
//! [`resolve_starts`], which also folds the source's *content* into the
//! run's lineage so rewriting a file in place re-keys the run (gh#541).
//!
//! The seam between parameter resolution and chain initialization is the
//! [`ChainStart::values`] map: it only contains parameters in
//! `resolved.estimate_set`. Every loader builds the map by iterating that
//! set — there is no way to ask "what's the starting value for `gamma`?"
//! when `gamma` is fixed. This is what guarantees a fixed value always wins
//! over the starts rule.
//!
//! The starts themselves are recorded by [`write_chain_starts_tsv`], the one
//! writer of `chain_starts.tsv`; every multi-chain sampler calls it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sim::inference::prior::{Density, Prior};
use sim::inference::types::EstimatedParam;
use sim::rng::StatefulRng;

use crate::params_resolver::ResolvedParameters;
use crate::util::derive_chain_seed;

use super::starts::{ChainStarts, Point, Spread};

/// Per-chain provenance tag. Stored on each [`ChainStart`] and rendered
/// into `run.json`'s `init_provenance.chains[i][param].source` by the
/// profile runner.
///
/// Each variant carries enough information to identify the specific
/// draw: the seed for stochastic samplers, the row index + path for
/// file-based draws.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InitSource {
    /// All chains at the seeded base param values (`starts = "single"`, or
    /// a spread rule degraded to the base point at one chain).
    SeededBase,
    /// Per-chain uniform random draw within parameter bounds
    /// (`starts = "uniform"`).
    UniformDraw { seed: u64 },
    /// Per-chain Stan-style draw: i.i.d. `Uniform(-2, 2)` on the
    /// unconstrained scale, squashed and mapped into bounds
    /// (`starts = "uniform_unconstrained"`).
    UnconstrainedDraw { seed: u64 },
    /// One stratum of an LHS layout.
    LhsCell { row: usize },
    /// Per-chain draw from a parameter's prior (or bounds-uniform fall-back
    /// when it has none).
    PriorDraw { seed: u64 },
    /// One row of a posterior draws TSV.
    PosteriorRow { row: usize, path: PathBuf },
    /// All chains at the point estimate a stored fit's `fit_state.toml`
    /// records.
    MlePoint { path: PathBuf },
    /// All chains at a hand-written flat params TOML.
    ParamsPoint { path: PathBuf },
}

impl InitSource {
    /// Stable string tag for one-line / column-oriented provenance.
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

/// How many starts a chain is drawn before it is refused (gh#887). A spread
/// rule is a lottery over one draw; ten independent draws make the refusal
/// a statement about the rule and the data rather than about luck. A point
/// rule has nothing to redraw and gets one.
pub const MAX_START_ATTEMPTS: usize = 10;

/// What to say when no chain could be started — the reason and the remedies,
/// written once (gh#880, gh#885).
///
/// Three aggregate errors describe this one failure: PGAS's "all N chains were
/// refused at their starting point", PMMH's "all N chains failed init-eval
/// with PFDegenerate", and IF2's "all N chains bailed via the PF degeneracy
/// watchdog". Each used to carry its own advice, and two of them carried none
/// of the reason; one of those two recommended tightening bounds, which is the
/// half of the story that reads as if widening them would help.
///
/// The measurement behind the sentence is gh#876: a start whose projected mean
/// sits several standard deviations from the observed counts leaves no
/// particle with appreciable weight, and that standardised distance grows with
/// the square root of the population — a relative error that is harmless at
/// ten thousand people is fatal at a million. The remedies follow from it:
/// start somewhere the filter can score, or give the filter more particles.
/// Widening the bounds is named as the thing that does NOT help, because the
/// drawing rules map their draws through the bounds, so a wider range makes an
/// unscoreable start more likely, not less.
pub const UNSCOREABLE_START_ADVICE: &str =
    "A start whose projected mean sits several standard deviations from the \
     observed counts leaves no particle with appreciable weight, and that \
     standardised distance grows with the square root of the population — so a \
     relative error that is harmless at ten thousand people is fatal at a \
     million. Start every chain at the declared values (`starts = \"single\"`), \
     draw the starts from the priors (`starts = \"from_prior\"`), or raise \
     `particles`. Note that widening the parameter bounds widens the range the \
     starts are drawn from, so it makes this refusal more likely, not less.";

/// A start the filter could not score, kept so `chain_starts.tsv` says how
/// many starts were tried and why each was dropped (gh#887).
#[derive(Debug, Clone)]
pub struct RejectedStart {
    pub chain_id: usize,
    /// 0-based attempt index; the accepted start's attempt is one past the
    /// last rejection.
    pub attempt: usize,
    pub values: HashMap<String, f64>,
    /// Why the filter (or the sampler's first sweep) refused it.
    pub reason: String,
    /// The ESS the filter reached before refusing, when it measured one.
    pub ess: Option<f64>,
}

/// The full set of chain starts produced by [`draw_chain_starts`], plus the
/// attempts a retry rejected on the way to them.
#[derive(Debug, Clone)]
pub struct DrawnStarts {
    /// Length = `n_chains` requested: the start each chain ran from (or was
    /// refused at, when every attempt failed).
    pub starts: Vec<ChainStart>,
    /// The rule that produced these starts.
    pub rule: ChainStarts,
    /// Every start a retry rejected, in (chain, attempt) order.
    pub rejected: Vec<RejectedStart>,
    /// Chains whose last attempt was refused too — they did not run.
    pub refused: Vec<usize>,
}

/// Does a rule admit a fresh draw for one chain? A point rule does not (the
/// point is the point); the three bounds-based spread rules fall back to the
/// base point at one chain, so there is nothing new to draw there either.
pub fn can_redraw(rule: &ChainStarts, n_chains: usize) -> bool {
    match rule {
        ChainStarts::Point(_) => false,
        ChainStarts::Spread(Spread::Uniform)
        | ChainStarts::Spread(Spread::Lhs)
        | ChainStarts::Spread(Spread::UniformUnconstrained) => n_chains >= 2,
        ChainStarts::Spread(Spread::FromPrior) | ChainStarts::Spread(Spread::FromPosterior { .. }) => true,
    }
}

/// A fresh, independent start for `chain_id` under the same rule: attempt
/// `attempt` (1-based; 0 is the original draw) of the bounded retry. The
/// rule is re-run at a seed derived from `(seed, attempt)` and this chain's
/// slot is taken, so a redraw is a draw the rule could have made in the
/// first place — an LHS redraw is a cell of a fresh design, a prior redraw
/// is another prior draw. Under `uniform` chain 0 is the base point by
/// definition, so its redraw takes slot 1's draw.
pub fn redraw_chain_start(
    ctx: &StartContext<'_>,
    starts: &ResolvedStarts,
    n_chains: usize,
    chain_id: usize,
    seed: u64,
    attempt: usize,
) -> Result<ChainStart, InitError> {
    let seed_k = seed ^ (attempt as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    let fresh = draw_chain_starts(ctx, starts, n_chains, seed_k)?;
    let slot = match starts.rule {
        ChainStarts::Spread(Spread::Uniform) if chain_id == 0 => 1,
        _ => chain_id,
    };
    let mut cs = fresh.starts.into_iter().nth(slot).ok_or(InitError::Unresolved {
        rule: starts.rule.spelled(),
    })?;
    cs.chain_id = chain_id;
    Ok(cs)
}

impl ChainStart {
    /// This chain's full parameter vector: `base_params` with each estimated
    /// slot overwritten by the start.
    pub fn to_param_vec(&self, base_specs: &[EstimatedParam], base_params: &[f64]) -> Vec<f64> {
        let mut params = base_params.to_vec();
        for spec in base_specs {
            if let Some(v) = self.values.get(&spec.name) {
                params[spec.index] = *v;
            }
        }
        params
    }
}

impl DrawnStarts {
    /// The starts as first drawn, with no retry yet.
    pub fn fresh(starts: Vec<ChainStart>, rule: ChainStarts) -> Self {
        DrawnStarts { starts, rule, rejected: Vec::new(), refused: Vec::new() }
    }

    /// Fold a retry's outcome in: `accepted[c]` replaces chain `c`'s start
    /// when a redraw was taken, `rejected` lists every attempt dropped on the
    /// way, and `refused` names the chains whose last attempt failed too.
    pub fn with_retry_outcome(
        mut self,
        accepted: Vec<Option<ChainStart>>,
        mut rejected: Vec<RejectedStart>,
        refused: Vec<usize>,
    ) -> Self {
        for (chain_id, cs) in accepted.into_iter().enumerate() {
            if let Some(cs) = cs {
                if let Some(slot) = self.starts.get_mut(chain_id) {
                    *slot = cs;
                }
            }
        }
        rejected.sort_by_key(|r| (r.chain_id, r.attempt));
        self.rejected = rejected;
        self.refused = refused;
        self
    }

    /// How many attempts chain `chain_id` took before its start was accepted
    /// (or refused): the rejected count for that chain.
    pub fn attempts_before(&self, chain_id: usize) -> usize {
        self.rejected.iter().filter(|r| r.chain_id == chain_id).count()
    }
    /// Adapt to the IF2-shaped `Vec<Vec<EstimatedParam>>` view that
    /// `runner::run_chains_with_per_chain_params` and the PMMH / PGAS /
    /// NUTS / NLopt dispatch sites consume. Each chain's `EstimatedParam`s
    /// start from `base_specs` and have `.initial` overwritten from
    /// `ChainStart.values` for any name present in the map.
    ///
    /// Names in `estimate_set` that the map doesn't carry (e.g. when a
    /// loader fell back to bounds-uniform for a missing column) take
    /// `base_specs[i].initial`; the loader already emitted a startup
    /// warning so the silent fall-through is auditable.
    pub fn to_estimated_params(&self, base_specs: &[EstimatedParam]) -> Vec<Vec<EstimatedParam>> {
        self.starts.iter().map(|cs| {
            base_specs.iter().map(|spec| {
                let initial = cs.values.get(&spec.name)
                    .copied()
                    .unwrap_or(spec.initial);
                EstimatedParam { initial, ..spec.clone() }
            }).collect()
        }).collect()
    }

    /// Per-chain full parameter vectors: `base_params` with each estimated
    /// slot overwritten by that chain's start.
    pub fn to_param_vecs(&self, base_specs: &[EstimatedParam], base_params: &[f64]) -> Vec<Vec<f64>> {
        super::init::chain_starts_to_param_vecs(&self.to_estimated_params(base_specs), base_params)
    }
}

/// Errors specific to chain-start drawing. Returned by
/// [`draw_chain_starts`] and the per-variant loaders.
///
/// Missing parameters in a `from_mle` / `from_params` / `from_prior`
/// source are handled by those loaders via bounds-uniform fall-back +
/// a stderr warning, not by a distinct error variant. `from_posterior`
/// is the exception: an explicit draws source that can't bind an
/// estimated parameter (missing column or unparseable cell) is a hard
/// [`InitError::SchemaMismatch`], not a silent fall-back (gh#274).
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
    /// `from_prior` was requested but at least one parameter had no
    /// sampleable prior and no bounds for the uniform fall-back. Lists
    /// the offending parameter names.
    NoPriorAndNoBounds { params: Vec<String> },
    /// A sourced rule reached the draw without its source resolved — a
    /// wiring bug ([`resolve_starts`] must run first), never user input.
    Unresolved { rule: String },
    /// I/O error reading a source file.
    Io { path: PathBuf, msg: String },
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::UnknownSource { path } => write!(
                f, "starts source `{}` does not exist", path.display()),
            InitError::SchemaMismatch { path, expected, msg } => write!(
                f, "starts source `{}` does not look like a {}: {}",
                path.display(), expected, msg),
            InitError::NoPriorAndNoBounds { params } => write!(
                f,
                "starts = \"from_prior\" requires either a prior or finite bounds on \
                 every estimated parameter; the following have neither: {}",
                params.join(", ")),
            InitError::Unresolved { rule } => write!(
                f, "internal: starts rule `{rule}` reached the draw with its source \
                    unresolved (resolve_starts must run first)"),
            InitError::Io { path, msg } => write!(
                f, "cannot read `{}`: {}", path.display(), msg),
        }
    }
}

impl std::error::Error for InitError {}

// ─── Source resolution ───────────────────────────────────────────────────────

/// A sourced rule's handle, resolved to the file it reads and the lineage
/// dep that keys the run on that file's content.
#[derive(Debug, Clone)]
pub struct ResolvedSource {
    /// The file the draw reads: `fit_state.toml` (`from_mle`), `draws.tsv`
    /// (`from_posterior`), or the params TOML (`from_params`).
    pub file: PathBuf,
    /// The stored fit leaf the file came from, when the handle named one.
    pub leaf_dir: Option<PathBuf>,
    /// Folded into the method level's `deps`, so a regenerated source
    /// re-keys the run even at the same path.
    pub dep: runid::inputs::ArtifactRef,
}

/// A starts rule with its source, if any, resolved.
#[derive(Debug, Clone)]
pub struct ResolvedStarts {
    pub rule: ChainStarts,
    pub source: Option<ResolvedSource>,
}

impl ResolvedStarts {
    /// A bare rule, which has nothing to resolve.
    pub fn bare(rule: ChainStarts) -> Self {
        ResolvedStarts { rule, source: None }
    }

    /// A sourced rule bound directly to a file — the test seam, and the
    /// shape [`resolve_starts`] produces for a file-shaped handle.
    pub fn from_file(rule: ChainStarts, file: PathBuf) -> Result<Self, String> {
        let artifact = source_artifact_name(&rule);
        let dep = super::cas::cas_file_dep(&file, artifact)
            .ok_or_else(|| format!("starts = {}: cannot read `{}`", rule.spelled(), file.display()))?;
        Ok(ResolvedStarts { rule, source: Some(ResolvedSource { file, leaf_dir: None, dep }) })
    }
}

/// The artifact name a sourced rule's dep records.
fn source_artifact_name(rule: &ChainStarts) -> &'static str {
    match rule {
        ChainStarts::Spread(Spread::FromPosterior { .. }) => "draws.tsv",
        ChainStarts::Point(Point::FromMle { .. }) => "fit_state.toml",
        ChainStarts::Point(Point::FromParams { .. }) => "params.toml",
        _ => "",
    }
}

/// Resolve a rule's handle (proposal §3.1): `from_params` names a flat
/// params TOML directly; `from_posterior` names a draws TSV directly or a
/// fit handle whose leaf wrote one; `from_mle` names a fit handle and reads
/// the point estimate its `fit_state.toml` records. A fit handle goes
/// through [`crate::fit::handle::FitRef::classify`] (`@label`, hash prefix,
/// run directory, `fit.toml`); a segment with one method leaf resolves to
/// it, and one with several is refused with the leaves listed.
///
/// A fit source whose stored verdict is not converged is refused unless
/// `allow_nonconverged` — the purpose of the old pre-refine gate, kept at the
/// one seam where an upstream fit is consumed (proposal §8, item 11).
pub fn resolve_starts(rule: &ChainStarts, allow_nonconverged: bool) -> Result<ResolvedStarts, String> {
    let handle = match rule {
        ChainStarts::Point(Point::FromParams { path }) => {
            return ResolvedStarts::from_file(rule.clone(), path.clone());
        }
        ChainStarts::Spread(Spread::FromPosterior { source })
        | ChainStarts::Point(Point::FromMle { source }) => source.0.clone(),
        _ => return Ok(ResolvedStarts::bare(rule.clone())),
    };
    // A draws TSV named directly.
    if matches!(rule, ChainStarts::Spread(Spread::FromPosterior { .. }))
        && handle.ends_with(".tsv")
    {
        return ResolvedStarts::from_file(rule.clone(), PathBuf::from(&handle));
    }
    let leaf = resolve_fit_leaf(&handle)
        .map_err(|e| format!("starts = {}: {}", rule.spelled(), e))?;
    super::gating::check_source_converged(&leaf, &handle, allow_nonconverged)?;
    let (file, artifact) = match rule {
        ChainStarts::Spread(Spread::FromPosterior { .. }) => (leaf.join("draws.tsv"), "draws.tsv"),
        _ => (leaf.join("fit_state.toml"), "fit_state.toml"),
    };
    if !file.is_file() {
        return Err(format!(
            "starts = {}: {} holds no {} — an optimizer-only method writes no \
             posterior cloud; use `from_mle` for its point estimate",
            rule.spelled(),
            leaf.display(),
            artifact
        ));
    }
    let dep = super::cas::cas_leaf_file_dep(&leaf, artifact)
        .ok_or_else(|| format!("starts = {}: cannot read {}", rule.spelled(), file.display()))?;
    Ok(ResolvedStarts {
        rule: rule.clone(),
        source: Some(ResolvedSource { file, leaf_dir: Some(leaf), dep }),
    })
}

/// A fit handle → the one method leaf it names.
///
/// A leaf is named exactly by its directory or by a prefix of its own
/// `run_id` (what `camdl show` resolves); a segment handle — `@label`, a
/// fit-id prefix, a `fit.toml`, the segment directory — names a leaf only
/// while the segment holds one, since the label lives on the segment and two
/// method files with one problem share it.
pub fn resolve_fit_leaf(handle: &str) -> Result<PathBuf, String> {
    use super::handle::FitRef;
    match FitRef::classify(handle) {
        // A leaf directory named directly.
        FitRef::RunDir(dir) if dir.join("fit_state.toml").is_file() => return Ok(dir),
        // A leaf `run_id` prefix, before the fit-id prefixes a segment answers to.
        FitRef::HashPrefix(prefix) => {
            let root = crate::run_paths::output_root(None, None);
            let mut leaves = crate::cas_read::resolve_fit_prefix(&root, &prefix);
            match leaves.len() {
                0 => {}
                1 => return Ok(leaves.remove(0).dir),
                n => {
                    let listed: Vec<String> =
                        leaves.iter().map(|l| format!("    {}", l.dir.display())).collect();
                    return Err(format!(
                        "{handle} is a prefix of {n} method leaves' run_ids; give more \
                         characters or the leaf directory:\n{}",
                        listed.join("\n")
                    ));
                }
            }
        }
        _ => {}
    }
    let segment = super::handle::resolve_fit_segment(handle).map_err(|e| e.to_string())?;
    let view = super::fit_view::FitView::read(&segment).ok_or_else(|| {
        format!("{} is not a completed fit (no method leaf with a run.json)", segment.display())
    })?;
    match view.stages.len() {
        0 => Err(format!("{} holds no completed method leaf", segment.display())),
        1 => Ok(view.stages[0].stage_dir.clone()),
        n => {
            let listed: Vec<String> =
                view.stages.iter().map(|s| format!("    {}", s.stage_dir.display())).collect();
            Err(format!(
                "{handle} resolves to {n} method leaves (several seeds or cells); pass the \
                 leaf directory to name one:\n{}",
                listed.join("\n")
            ))
        }
    }
}

// ─── Context and entry point ─────────────────────────────────────────────────

/// What the draw needs beyond the rule: the resolved parameters (names,
/// bounds, base values, the model), the `EstimatedParam` specs the
/// transform-aware builders read, and the prior each estimated parameter
/// resolved to — fit-toml over model, through the same precedence the
/// sampler scores against — which is the distribution `from_prior` draws.
pub struct StartContext<'a> {
    pub resolved: &'a ResolvedParameters,
    pub base_specs: &'a [EstimatedParam],
    pub priors: &'a [(String, Prior)],
}

/// Build a minimal `ResolvedParameters` view from a fit-runner-style
/// triple of (model, base_params, estimated_specs). Used by the runners to
/// thread the sourced rules through [`draw_chain_starts`] without forcing a
/// full `ParameterInputs` reconstruction. Provenance from the original
/// resolve is not preserved here (the fit runner already recorded it into
/// run.json upstream); only the fields the draw reads — `model`,
/// `estimate_set`, and per-name `params[i].value` — are populated.
pub fn build_resolved_view_for_init(
    model: &ir::Model,
    base_params: &[f64],
    estimated_specs: &[EstimatedParam],
) -> ResolvedParameters {
    use crate::params_resolver::{
        FixReason, ParameterRole, ResolvedParameter, ValueSource,
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

/// Draw `n_chains` starts under `starts`.
///
/// The bare spread rules fall back to the base point at one chain (there is
/// nothing to spread); the sourced rules read every chain from their source
/// at any chain count. Every rule yields the same shape, so one writer
/// records the result and one provenance serializer reads it.
pub fn draw_chain_starts(
    ctx: &StartContext<'_>,
    starts: &ResolvedStarts,
    n_chains: usize,
    seed: u64,
) -> Result<DrawnStarts, InitError> {
    let rule = starts.rule.clone();
    if n_chains == 0 {
        return Ok(DrawnStarts::fresh(Vec::new(), rule));
    }
    let source_file = || -> Result<&Path, InitError> {
        starts
            .source
            .as_ref()
            .map(|s| s.file.as_path())
            .ok_or_else(|| InitError::Unresolved { rule: rule.spelled() })
    };
    let drawn = match &rule {
        ChainStarts::Point(Point::Declared) => draw_single(ctx.resolved, n_chains),
        ChainStarts::Spread(Spread::Uniform) => {
            if n_chains < 2 {
                draw_single(ctx.resolved, n_chains)
            } else {
                from_specs(
                    super::init::build_uniform_chain_starts(ctx.base_specs, n_chains, seed),
                    |chain_id| InitSource::UniformDraw { seed: derive_chain_seed(seed, chain_id) },
                )
            }
        }
        ChainStarts::Spread(Spread::Lhs) => {
            if n_chains < 2 {
                draw_single(ctx.resolved, n_chains)
            } else {
                from_specs(
                    super::init::build_lhs_chain_starts(ctx.base_specs, n_chains, seed),
                    |chain_id| InitSource::LhsCell { row: chain_id },
                )
            }
        }
        ChainStarts::Spread(Spread::UniformUnconstrained) => {
            if n_chains < 2 {
                draw_single(ctx.resolved, n_chains)
            } else {
                from_specs(
                    super::init::build_uniform_unconstrained_chain_starts(
                        ctx.base_specs, n_chains, seed,
                    ),
                    |chain_id| InitSource::UnconstrainedDraw {
                        seed: derive_chain_seed(seed, chain_id),
                    },
                )
            }
        }
        ChainStarts::Spread(Spread::FromPrior) => {
            draw_from_prior(ctx.resolved, ctx.priors, n_chains, seed)?
        }
        ChainStarts::Spread(Spread::FromPosterior { .. }) => {
            draw_from_posterior(ctx.resolved, source_file()?, n_chains, seed)?
        }
        ChainStarts::Point(Point::FromMle { .. }) => {
            draw_from_mle(ctx.resolved, source_file()?, n_chains)?
        }
        ChainStarts::Point(Point::FromParams { .. }) => {
            draw_from_params(ctx.resolved, source_file()?, n_chains)?
        }
    };
    Ok(DrawnStarts::fresh(drawn, rule))
}

/// Lift the builders' `Vec<Vec<EstimatedParam>>` into `ChainStart`s.
fn from_specs(
    per_chain: Vec<Vec<EstimatedParam>>,
    source: impl Fn(usize) -> InitSource,
) -> Vec<ChainStart> {
    per_chain
        .into_iter()
        .enumerate()
        .map(|(chain_id, specs)| ChainStart {
            chain_id,
            values: specs.iter().map(|s| (s.name.clone(), s.initial)).collect(),
            source: source(chain_id),
        })
        .collect()
}

fn draw_single(resolved: &ResolvedParameters, n_chains: usize) -> Vec<ChainStart> {
    // Every chain starts at the resolved base values. The values map
    // is restricted to `estimate_set`.
    let base = estimate_values_from_resolved(resolved);
    (0..n_chains).map(|chain_id| ChainStart {
        chain_id,
        values: base.clone(),
        source: InitSource::SeededBase,
    }).collect()
}

// ─── Sourced rules ───────────────────────────────────────────────────────────

/// `from_prior`: per-chain draw from each parameter's resolved prior.
/// Parameters whose prior cannot be drawn from — flat, or hierarchical
/// (its hyperparameters have no values at chain-init time) — fall back to a
/// bounds-uniform draw with a startup warning, and are refused when they
/// have no finite bounds either.
fn draw_from_prior(
    resolved: &ResolvedParameters,
    priors: &[(String, Prior)],
    n_chains: usize,
    seed: u64,
) -> Result<Vec<ChainStart>, InitError> {
    let bounds_map = bounds_map_for_estimate(resolved);
    let mut no_prior_names: Vec<String> = Vec::new();
    let mut no_prior_no_bounds: Vec<String> = Vec::new();
    let mut priors_by_name: HashMap<String, Prior> = HashMap::new();
    for name in &resolved.estimate_set {
        let sampleable = priors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
            .filter(|p| !matches!(p, Prior::Fixed(Density::Flat)) && !p.is_hierarchical());
        match sampleable {
            Some(p) => {
                priors_by_name.insert(name.clone(), p.clone());
            }
            None => {
                no_prior_names.push(name.clone());
                if bounds_map.get(name)
                    .map(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
                    .unwrap_or(true)
                {
                    no_prior_no_bounds.push(name.clone());
                }
            }
        }
    }
    if !no_prior_no_bounds.is_empty() {
        return Err(InitError::NoPriorAndNoBounds {
            params: no_prior_no_bounds,
        });
    }
    if !no_prior_names.is_empty() {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m starts = \"from_prior\": no sampleable prior for \
             {}; drawing those parameter(s) uniformly within their bounds instead. \
             Declare a `~ <dist>` prior in the model or a `prior = {{ ... }}` in \
             [estimate] to draw them from one.",
            no_prior_names.join(", "),
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
        // `draw_from_prior` routes hierarchical priors to the bounds
        // fall-back before reaching here.
        Prior::Hierarchical(_) => base.unwrap_or(1.0),
    }
}

/// Marsaglia & Tsang's Gamma(shape, rate) sampler — accepts on a single
/// Normal² rejection step in the shape ≥ 1 branch; uses a recursion
/// `Gamma(s, r) = U^(1/s) · Gamma(s+1, r)` for shape < 1.
fn sample_gamma_shape_rate(rng: &mut StatefulRng, shape: f64, rate: f64) -> f64 {
    if shape < 1.0 {
        // Recursive boost: Gamma(α, r) = U^(1/α) · Gamma(α+1, r).
        let u: f64 = {
            let x = rng.uniform();
            if x <= 0.0 { f64::MIN_POSITIVE } else { x }
        };
        return u.powf(1.0 / shape) * sample_gamma_shape_rate(rng, shape + 1.0, rate);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = 1.0 + c * x;
        if v <= 0.0 { continue; }
        let v3 = v * v * v;
        let u = rng.uniform();
        if u < 1.0 - 0.0331 * x.powi(4)
            || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln())
        {
            return d * v3 / rate;
        }
    }
}

/// `from_posterior`: one row per chain, drawn uniformly with replacement
/// from a draws TSV. An explicit source that cannot bind every estimated
/// parameter — a missing column or an unparseable cell — is a hard error,
/// never a silent bounds-uniform fallback (gh#274).
fn draw_from_posterior(
    resolved: &ResolvedParameters,
    path: &Path,
    n_chains: usize,
    seed: u64,
) -> Result<Vec<ChainStart>, InitError> {
    let path = path.to_path_buf();
    if !path.is_file() {
        return Err(InitError::UnknownSource { path });
    }
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
    // explicitly-requested from_posterior source that cannot bind the
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
    // The literal is grouped to spell its mnemonic, not in equal-width groups.
    #[allow(clippy::unusual_byte_groupings)]
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

/// `from_mle`: every chain at the point estimate a stored fit's
/// `fit_state.toml` records in `start_values` — the clean-eval winner for
/// IF2, the MAP for a sampler, the best chain for an optimizer. The same
/// file the old in-file chaining read, so the two agree exactly.
fn draw_from_mle(
    resolved: &ResolvedParameters,
    path: &Path,
    n_chains: usize,
) -> Result<Vec<ChainStart>, InitError> {
    let path_buf: PathBuf = path.to_path_buf();
    if !path_buf.is_file() {
        return Err(InitError::UnknownSource { path: path_buf });
    }
    let dir = path_buf.parent().map(Path::to_path_buf).unwrap_or_default();
    let state = super::state::FitState::load(&dir.to_string_lossy()).map_err(|e| {
        InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "fit_state.toml of a completed method leaf",
            msg: e,
        }
    })?;
    let values_in_file: HashMap<String, f64> = state.start_values.into_iter().collect();
    apply_point_to_all_chains(resolved, &path_buf, &values_in_file, n_chains,
        |path| InitSource::MlePoint { path })
}

/// `from_params`: all chains at a hand-written flat params TOML.
fn draw_from_params(
    resolved: &ResolvedParameters,
    path: &Path,
    n_chains: usize,
) -> Result<Vec<ChainStart>, InitError> {
    // Reject files that look like fit-output (have `[focal]` or
    // `[mle]` sections, or a `final_loglik` scalar) — the actionable
    // hint redirects the user to `from_mle`.
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
    if table.contains_key("focal") || table.contains_key("mle")
        || table.contains_key("final_loglik")
    {
        return Err(InitError::SchemaMismatch {
            path: path_buf.clone(),
            expected: "flat params TOML (top-level keys = parameter names)",
            msg: "the file looks like mle.toml / fit output (it has a `[focal]` / \
                  `[mle]` section or a `final_loglik` scalar). To start every chain \
                  at a stored fit's estimate write `starts = { from_mle = \"@handle\" }`."
                .into(),
        });
    }
    let mut values_in_file: HashMap<String, f64> = HashMap::new();
    for (key, val) in &table {
        match val {
            toml::Value::Float(f)   => { values_in_file.insert(key.clone(), *f); }
            toml::Value::Integer(i) => { values_in_file.insert(key.clone(), *i as f64); }
            other => {
                return Err(InitError::SchemaMismatch {
                    path: path_buf.clone(),
                    expected: "flat params TOML (top-level keys = parameter names)",
                    msg: format!("key `{}` has non-numeric value {:?}", key, other),
                });
            }
        }
    }
    apply_point_to_all_chains(resolved, &path_buf, &values_in_file, n_chains,
        |path| InitSource::ParamsPoint { path })
}

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
            "\x1b[33mwarning:\x1b[0m starts source `{}` is missing \
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

// ─── The record ─────────────────────────────────────────────────────────────

/// One row of `chain_starts.tsv`.
#[derive(Debug, Clone)]
pub struct ChainStartRecord {
    /// 0-based, as the producer counts chains. The writer renders it 1-based,
    /// which is what the file's `chain_id` column holds (gh#781).
    pub chain_id: usize,
    /// 0-based attempt index within the chain's bounded retry (gh#887).
    pub attempt: usize,
    /// `accepted` (the start the chain ran from), `rejected` (a draw the
    /// filter could not score; a redraw followed), or `refused` (the last
    /// attempt, also unscoreable; the chain did not run).
    pub status: &'static str,
    /// The `source` column: the rule's tag, with `:chain-<id>` appended only
    /// when that chain got a point of its own (gh#871).
    pub source: String,
    /// Values in `base_specs` order.
    pub values: Vec<f64>,
    /// The ESS the filter reached before refusing, when it measured one.
    pub ess: Option<f64>,
    /// Why a rejected or refused start was dropped; empty for an accepted one.
    pub reason: String,
}

impl DrawnStarts {
    /// The rows the writer records, in (chain, attempt) order and
    /// `base_specs` order within a row: every rejected attempt, then the
    /// chain's final start as `accepted` or `refused`.
    pub fn records(&self, base_specs: &[EstimatedParam]) -> Vec<ChainStartRecord> {
        let per_chain = self.to_estimated_params(base_specs);
        let mut rows = Vec::new();
        for (chain_id, specs) in per_chain.iter().enumerate() {
            for r in self.rejected.iter().filter(|r| r.chain_id == chain_id) {
                rows.push(ChainStartRecord {
                    chain_id,
                    attempt: r.attempt,
                    status: "rejected",
                    source: source_label(&self.rule, chain_id),
                    values: base_specs
                        .iter()
                        .map(|s| r.values.get(&s.name).copied().unwrap_or(s.initial))
                        .collect(),
                    ess: r.ess,
                    reason: r.reason.clone(),
                });
            }
            let refused = self.refused.contains(&chain_id);
            rows.push(ChainStartRecord {
                chain_id,
                attempt: self.attempts_before(chain_id),
                status: if refused { "refused" } else { "accepted" },
                source: source_label(&self.rule, chain_id),
                values: specs.iter().map(|s| s.initial).collect(),
                ess: None,
                reason: String::new(),
            });
        }
        rows
    }
}

/// The `source` column for one chain: a rule that drew each chain its own
/// point earns `:chain-<id>`; a rule that put every chain at one point does
/// not, because that suffix reads as independent draws that happen to
/// coincide (gh#871).
///
/// `chain_id` is the 0-based index the producer works in; the suffix is the
/// 1-based number the `chain_id` column, the `chain_N/` directories and every
/// stderr line use, so the two columns of one row agree (gh#781).
pub fn source_label(rule: &ChainStarts, chain_id: usize) -> String {
    if rule.is_point() {
        rule.tag().to_string()
    } else {
        format!("{}:chain-{}", rule.tag(), chain_id + 1)
    }
}

/// Write `chain_starts.tsv` — the sidecar recording every chain's starting
/// parameter vector and its provenance. Lives at the method leaf's root.
/// The one writer, called by every multi-chain sampler (IF2, PGAS, PMMH,
/// MH, NUTS); the optimizer-only methods write none.
///
/// The values are captured before the sampler runs, which is what makes the
/// file worth having: it answers "did the starts span the declared bounds?"
/// and "did the chains collapse into one basin immediately?", and the
/// per-chain trace cannot. An IF2 run perturbs its parameter swarm before
/// the first filter pass (`sim::inference::if2`, the `t=0` perturbation), so
/// iteration 0 of `chain_<chain_id>/parameter_traces.tsv` already shows
/// moved values. The file header says so, because a reader who has only the
/// TSV would otherwise pair the two row-by-row.
///
/// The `chain_id` column is **1-based** — the number the `chain_N/`
/// directories, the stderr refusals and `diagnostics.json` all use — so a
/// refusal naming "chain 4" leads to the row describing chain 4's start with
/// no arithmetic (gh#781). [`ChainStartRecord::chain_id`] stays 0-based;
/// this writer is the boundary.
pub fn write_chain_starts_tsv(
    dir: &Path,
    base: &[EstimatedParam],
    rule: &ChainStarts,
    records: &[ChainStartRecord],
) -> std::io::Result<()> {
    use std::io::Write as _;
    let path = dir.join("chain_starts.tsv");
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let n_chains = records.iter().map(|r| r.chain_id + 1).max().unwrap_or(0);
        let n_retried = records.iter().filter(|r| r.status == "rejected").count();
        // Comment header — stable, machine-parseable.
        writeln!(f, "# camdl chain_starts; starts={}; chains={}; kind={}; retried={}",
            rule.spelled(), n_chains, rule.kind().as_str(), n_retried)?;
        // What the numbers are, for a reader holding only this file.
        writeln!(f, "# each chain's starting point, captured before the \
            sampler ran; an IF2 run")?;
        writeln!(f, "# perturbs its swarm before the first filter pass, so \
            a per-chain trace opens on")?;
        writeln!(f, "# values that have already moved.")?;
        writeln!(f, "# chain_id is 1-based: that chain's outputs are under \
            chain_<chain_id>/, and it is")?;
        writeln!(f, "# the number every stderr refusal and diagnostic uses.")?;
        writeln!(f, "# status: accepted = the start the chain ran from; \
            rejected = a draw the filter")?;
        writeln!(f, "# could not score, so a fresh one was drawn (gh#887); \
            refused = the last attempt,")?;
        writeln!(f, "# also unscoreable, so the chain did not run. ess is \
            the filter's ESS at refusal.")?;
        // Header row.
        let mut cols = vec![
            "chain_id".to_string(), "attempt".to_string(), "status".to_string(),
            "source".to_string(),
        ];
        for spec in base { cols.push(spec.name.clone()); }
        cols.push("ess".to_string());
        cols.push("reason".to_string());
        writeln!(f, "{}", cols.join("\t"))?;
        for rec in records {
            // 1-based on the way out (gh#781): the producer counts chains from
            // zero, every artifact a reader joins this against counts from one.
            let mut fields = vec![
                (rec.chain_id + 1).to_string(), rec.attempt.to_string(),
                rec.status.to_string(), rec.source.clone(),
            ];
            for v in &rec.values {
                fields.push(format_float_for_tsv(*v));
            }
            fields.push(rec.ess.map(|e| format!("{e:.3}")).unwrap_or_default());
            fields.push(rec.reason.replace(['\t', '\n'], " "));
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params_resolver::{ParameterRole, ResolvedParameter, ValueSource};
    use sim::inference::types::Transform;

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

    /// `EstimatedParam` specs for the resolved view's estimate set, with the
    /// bounds the model declares and an identity transform.
    fn specs_for(resolved: &ResolvedParameters) -> Vec<EstimatedParam> {
        resolved.estimate_set.iter().enumerate().map(|(i, name)| {
            let p = resolved.model.parameters.iter().find(|p| &p.name == name).unwrap();
            let (lower, upper) = crate::params_resolver::resolved_bounds(p)
                .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            EstimatedParam {
                name: name.clone(),
                index: i,
                initial: p.value.resolved_value().unwrap(),
                rw_sd: 0.1,
                transform: Transform::None,
                lower,
                upper,
                rw_sd_auto: false,
                perturb_only_at_t0: false,
            }
        }).collect()
    }

    /// The priors the model declares, resolved the way the runner does.
    fn priors_for(resolved: &ResolvedParameters) -> Vec<(String, Prior)> {
        resolved.estimate_set.iter().map(|name| {
            let p = resolved.model.parameters.iter().find(|p| &p.name == name).unwrap();
            let prior = match p.prior_dist() {
                Some(pd) => Prior::from_ir(pd),
                None => Prior::Fixed(Density::Flat),
            };
            (name.clone(), prior)
        }).collect()
    }

    fn draw(
        resolved: &ResolvedParameters,
        starts: &ResolvedStarts,
        n_chains: usize,
        seed: u64,
    ) -> Result<DrawnStarts, InitError> {
        let base_specs = specs_for(resolved);
        let priors = priors_for(resolved);
        let ctx = StartContext { resolved, base_specs: &base_specs, priors: &priors };
        draw_chain_starts(&ctx, starts, n_chains, seed)
    }

    fn from_params(path: &Path) -> ResolvedStarts {
        ResolvedStarts::from_file(
            ChainStarts::Point(Point::FromParams { path: path.to_path_buf() }),
            path.to_path_buf(),
        ).unwrap()
    }

    fn from_posterior(path: &Path) -> ResolvedStarts {
        ResolvedStarts::from_file(
            ChainStarts::Spread(Spread::FromPosterior {
                source: super::super::starts::Handle(path.display().to_string()),
            }),
            path.to_path_buf(),
        ).unwrap()
    }

    fn from_mle(fit_state: &Path) -> ResolvedStarts {
        ResolvedStarts::from_file(
            ChainStarts::Point(Point::FromMle {
                source: super::super::starts::Handle(fit_state.display().to_string()),
            }),
            fit_state.to_path_buf(),
        ).unwrap()
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

    // ─── `from_params` ────────────────────────────────────────────────

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
        let starts = draw(&resolved, &from_params(&path), 3, 42).unwrap();
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
        // — `from_params` must refuse and point at `from_mle`.
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let path = write_tmp("from_params_mle_shape",
            "[focal]\nname = \"beta\"\n\n[mle]\nbeta = 0.42\n");
        let err = draw(&resolved, &from_params(&path), 1, 42).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("from_mle"),
            "error must hint at from_mle: {}", msg);
        assert!(msg.contains("mle.toml"),
            "error must explain the file looks like mle.toml: {}", msg);
        std::fs::remove_file(&path).ok();
    }

    // ─── `from_mle` ───────────────────────────────────────────────────

    /// A `fit_state.toml` the way a completed method leaf writes it, with
    /// only the fields `from_mle` reads populated.
    fn write_fit_state(dir: &Path, values: &[(&str, f64)]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let mut body = String::from(
            "method = \"if2\"\nseed = 1\ntimestamp = \"2026-01-01T00:00:00Z\"\n\
             best_loglik = -10.0\ninitial_loglik = -20.0\nbest_chain = 1\nn_chains = 2\n\n\
             [start_values]\n");
        for (k, v) in values {
            body.push_str(&format!("{k} = {v}\n"));
        }
        body.push_str("\n[rw_sd]\n");
        let path = dir.join("fit_state.toml");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn from_mle_reads_the_stored_point_estimate() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta",  0.3, None, Some((0.0, 1.0))),
                mk_param("gamma", 0.1, None, Some((0.0, 1.0))),
            ],
            &["beta", "gamma"],
        );
        let dir = std::env::temp_dir().join(format!(
            "camdl_from_mle_leaf_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let state = write_fit_state(&dir, &[("beta", 0.55), ("gamma", 0.22)]);
        let starts = draw(&resolved, &from_mle(&state), 2, 0).unwrap();
        for cs in &starts.starts {
            assert!((cs.values["beta"]  - 0.55).abs() < 1e-12);
            assert!((cs.values["gamma"] - 0.22).abs() < 1e-12);
            match &cs.source {
                InitSource::MlePoint { path } => assert_eq!(path, &state),
                other => panic!("unexpected: {:?}", other),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // ─── `from_posterior` ──────────────────────────────────────────────

    // ─── gh#887: the bounded retry's seam ────────────────────────────────

    /// A point rule has nothing to redraw; the bounds-based spread rules fall
    /// back to the base point at one chain, so there is nothing new there
    /// either; every other spread rule admits a fresh draw.
    #[test]
    fn can_redraw_is_a_property_of_the_rule_and_the_chain_count() {
        assert!(!can_redraw(&ChainStarts::Point(Point::Declared), 4));
        assert!(!can_redraw(&ChainStarts::Point(Point::FromMle { source: super::super::starts::Handle("@a".into()) }), 4));
        assert!(!can_redraw(&ChainStarts::Spread(Spread::Lhs), 1));
        assert!(!can_redraw(&ChainStarts::Spread(Spread::UniformUnconstrained), 1));
        assert!(can_redraw(&ChainStarts::Spread(Spread::UniformUnconstrained), 2));
        assert!(can_redraw(&ChainStarts::Spread(Spread::Lhs), 2));
        assert!(can_redraw(&ChainStarts::Spread(Spread::FromPrior), 1));
        assert!(can_redraw(&ChainStarts::Spread(Spread::FromPosterior { source: super::super::starts::Handle("d.tsv".into()) }), 1));
    }

    /// A redraw is a fresh draw under the same rule — a different point,
    /// reproducible from `(seed, attempt)`, and a different one per attempt.
    #[test]
    fn redraw_gives_a_fresh_reproducible_point_under_the_same_rule() {
        let resolved = mk_resolved(
            vec![
                mk_param("beta", 0.3, Some(ir::parameter::PriorDist::LogNormal(
                    ir::parameter::LogNormalPrior { mu: -1.0, sigma: 0.5 })), Some((0.01, 5.0))),
                mk_param("gamma", 0.1, Some(ir::parameter::PriorDist::LogNormal(
                    ir::parameter::LogNormalPrior { mu: -2.0, sigma: 0.5 })), Some((0.01, 1.0))),
            ],
            &["beta", "gamma"],
        );
        let base_specs = specs_for(&resolved);
        let priors = priors_for(&resolved);
        let ctx = StartContext { resolved: &resolved, base_specs: &base_specs, priors: &priors };
        for rule in [
            ChainStarts::Spread(Spread::FromPrior),
            ChainStarts::Spread(Spread::UniformUnconstrained),
            ChainStarts::Spread(Spread::Lhs),
            ChainStarts::Spread(Spread::Uniform),
        ] {
            let starts = ResolvedStarts::bare(rule.clone());
            let first = draw_chain_starts(&ctx, &starts, 3, 7).unwrap();
            for chain_id in 0..3 {
                let a1 = redraw_chain_start(&ctx, &starts, 3, chain_id, 7, 1).unwrap();
                let a1_again = redraw_chain_start(&ctx, &starts, 3, chain_id, 7, 1).unwrap();
                let a2 = redraw_chain_start(&ctx, &starts, 3, chain_id, 7, 2).unwrap();
                assert_eq!(a1.chain_id, chain_id);
                assert_eq!(a1.values, a1_again.values, "{rule}: a redraw is reproducible");
                assert_ne!(a1.values["beta"], first.starts[chain_id].values["beta"],
                    "{rule} chain {chain_id}: a redraw is a different point");
                assert_ne!(a1.values["beta"], a2.values["beta"],
                    "{rule} chain {chain_id}: each attempt is its own draw");
                // In bounds, like any draw under the rule.
                assert!(a1.values["beta"] > 0.01 && a1.values["beta"] < 5.0);
            }
        }
    }

    /// The record carries every attempt: rejected rows first, then the
    /// chain's final start as `accepted` or `refused`, in (chain, attempt)
    /// order, and the writer spells the columns a reader joins on.
    #[test]
    fn records_and_the_writer_carry_every_attempt() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let base_specs = specs_for(&resolved);
        let mut drawn = draw(&resolved, &ResolvedStarts::bare(ChainStarts::Spread(Spread::Lhs)), 2, 1)
            .unwrap();
        let redrawn = ChainStart {
            chain_id: 1,
            values: HashMap::from([("beta".to_string(), 0.42)]),
            source: InitSource::LhsCell { row: 1 },
        };
        drawn = drawn.with_retry_outcome(
            vec![None, Some(redrawn)],
            vec![RejectedStart {
                chain_id: 1, attempt: 0,
                values: HashMap::from([("beta".to_string(), 0.99)]),
                reason: "EssCollapsed at obs_window=3".into(), ess: Some(1.02),
            }],
            vec![],
        );
        let rows = drawn.records(&base_specs);
        let shape: Vec<(usize, usize, &str)> = rows.iter().map(|r| (r.chain_id, r.attempt, r.status)).collect();
        assert_eq!(shape, vec![(0, 0, "accepted"), (1, 0, "rejected"), (1, 1, "accepted")]);
        assert_eq!(rows[1].values, vec![0.99]);
        assert_eq!(rows[1].ess, Some(1.02));
        assert_eq!(rows[2].values, vec![0.42], "the accepted row is the redraw");
        assert_eq!(drawn.starts[1].values["beta"], 0.42, "the chain runs from the redraw");

        let dir = std::env::temp_dir().join(format!(
            "camdl_chain_starts_retry_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        write_chain_starts_tsv(&dir, &base_specs, &drawn.rule, &rows).unwrap();
        let text = std::fs::read_to_string(dir.join("chain_starts.tsv")).unwrap();
        assert!(text.starts_with("# camdl chain_starts; starts=lhs; chains=2; kind=spread; retried=1\n"), "{text}");
        let header = text.lines().find(|l| !l.starts_with('#')).unwrap();
        assert_eq!(header, "chain_id\tattempt\tstatus\tsource\tbeta\tess\treason");
        assert!(text.contains("2\t0\trejected\tlhs:chain-2\t0.99\t1.020\tEssCollapsed at obs_window=3"), "{text}");
        assert!(text.contains("2\t1\taccepted\tlhs:chain-2\t0.42\t\t\n"), "{text}");
        std::fs::remove_dir_all(&dir).ok();

        // A chain whose last attempt failed too is `refused`.
        let refused = drawn.clone().with_retry_outcome(vec![None, None], vec![], vec![1]);
        let rows = refused.records(&base_specs);
        assert_eq!(rows.iter().find(|r| r.chain_id == 1).unwrap().status, "refused");
    }

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
        let starts = draw(&resolved, &from_posterior(&path), 50, 42).unwrap();
        // All values should be from {0.10, 0.20, 0.30, 0.40}.
        let allowed: [f64; 4] = [0.10, 0.20, 0.30, 0.40];
        let used: HashSet<i64> = starts.starts.iter()
            .map(|cs| (cs.values["beta"] * 100.0).round() as i64)
            .collect();
        for u in &used {
            assert!(allowed.iter().any(|a| ((*a * 100.0).round() as i64) == *u),
                "value {} not in allowed {:?}", u, allowed);
        }
        assert!(used.len() >= 3,
            "expected ≥ 3 distinct rows used in 50 draws, got {}: {:?}",
            used.len(), used);
        std::fs::remove_file(&path).ok();
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
        let err = draw(&resolved, &from_posterior(&path), 4, 42).unwrap_err();
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
        let err = draw(&resolved, &from_posterior(&path), 4, 42).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&path.display().to_string()),
            "error must name the file: {}", msg);
        assert!(msg.contains("beta"),
            "error must name the offending column: {}", msg);
        assert!(msg.contains("notanumber"),
            "error must name the offending value: {}", msg);
        std::fs::remove_file(&path).ok();
    }

    // ─── `from_prior` ──────────────────────────────────────────────────

    #[test]
    fn from_prior_falls_back_to_bounds_uniform_with_warning_for_no_prior_params() {
        // beta has a prior, gamma does not. Both are bounded — the
        // fallback uniform-on-bounds path must engage for gamma.
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
        let starts = draw(&resolved, &ResolvedStarts::bare(ChainStarts::from_prior()), 4, 42)
            .unwrap();
        for cs in &starts.starts {
            assert!(cs.values.contains_key("beta"));
            assert!(cs.values.contains_key("gamma"));
            let b = cs.values["beta"];
            let g = cs.values["gamma"];
            assert!((0.0..=1.0).contains(&b));
            assert!((0.0..=1.0).contains(&g));
            match &cs.source {
                InitSource::PriorDraw { .. } => {}
                other => panic!("unexpected source: {:?}", other),
            }
        }
    }

    #[test]
    fn from_prior_uses_the_resolved_prior_not_just_the_model_declaration() {
        // The model declares nothing; the resolved prior (as a fit.toml
        // `prior = { uniform = ... }` would supply through the precedence
        // resolver) is Uniform(0.4, 0.5). Every draw must land in it: the
        // draw reads the prior the sampler scores against, not the model
        // alone.
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.42, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let base_specs = specs_for(&resolved);
        let priors = vec![(
            "beta".to_string(),
            Prior::Fixed(Density::Uniform { lower: 0.4, upper: 0.5 }),
        )];
        let ctx = StartContext { resolved: &resolved, base_specs: &base_specs, priors: &priors };
        let starts = draw_chain_starts(
            &ctx, &ResolvedStarts::bare(ChainStarts::from_prior()), 32, 7,
        ).unwrap();
        for cs in &starts.starts {
            let v = cs.values["beta"];
            assert!((0.4 - 1e-9..=0.5 + 1e-9).contains(&v),
                "draw {} outside the resolved prior [0.4, 0.5]", v);
        }
    }

    #[test]
    fn from_prior_refuses_a_flat_prior_with_no_bounds() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, None)],
            &["beta"],
        );
        let err = draw(&resolved, &ResolvedStarts::bare(ChainStarts::from_prior()), 2, 1)
            .unwrap_err();
        assert!(matches!(err, InitError::NoPriorAndNoBounds { .. }), "{err}");
    }

    // ─── Estimate-set domain invariant (per-variant) ──────────────────

    #[test]
    fn from_params_chainstart_values_restricted_to_estimate_set() {
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
        let starts = draw(&resolved, &from_params(&path), 1, 0).unwrap();
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
        let dir = std::env::temp_dir().join(format!(
            "camdl_from_mle_restricted_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let state = write_fit_state(&dir, &[("beta", 0.66), ("rho", 0.05)]);
        let starts = draw(&resolved, &from_mle(&state), 1, 0).unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
        std::fs::remove_dir_all(&dir).ok();
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
        let starts = draw(&resolved, &from_posterior(&path), 1, 0).unwrap();
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
        let starts = draw(&resolved, &ResolvedStarts::bare(ChainStarts::from_prior()), 1, 0)
            .unwrap();
        let keys: HashSet<&str> = starts.starts[0].values.keys()
            .map(String::as_str).collect();
        assert_eq!(keys, HashSet::from(["beta"]));
    }

    /// gh#885: the one explanation the three all-chains-refused errors share
    /// says both halves — why an unscoreable start happens, and what to do.
    ///
    /// The halves are asserted separately because each was missing somewhere
    /// before: `runner.rs`'s IF2 bail and `pgas.rs`'s start refusal gave
    /// advice without the reason, and the IF2 one recommended tightening the
    /// bounds without saying which direction helps or why.
    #[test]
    fn the_shared_refusal_advice_carries_the_reason_and_the_remedies() {
        let a = UNSCOREABLE_START_ADVICE;
        // The reason (gh#876): the standardised distance, and that it scales.
        assert!(a.contains("standard deviations"), "{a}");
        assert!(a.contains("square root of the population"), "{a}");
        // The remedies, all three.
        assert!(a.contains("`starts = \"single\"`"), "{a}");
        assert!(a.contains("`starts = \"from_prior\"`"), "{a}");
        assert!(a.contains("`particles`"), "{a}");
        // And the anti-remedy, which two of the three messages used to imply.
        assert!(a.contains("widening the parameter bounds"), "{a}");
        assert!(a.contains("more likely, not less"), "{a}");
    }

    /// gh#899: no message this crate emits may *recommend* `--init`. The flag
    /// was removed from `fit run` with the `[stages]` → `[method]` split and
    /// from `camdl profile` by gh#889; chain starts are a `--starts` rule.
    /// Three no-finite-anchor refusals (PGAS, PMMH, IF2) kept telling the user
    /// to "try `--init lhs`", which `fit run` answers by refusing to parse.
    ///
    /// Scan every `.rs` under the cli crate's `src/` and flag any production
    /// line naming `--init ` (the trailing space excludes the live
    /// `simulate --init-state`). One shape is allowed: the removed-flag
    /// corrector's `--init <mode>: write `<replacement>`.` line, which names
    /// the removed flag only to say what replaced it. Comment lines and test
    /// code are skipped, so a comment recording the history stays legal.
    #[test]
    fn no_emitted_message_recommends_the_removed_init_flag() {
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders: Vec<String> = Vec::new();
        let mut stack = vec![src_root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let rel = path.strip_prefix(&src_root).unwrap()
                    .to_string_lossy().replace('\\', "/");
                let text = std::fs::read_to_string(&path).unwrap();
                let mut in_test = false;
                for line in text.lines() {
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("#[cfg(test)]")
                        || trimmed.starts_with("#[test]")
                        || trimmed.starts_with("mod tests")
                    {
                        in_test = true;
                    }
                    if in_test || trimmed.starts_with("//") {
                        continue;
                    }
                    if !line.contains("--init ") || line.contains("--init-state") {
                        continue;
                    }
                    // The removed-flag corrector: names the flag, then says
                    // what to write instead.
                    if line.contains(": write ") {
                        continue;
                    }
                    offenders.push(format!("{}: {}", rel, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "`--init` was removed — these emitted messages recommend a flag \
             `camdl fit run` refuses to parse; say `--starts` instead \
             (gh#899):\n{}",
            offenders.join("\n")
        );
    }

    // ─── Per-variant provenance tag check ─────────────────────────────

    #[test]
    fn every_rule_records_its_provenance_tag() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3,
                Some(ir::parameter::PriorDist::Uniform(
                    ir::parameter::UniformPrior { lower: 0.0, upper: 1.0 })),
                Some((0.0, 1.0)))],
            &["beta"],
        );
        let params = write_tmp("tag_params", "beta = 0.42\n");
        let post = write_tmp("tag_post", "beta\n0.42\n");
        let dir = std::env::temp_dir().join(format!(
            "camdl_tag_mle_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let state = write_fit_state(&dir, &[("beta", 0.42)]);
        let cases: Vec<(ResolvedStarts, &str)> = vec![
            (from_params(&params), "params_point"),
            (from_mle(&state), "mle_point"),
            (from_posterior(&post), "posterior_row"),
            (ResolvedStarts::bare(ChainStarts::from_prior()), "prior_draw"),
            (ResolvedStarts::bare(ChainStarts::Point(Point::Declared)), "seeded_base"),
            (ResolvedStarts::bare(ChainStarts::Spread(Spread::Lhs)), "lhs_cell"),
            (ResolvedStarts::bare(ChainStarts::Spread(Spread::Uniform)), "uniform_draw"),
            (ResolvedStarts::bare(ChainStarts::uniform_unconstrained()), "unconstrained_draw"),
        ];
        for (starts, tag) in cases {
            let drawn = draw(&resolved, &starts, 2, 42).unwrap();
            assert_eq!(drawn.starts[1].source.tag(), tag, "{}", starts.rule.spelled());
        }
        std::fs::remove_file(&params).ok();
        std::fs::remove_file(&post).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_puts_every_chain_at_the_seeded_base() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let starts = draw(
            &resolved, &ResolvedStarts::bare(ChainStarts::Point(Point::Declared)), 3, 0,
        ).unwrap();
        for cs in &starts.starts {
            assert!((cs.values["beta"] - 0.3).abs() < 1e-12);
            assert!(matches!(cs.source, InitSource::SeededBase));
        }
    }

    #[test]
    fn spread_rules_degrade_to_the_base_point_at_one_chain() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        for rule in [
            ChainStarts::Spread(Spread::Lhs),
            ChainStarts::Spread(Spread::Uniform),
            ChainStarts::uniform_unconstrained(),
        ] {
            let starts = draw(&resolved, &ResolvedStarts::bare(rule.clone()), 1, 7).unwrap();
            assert_eq!(starts.starts.len(), 1);
            assert!((starts.starts[0].values["beta"] - 0.3).abs() < 1e-12, "{}", rule.tag());
            assert!(matches!(starts.starts[0].source, InitSource::SeededBase));
        }
    }

    #[test]
    fn a_sourced_rule_without_its_source_is_a_wiring_error() {
        let resolved = mk_resolved(
            vec![mk_param("beta", 0.3, None, Some((0.0, 1.0)))],
            &["beta"],
        );
        let starts = ResolvedStarts::bare(ChainStarts::Point(Point::FromMle {
            source: super::super::starts::Handle("@x".into()),
        }));
        let err = draw(&resolved, &starts, 2, 0).unwrap_err();
        assert!(matches!(err, InitError::Unresolved { .. }), "{err}");
    }

    // ─── chain_starts.tsv source labels (gh#871) ─────────────────────

    /// A rule that drew each chain its own point earns `:chain-<id>`; a rule
    /// that put every chain at one point does not, because that suffix reads
    /// as independent draws that happen to coincide. gh#871.
    #[test]
    fn chain_starts_source_marks_per_chain_draws_only() {
        for rule in [
            ChainStarts::Spread(Spread::Uniform),
            ChainStarts::Spread(Spread::Lhs),
            ChainStarts::uniform_unconstrained(),
            ChainStarts::from_prior(),
        ] {
            let got: Vec<String> = (0..3).map(|i| source_label(&rule, i)).collect();
            assert_eq!(got, vec![format!("{rule}:chain-1"),
                                 format!("{rule}:chain-2"),
                                 format!("{rule}:chain-3")],
                "{rule} draws each chain its own point, so each row names its \
                 chain — by the same 1-based number the `chain_id` column of \
                 the same row carries (gh#781)");
        }
        for rule in [
            ChainStarts::Point(Point::Declared),
            ChainStarts::Point(Point::FromMle { source: super::super::starts::Handle("@up".into()) }),
            ChainStarts::Point(Point::FromParams { path: "p.toml".into() }),
        ] {
            let got: Vec<String> = (0..3).map(|i| source_label(&rule, i)).collect();
            assert_eq!(got, vec![rule.to_string(); 3],
                "{rule} puts every chain at one point, so no row may claim \
                 a draw of its own");
        }
    }

    #[test]
    fn the_writer_records_one_row_per_chain_in_spec_order() {
        let dir = tempfile::tempdir().unwrap();
        let base = vec![
            EstimatedParam {
                name: "beta".into(), index: 0, initial: 0.3, rw_sd: 0.1,
                transform: Transform::None, lower: 0.0, upper: 1.0,
                rw_sd_auto: false, perturb_only_at_t0: false,
            },
        ];
        let rule = ChainStarts::Point(Point::FromMle {
            source: super::super::starts::Handle("@up".into()),
        });
        let accepted = |chain_id: usize| ChainStartRecord {
            chain_id, attempt: 0, status: "accepted",
            source: source_label(&rule, chain_id), values: vec![0.25],
            ess: None, reason: String::new(),
        };
        let records = vec![accepted(0), accepted(1)];
        write_chain_starts_tsv(dir.path(), &base, &rule, &records).unwrap();
        let txt = std::fs::read_to_string(dir.path().join("chain_starts.tsv")).unwrap();
        let header = txt.lines().next().unwrap();
        assert!(header.contains("starts=from_mle @up"), "{header}");
        assert!(header.contains("kind=point"), "{header}");
        let body: Vec<&str> = txt.lines().filter(|l| !l.starts_with('#')).collect();
        assert!(header.ends_with("retried=0"), "{header}");
        assert_eq!(body[0], "chain_id\tattempt\tstatus\tsource\tbeta\tess\treason");
        assert_eq!(body[1], "1\t0\taccepted\tfrom_mle\t0.25\t\t");
        assert_eq!(body[2], "2\t0\taccepted\tfrom_mle\t0.25\t\t");
    }

    /// gh#781: the file's `chain_id` is the number the reader already has —
    /// the `chain_N/` directory, the stderr refusal, the `bad_init` record —
    /// so a refusal naming "chain 2" leads to chain 2's row with no
    /// arithmetic. The producer's 0-based index stays inside the producer.
    ///
    /// The two columns of one row are asserted together: `chain_id` and the
    /// `:chain-<id>` suffix of `source` used to carry different conventions
    /// side by side, which is the form the off-by-one took in the file that
    /// exists to be read while debugging.
    #[test]
    fn the_file_numbers_chains_the_way_every_other_artifact_does() {
        let dir = tempfile::tempdir().unwrap();
        let base = vec![
            EstimatedParam {
                name: "beta".into(), index: 0, initial: 0.3, rw_sd: 0.1,
                transform: Transform::None, lower: 0.0, upper: 1.0,
                rw_sd_auto: false, perturb_only_at_t0: false,
            },
        ];
        let rule = ChainStarts::uniform_unconstrained();
        let records: Vec<ChainStartRecord> = (0..3).map(|chain_id| ChainStartRecord {
            chain_id, attempt: 0, status: "accepted",
            source: source_label(&rule, chain_id),
            values: vec![0.1 * (chain_id as f64 + 1.0)],
            ess: None, reason: String::new(),
        }).collect();
        write_chain_starts_tsv(dir.path(), &base, &rule, &records).unwrap();
        let txt = std::fs::read_to_string(dir.path().join("chain_starts.tsv")).unwrap();

        let rows: Vec<Vec<&str>> = txt.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .skip(1)
            .map(|l| l.split('\t').collect())
            .collect();
        let ids: Vec<&str> = rows.iter().map(|r| r[0]).collect();
        assert_eq!(ids, ["1", "2", "3"],
            "chain_id must name the chain_N/ directory the reader opens, \
             not the producer's 0-based index:\n{txt}");
        for row in &rows {
            assert_eq!(row[3], format!("uniform_unconstrained:chain-{}", row[0]),
                "the `source` suffix and the `chain_id` column of one row must \
                 name the same chain:\n{txt}");
        }
        assert!(txt.contains("# chain_id is 1-based"),
            "the header states the file's own convention:\n{txt}");
    }
}
