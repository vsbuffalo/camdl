//! `camdl profile` on the run store: the identity of one profile point, the
//! stored result, and the profile table rendered from an invocation's points
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "Profiles" and "A
//! profile, and a grid that grows").
//!
//! A profile is of a fit problem in the store ([`pose_problem`]): a stored
//! fit's, or the one a fit file poses, stored as `camdl fit run` stores it,
//! with what the command line adds written in. A profile point
//! ([`ProfilePoint`]) is one fit of that problem with the focal parameters
//! held at given values, started at one start's values, computed from what
//! the store holds. It is written in place and stores `mle.toml`. Its
//! identity holds the problem, the focal values, the method, the start's
//! values and a seed derived from
//! the user's seed, the focal values and the start's values
//! ([`point_seed`]), and nothing about the grid the point was asked for in.
//! So a larger grid finds every point a smaller one stored, and the same
//! point in two grids is one computation. Starts are drawn once for an
//! invocation, so the start's index is a label: two starts drawn at the same
//! values are one point.
//!
//! A point whose log-likelihood is not finite (the optimizer or the filter
//! failed at those values) is a completed result whose `mle.toml` holds
//! `final_loglik = -inf`: the failure is determined by the point's identity.
//!
//! ```text
//! <build>/profile_point/<model>-…/<focal>-…/<method>-…/seed_<n>-start_<i>-…/
//!   identity.json  run.json  mle.toml  provenance.json
//! ```
//!
//! The profile table is a rendering of an invocation's points
//! ([`ProfileTableRendering`]): one row per grid cell, in the order the
//! invocation asked for its cells, with every seed and start of the cell.

use std::collections::BTreeMap;
use std::io::Write;

use store::{Artifact, ContentAddressed, Dep, Finite, Kinds, Layout, Placed, Primary, Rendering, WriteMode};

use crate::camdl_store::{CompiledModel, ProcessSeed};

// ── The problem ──────────────────────────────────────────────────────────────

/// What the command line adds to the problem `--fit` names: each of these
/// changes what the profile fits, so each is written into the problem that
/// is stored, and the profile is of that problem.
pub(crate) struct ProblemOverrides<'a> {
    /// `--fixed NAME=VALUE`: held at the value and taken out of `[estimate]`.
    pub fixed: &'a [(String, f64)],
    /// `--fixed-file`: its values into `[fixed]`, each name taken out of
    /// `[estimate]`.
    pub fixed_files: &'a [std::path::PathBuf],
    /// The swept parameters, which a `--fixed-file` does not hold: the grid
    /// gives each cell its value.
    pub swept: &'a [String],
    /// `--table NAME=FILE`, over `[data.tables]`.
    pub tables: &'a [(String, std::path::PathBuf)],
    /// `--scenario`, `--enable`, `--disable`, combined with the file's by the
    /// rule `pfilter --fit` uses.
    pub scenario: &'a crate::pfilter_config::ScenarioFlags,
    /// The two policies, the flags over the file's `[config]`.
    pub allow_degenerate_rates: bool,
    pub allow_forcing_extrapolation: bool,
    /// `--dt`, by the one dt rule.
    pub dt: Option<f64>,
}

impl ProblemOverrides<'_> {
    fn any(&self, problem: &crate::fit::config_v2::Problem) -> bool {
        !self.fixed.is_empty()
            || !self.fixed_files.is_empty()
            || !self.tables.is_empty()
            || self.scenario.any()
            || self.allow_degenerate_rates != problem.config.allow_degenerate_rates
            || self.allow_forcing_extrapolation != problem.config.allow_forcing_extrapolation
            || self.dt.is_some()
    }
}

/// The problem a profile's points are of, stored: its dependency, the fit
/// argument reopened on it (every path naming a checked copy of what the
/// store holds), and the compiled IR the cells run.
pub(crate) struct PosedProblem {
    pub dep: Dep<crate::fit_store::FitProblem>,
    pub fit: crate::fit_store::FitArg,
    /// The stored compile, as an IR file.
    pub model_ir: String,
    /// The presentation of the model file the profile names, for a profile
    /// of a stored fit, whose problem's compile is not of that file; `None`
    /// for a fit file's problem, compiled from the file itself.
    pub presentation: Option<crate::camdl_store::PresentationIdentity>,
}

/// The fit problem a profile is of, stored as `camdl fit run` stores the
/// problem of a fit file: `fit`'s problem, a fit file's or a stored fit's,
/// with `overrides` written in. A stored fit with no override is its own
/// problem. The positional `model` must be the model the problem names.
///
/// The cells then compute from what the store holds: the returned fit names
/// the stored problem's copies, with the method and the run settings of
/// `[estimate]` of `fit`.
#[allow(clippy::too_many_arguments)]
#[expect(clippy::disallowed_methods, reason = "reads the model `model` names (declared `#[input(identity)]` on `ProfileArgs`) and the stored fit's model source copied out by `OpenedFit::model_source`, to refuse a profile on another model and to keep the source as provenance")]
pub(crate) fn pose_problem(
    out: &crate::out::Out,
    store: &store::Store,
    info: &store::RunInfo,
    fit: &crate::fit_store::FitArg,
    model: &str,
    compile: crate::util::CompileSettings,
    overrides: &ProblemOverrides<'_>,
) -> Result<PosedProblem, String> {
    use crate::fit_store::{FitProblem, ModelSourceFile, ProblemLabels};
    let given_model = std::fs::read(model).map_err(|e| format!("cannot read the model {model}: {e}"))?;
    // A stored fit's problem is compiled from the model file it keeps; the
    // file this profile names is compiled once, to refuse another model and
    // to be this invocation's presentation.
    let named = match &fit.stored {
        Some(opened) => {
            refuse_another_model(out, opened, &given_model, model, &fit.handle, compile)?;
            Some(named_presentation(&given_model, model, compile)?)
        }
        None => None,
    };
    let dep: Dep<FitProblem> = match (&fit.stored, overrides.any(&fit.config.problem)) {
        (Some(opened), false) => {
            let stem = crate::hashing::path_stem_slug(model).unwrap_or_else(|| "model".to_string());
            let shown = named.clone().expect("a stored fit's named model is compiled");
            crate::camdl_store::presentation::link_presentation(store, info, &opened.fit.model, shown, &stem)?;
            opened.fit.problem.dep_all()
        }
        (stored, _) => {
            let mut problem = fit.config.problem.clone();
            // The model the problem is compiled from, and the file kept as its
            // provenance: the fit file's `[model]`, or the stored fit's compile
            // and source.
            let (source_ir, source_bytes, source_path) = match stored {
                Some(opened) => {
                    let src = opened.model_source()?;
                    let bytes = std::fs::read(&src).map_err(|e| format!("cannot read {}: {e}", src.display()))?;
                    (opened.model_ir.to_string_lossy().into_owned(), bytes, src.to_string_lossy().into_owned())
                }
                None => {
                    let named = &problem.model.camdl;
                    let same = std::fs::canonicalize(named).ok().is_some_and(|a| std::fs::canonicalize(model).ok() == Some(a));
                    if !same {
                        return Err(format!(
                            "the model {model} is not the model the fit file {} names ({named}); a profile \
                             is of the problem the fit file poses, so name its model",
                            fit.handle
                        ));
                    }
                    if problem.is_synthetic_fit() {
                        return Err(format!(
                            "{} generates its data from [synthetic]; profile a fit of real data, or a stored \
                             fit of one generated dataset (`--fit @RUN`)",
                            fit.handle
                        ));
                    }
                    let (ir, _tmp) = crate::util::resolve_ir_path(named, false, None, compile)?;
                    (ir, given_model.clone(), named.clone())
                }
            };
            // The command line's additions, as `fit run`'s flags write
            // theirs into the problem it stores.
            let crate::pfilter_config::ScenarioFlags { scenario, enable, disable } = overrides.scenario;
            let combined = crate::util::combine_fit_scenario(
                crate::util::ScenarioChoice { scenario: scenario.as_deref(), enable, disable },
                crate::util::ScenarioChoice {
                    scenario: problem.scenario.as_deref(),
                    enable: &problem.enable,
                    disable: &problem.disable,
                },
            )?;
            problem.scenario = combined.scenario;
            problem.enable = combined.enable;
            problem.disable = combined.disable;
            problem.config.allow_degenerate_rates = overrides.allow_degenerate_rates;
            problem.config.allow_forcing_extrapolation = overrides.allow_forcing_extrapolation;
            if !overrides.tables.is_empty() {
                let data = problem.data.as_mut().ok_or("the fit problem has no [data] to bind a --table in")?;
                for (name, path) in overrides.tables {
                    data.tables.insert(name.clone(), path.to_string_lossy().into_owned());
                }
            }
            let ir = crate::fit::resolve_anchors_into_temp_ir(Some(out), &source_ir, &problem)?;
            let (ir, _bound) = crate::fit::bind_tables_into_temp_ir(&ir, &problem)?;
            let (model_ir, _) = crate::util::load_ir(&ir)?;
            problem.config.dt =
                crate::dt_rule::effective_dt_for_command(overrides.dt, crate::fit::config_v2::written_config_dt(fit.config_text()), model_ir.simulation.dt, crate::dt_rule::FitFile::Flag)?;
            problem.expand_fixed_from_scenario(&model_ir)?;
            // A name the model does not declare is refused naming where it was
            // typed, before it reaches the problem's [fixed].
            let unknown = |name: &str, from: &str| -> Option<String> {
                (!model_ir.parameters.iter().any(|p| p.name == name)).then(|| {
                    let available: Vec<&str> = model_ir.parameters.iter().map(|p| p.name.as_str()).collect();
                    format!(
                        "unknown parameter '{name}' from {from}.\n  Available parameters: {}{}",
                        if available.is_empty() { "(none)".to_string() } else { available.join(", ") },
                        crate::util::constant_hints(&model_ir, [name])
                    )
                })
            };
            if let Some(e) = overrides.fixed.iter().find_map(|(name, _)| unknown(name, "--fixed")) {
                return Err(e);
            }
            // A `--fixed` the scenario sets or scales would override it in
            // every cell: refused, naming the flag, before it is written in.
            crate::params_resolver::refuse_fixed_flag_over_scenario(&model_ir, problem.scenario.as_deref(), overrides.fixed)?;
            for path in overrides.fixed_files {
                let (values, _) = crate::params_resolver::load_fixed_file(&model_ir, path).map_err(|e| e.to_string())?;
                let from = format!("--fixed-file {}", path.display());
                if let Some(e) = values.keys().find_map(|name| unknown(name, &from)) {
                    return Err(e);
                }
                let mut values: Vec<(String, f64)> = values.into_iter().collect();
                values.sort_by(|a, b| a.0.cmp(&b.0));
                // Every name the file gives is held, out of `[estimate]`. A
                // file's value is below the scenario's (§1.3): a parameter the
                // scenario sets takes the scenario's value, which the
                // problem's scenario selection carries.
                for (name, value) in values.into_iter().filter(|(n, _)| !overrides.swept.contains(n)) {
                    problem.estimate.shift_remove(&name);
                    let set_by_scenario = match problem.scenario.as_deref() {
                        Some(sc) => crate::params_resolver::scenario_only_sets(&model_ir, sc, &name)?,
                        None => false,
                    };
                    if !set_by_scenario {
                        problem.fixed.values.insert(name, value);
                    }
                }
            }
            for (name, value) in overrides.fixed {
                problem.fixed.values.insert(name.clone(), *value);
                problem.estimate.shift_remove(name);
            }
            let stem = crate::hashing::path_stem_slug(&source_path).unwrap_or_else(|| "model".to_string());
            let model_dep = put_compile(store, info, &ir, &stem, &source_bytes, &source_path, named.clone())?;
            let posed = crate::fit_store::resolve_problem(
                &problem,
                &model_ir,
                model_dep,
                ModelSourceFile { bytes: source_bytes, camdl: crate::util::model_is_camdl_source(&source_path) },
                crate::fit_store::DataOrigin::Files,
            )?;
            let labels = ProblemLabels { model: stem, sweep_point: None };
            let stored = match crate::camdl_store::begin::<FitProblem>(store, &posed.identity, &labels, info, store::Reuse::IfCompleted, crate::fit_store::problem_files(&posed)).map_err(|e| e.to_string())? {
                store::Begin::Reused(stored) => stored,
                store::Begin::Busy(owner) => return Err(store::Error::InProgress(owner).to_string()),
                store::Begin::Claimed(mut run) => match crate::fit_store::write_problem(&mut run, &posed) {
                    Ok(()) => run.finish().map_err(|e| e.to_string())?,
                    Err(e) => {
                        let _ = run.fail("writing the problem", &e);
                        return Err(e);
                    }
                },
            };
            stored.dep_all()
        }
    };
    // The cells read the stored problem: reopened by its id, with the fit's
    // method as named (its `[method.perturbation]` with it) and the starts of
    // its `[estimate]`.
    let mut posed = crate::fit_store::FitArg::read(&dep.id().key().to_hex(), Some(store))?;
    posed.handle = fit.handle.clone();
    posed.config.inference = fit.config.inference.clone();
    for (name, spec) in posed.config.problem.estimate.iter_mut() {
        if let Some(given) = fit.config.problem.estimate.get(name) {
            spec.start = given.start;
        }
    }
    let model_ir = posed.stored.as_ref().map(|o| o.model_ir.to_string_lossy().into_owned()).ok_or("internal: the stored problem reopened as no stored fit")?;
    Ok(PosedProblem { dep, fit: posed, model_ir, presentation: named })
}

/// The compile with the state Jacobian of `posed`'s model, which an
/// `nl-lbfgs` cell runs, stored: rewritten from the problem's data as the
/// problem's compile is, and refused unless it is the same model as
/// `problem_ir`. It is stored with `presentation`, the presentation of the
/// model file the invocation names ([`PosedProblem::presentation`]), or,
/// when that is `None`, of the model file the problem keeps. Returns its IR
/// file and its dependency.
#[expect(clippy::disallowed_methods, reason = "reads the stored model source copied out by `OpenedFit::model_source` and the IR camdlc compiled from it, which is stored as a `CompiledModel` dependency")]
pub(crate) fn jacobian_compile(
    out: &crate::out::Out,
    store: &store::Store,
    info: &store::RunInfo,
    posed: &crate::fit_store::FitArg,
    presentation: Option<crate::camdl_store::PresentationIdentity>,
    problem_ir: &str,
    compile: crate::util::CompileSettings,
) -> Result<(String, Dep<CompiledModel>), String> {
    let opened = posed.stored.as_ref().ok_or("internal: the posed problem is not stored")?;
    let src = opened.model_source()?;
    let src_str = src.to_string_lossy().into_owned();
    let bytes = std::fs::read(&src).map_err(|e| format!("cannot read {}: {e}", src.display()))?;
    let (full, _tmp) = crate::util::resolve_ir_path(&src_str, true, None, compile)?;
    let full = crate::fit::resolve_anchors_into_temp_ir(Some(out), &full, &posed.config.problem)?;
    let (full, _bound) = crate::fit::bind_tables_into_temp_ir(&full, &posed.config.problem)?;
    let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("cannot read {p}: {e}"));
    if crate::resolve::model_identity_from_ir(&read(problem_ir)?) != crate::resolve::model_identity_from_ir(&read(&full)?) {
        return Err("internal error: the compile with the state Jacobian is not the same model as the problem's".into());
    }
    let stem = opened.model_name();
    let dep = put_compile(store, info, &full, &stem, &bytes, &src_str, presentation)?;
    Ok((full, dep))
}

/// The presentation of the model file `model`, whose bytes are `bytes`,
/// compiled under `compile` and rendered.
fn named_presentation(
    bytes: &[u8],
    model: &str,
    compile: crate::util::CompileSettings,
) -> Result<crate::camdl_store::PresentationIdentity, String> {
    let envelope = crate::util::load_model_envelope(model, compile)?;
    let source = crate::camdl_store::ModelSource::read(bytes, std::path::Path::new(model));
    Ok(crate::camdl_store::CompiledParts::split(envelope, String::new(), source)?.presentation)
}

/// Refuses a positional model other than the one a stored fit was posed on.
/// A file with the bytes of the model file the stored problem keeps is that
/// model. Another file is compiled under `compile` and rewritten from the
/// problem's data as the problem's compile was, and is the same model when
/// its structural hash, with the description left out, is the stored
/// compile's: a comment, a `#'` docstring or the description is not another
/// model. The kept file is never compiled, since files it reads by a
/// relative path are not beside its copy.
#[expect(clippy::disallowed_methods, reason = "reads the stored model source copied out by `OpenedFit::model_source`, and the stored and rewritten compiles, only to refuse a positional model that differs from them")]
fn refuse_another_model(
    out: &crate::out::Out,
    opened: &crate::fit_store::OpenedFit,
    given: &[u8],
    model: &str,
    handle: &str,
    compile: crate::util::CompileSettings,
) -> Result<(), String> {
    let src = opened.model_source()?;
    let kept = std::fs::read(&src).map_err(|e| format!("cannot read {}: {e}", src.display()))?;
    if kept == given {
        return Ok(());
    }
    let problem = &opened.config.problem;
    let (ir, _tmp) = crate::util::resolve_ir_path(model, false, None, compile)?;
    let ir = crate::fit::resolve_anchors_into_temp_ir(Some(out), &ir, problem)?;
    let (ir, _bound) = crate::fit::bind_tables_into_temp_ir(&ir, problem)?;
    let structural = |path: &str| -> Result<ir::model_hash::ContentHash, String> {
        let mut m = crate::util::load_ir(path)?.0;
        m.description = None;
        Ok(ir::model_hash::model_ir_hash(&m))
    };
    if structural(&ir)? != structural(&opened.model_ir.to_string_lossy())? {
        return Err(format!(
            "the model {model} is not the model the stored fit {handle} was posed on; a profile of a \
             stored fit is of its problem, so name the model it keeps ({})",
            crate::fit_store::shown(&src).display()
        ));
    }
    Ok(())
}

/// Stores the compile at `ir`, as `fit run` stores a problem's: the model
/// part, depended on, and its quantities, named as lineage. The compile is
/// stored with `presentation` when one is given, and otherwise with the
/// presentation of `source`, the model file at `source_path` it was
/// compiled from.
#[expect(clippy::disallowed_methods, reason = "reads the IR camdlc compiled, which this function stores as the `CompiledModel` the result depends on")]
pub(crate) fn put_compile(
    store: &store::Store,
    info: &store::RunInfo,
    ir: &str,
    stem: &str,
    source: &[u8],
    source_path: &str,
    presentation: Option<crate::camdl_store::PresentationIdentity>,
) -> Result<Dep<CompiledModel>, String> {
    use store::ContentArtifact;
    let text = std::fs::read_to_string(ir).map_err(|e| format!("cannot read the compiled model {ir}: {e}"))?;
    let envelope = ir::envelope_from_str(&text).map_err(|e| format!("IR load error from {ir}: {e}"))?;
    let path = std::path::Path::new(source_path);
    let source = match presentation {
        Some(_) => crate::camdl_store::ModelSource { bytes: source, path, rendered: crate::camdl_store::Rendered::nothing() },
        None => crate::camdl_store::ModelSource::read(source, path),
    };
    let parts = crate::camdl_store::CompiledParts::split(envelope, stem.to_string(), source)?;
    let crate::camdl_store::CompiledParts { model, quantities, presentation: of_source, labels, provenance } = parts;
    let presentation = presentation.unwrap_or(of_source);
    let quantities = store
        .put::<crate::camdl_store::Quantities>(quantities, &labels, &store::Provenance::none(), info)
        .map_err(|e| format!("storing the quantities: {e}"))?
        .dep(crate::camdl_store::Quantities::FILE)
        .map_err(|e| format!("storing the quantities: {e}"))?;
    info.lineage(&quantities);
    crate::camdl_store::put_compiled(store, info, model, presentation, &labels, &provenance)
        .map_err(|e| format!("storing the compiled model: {e}"))
}

// ── The method ───────────────────────────────────────────────────────────────

/// How each point is fitted: the algorithm, its settings, the step, and how
/// each estimated parameter is moved.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_method", version = 1)]
pub(crate) struct ProfileMethod {
    /// `if2`, `nl-sbplx`, `nl-bobyqa` or `nl-lbfgs`.
    pub algorithm: String,
    pub cell: CellSettings,
    pub dt: Finite,
    /// Each estimated parameter, in the order the cell estimates them.
    pub estimated: Vec<EstimatedSetting>,
    /// The `[[simplex_groups]]` IF2 perturbs jointly, member names in order.
    pub simplex_groups: Vec<Vec<String>>,
    /// The compile with the state Jacobian, which an `nl-lbfgs` cell runs;
    /// `None` for every other algorithm, which runs the problem's compile.
    pub jacobian_model: Option<Dep<CompiledModel>>,
}

impl ProfileMethod {
    /// The method group's label: the algorithm's name.
    pub fn label(&self) -> String {
        self.algorithm.clone()
    }
}

/// What one algorithm's cell reads.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_cell", version = 1)]
pub(crate) enum CellSettings {
    If2 {
        particles: u64,
        iterations: u64,
        /// The fraction the perturbation reaches at `cooling_target_iters`.
        cooling: Finite,
        cooling_target_iters: u64,
        /// The budget each filter evaluation may spend; a cell whose budget
        /// trips is stored with a log-likelihood of -inf.
        max_substeps: u64,
        binomial: crate::pfilter_store::Binomial,
    },
    Nlopt {
        tolerance: Finite,
        max_evals: u64,
    },
}

/// One estimated parameter as a cell moves it: its domain, and under IF2 its
/// perturbation.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_estimated", version = 1)]
pub(crate) struct EstimatedSetting {
    pub name: String,
    pub domain: Domain,
    /// The IF2 perturbation, its SD in z and its timing; `None` for an nl-*
    /// cell, which perturbs nothing and reads no step.
    pub perturbation: Option<crate::fit_store::PerturbationIdentity>,
}

/// An estimated parameter's domain, as the map onto it
/// (`sim::inference::types::Transform`).
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_domain", version = 1)]
pub(crate) enum Domain {
    Unbounded,
    LowerBounded { lo: Finite },
    UpperBounded { hi: Finite },
    Interval { lo: Finite, hi: Finite },
    LogInterval { lo: Finite, hi: Finite },
}

fn finite(x: f64, what: &str) -> Result<Finite, String> {
    Finite::new(x).map_err(|e| format!("{what}: {e}"))
}

impl EstimatedSetting {
    /// The setting of `p`, whose IF2 step is `step`; `None` for a cell that
    /// takes no step.
    pub fn of(
        p: &sim::inference::if2::EstimatedParam,
        step: Option<&crate::fit::runner::WrittenStep>,
    ) -> Result<Self, String> {
        use sim::inference::types::Transform as T;
        let what = |k: &str| format!("the {k} of `{}`", p.name);
        let domain = match p.transform {
            T::Identity => Domain::Unbounded,
            T::LowerBounded { lo } => Domain::LowerBounded { lo: finite(lo, &what("lower bound"))? },
            T::UpperBounded { hi } => Domain::UpperBounded { hi: finite(hi, &what("upper bound"))? },
            T::Interval { lo, hi } => Domain::Interval { lo: finite(lo, &what("lower bound"))?, hi: finite(hi, &what("upper bound"))? },
            T::LogInterval { lo, hi } => Domain::LogInterval { lo: finite(lo, &what("lower bound"))?, hi: finite(hi, &what("upper bound"))? },
        };
        Ok(EstimatedSetting {
            name: p.name.clone(),
            domain,
            perturbation: match step {
                Some(step) => Some(crate::fit_store::PerturbationIdentity {
                    sd: finite(step.z_sd, &what("random-walk step"))?,
                    at: step.at.into(),
                }),
                None => None,
            },
        })
    }
}

// ── The point ────────────────────────────────────────────────────────────────

/// One point of a profile: a fit of the problem with the focal parameters
/// held at given values, from one start.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_point", version = 1)]
pub(crate) struct ProfilePointIdentity {
    /// The fit problem it profiles.
    pub problem: Dep<crate::fit_store::FitProblem>,
    /// The focal parameters' values.
    pub focal: BTreeMap<String, Finite>,
    pub method: ProfileMethod,
    /// The start's values of the estimated parameters, not its index.
    pub start: BTreeMap<String, Finite>,
    /// From the user's seed, the focal values, the start's values and, for
    /// the second and later of starts with equal values, its replicate index.
    pub seed: ProcessSeed,
}

/// What [`point_seed`] derives a point's seed from.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_point_seed", version = 1)]
struct PointSeed {
    seed: u64,
    focal: BTreeMap<String, Finite>,
    start: BTreeMap<String, Finite>,
}

/// What [`point_seed`] derives the seed of a replicate from: a start whose
/// values equal those of `replicate` earlier starts of the invocation.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "profile_point_replicate_seed", version = 1)]
struct ReplicateSeed {
    seed: u64,
    focal: BTreeMap<String, Finite>,
    start: BTreeMap<String, Finite>,
    replicate: u32,
}

/// A point's seed: the first eight bytes, little-endian, of the key of the
/// user's seed with the point's focal values and start's values, and with
/// `replicate` when it is not 0. `replicate` is the number of earlier starts
/// of the invocation with the same values as this one, so equal starts are
/// independent runs, and a start's index is stable when `--n-starts` grows.
/// The seed depends on nothing about the grid, so one point has one seed in
/// every invocation that asks for it.
pub(crate) fn point_seed(seed: u64, focal: &BTreeMap<String, Finite>, start: &BTreeMap<String, Finite>, replicate: u32) -> ProcessSeed {
    let key = match replicate {
        0 => PointSeed { seed, focal: focal.clone(), start: start.clone() }.content_hash(),
        replicate => ReplicateSeed { seed, focal: focal.clone(), start: start.clone(), replicate }.content_hash(),
    };
    let mut first = [0u8; 8];
    first.copy_from_slice(&key.as_bytes()[..8]);
    ProcessSeed { seed: u64::from_le_bytes(first) }
}

/// The model file's stem, the focal values as a label
/// (`beta=0.3000__gamma=0.1000`), the method's name, the seed the user typed
/// and the start's index in the invocation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProfilePointLabels {
    pub model: String,
    pub focal: String,
    pub method: String,
    pub seed: u64,
    pub start: u32,
}

/// The focal values as a label: each `name=value` to four decimals, sorted
/// by name and joined by `__`.
pub(crate) fn focal_label(focal: &BTreeMap<String, Finite>) -> String {
    focal.iter().map(|(n, v)| format!("{n}={:.4}", v.get())).collect::<Vec<_>>().join("__")
}

/// One profile point, written in place.
pub(crate) struct ProfilePoint;

impl Artifact for ProfilePoint {
    const KIND: &'static str = "profile_point";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = ProfilePointIdentity;
    type Labels = ProfilePointLabels;
}

impl Primary for ProfilePoint {
    fn layout(id: &ProfilePointIdentity, labels: &ProfilePointLabels) -> Placed {
        Layout::in_kind()
            .group(&labels.model, &id.problem)
            .group(&labels.focal, &id.focal)
            .group(&labels.method, &id.method)
            .result(&format!("seed_{}-start_{}", labels.seed, labels.start))
    }
}

/// The point's result file: its log-likelihood, focal values, estimates and
/// convergence diagnostics.
pub(crate) const MLE_FILE: &str = "mle.toml";
/// The point's provenance: where each parameter's value came from, the
/// notices, the partition, the start's provenance and the wall time.
pub(crate) const PROVENANCE_FILE: &str = "provenance.json";

/// Adds every kind of this module to `kinds`.
pub(crate) fn register(kinds: Kinds) -> Kinds {
    kinds.register::<ProfilePoint>()
}

// ── The profile table ────────────────────────────────────────────────────────

/// The profile table, rendered from an invocation's points: one row per
/// grid cell, in the order the invocation first asked for each cell, its
/// starts every point the invocation asked for at the cell, in order. A
/// point asked for twice (two starts drawn at the same values) counts as
/// two starts.
pub(crate) struct ProfileTableRendering;

/// How the profile table is rendered.
pub(crate) struct ProfileTableOptions {
    /// The estimated parameters, in the cell's order.
    pub estimated: Vec<String>,
    pub objective: crate::fit::config_v2::LikelihoodObjective,
    pub loglik_type: &'static str,
    pub format: crate::args::ProfileTableFormat,
}

#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
impl Rendering for ProfileTableRendering {
    type Of = ProfilePoint;
    type Options = ProfileTableOptions;
    fn render(members: &[store::Member<ProfilePoint>], options: &ProfileTableOptions, out: &mut dyn Write) -> store::Result<()> {
        let parse = |why: String| store::Error::Parse { what: "a stored profile point".into(), why };
        let mut focal_names: Vec<String> = Vec::new();
        let mut cells: Vec<(BTreeMap<String, Finite>, Vec<Option<crate::profile_table::StartOutcome>>)> = Vec::new();
        for (stored, _) in members {
            let focal = stored.identity().focal.clone();
            if focal_names.is_empty() {
                focal_names = focal.keys().cloned().collect();
            }
            let bytes = crate::camdl_store::read_stored(stored, MLE_FILE).map_err(parse)?;
            let start = crate::profile_table::read_start(&bytes, &options.estimated, stored.id().key().to_hex()).map_err(parse)?;
            match cells.iter_mut().find(|(f, _)| *f == focal) {
                Some((_, starts)) => starts.push(Some(start)),
                None => cells.push((focal, vec![Some(start)])),
            }
        }
        let table = crate::profile_table::ProfileTable {
            focal_names,
            estimated: options.estimated.clone(),
            objective: options.objective,
            loglik_type: options.loglik_type,
            rows: cells
                .into_iter()
                .map(|(focal, starts)| crate::profile_table::CellRow { focal: focal.values().map(|v| v.get()).collect(), starts })
                .collect(),
        };
        let io = |e: std::io::Error| store::Error::Io { op: "write", path: "the profile table".into(), source: e };
        match options.format {
            crate::args::ProfileTableFormat::Tsv => out.write_all(table.to_tsv().as_bytes()).map_err(io),
            crate::args::ProfileTableFormat::Json => {
                let mut s = serde_json::to_string_pretty(&table.to_json()).map_err(|e| parse(e.to_string()))?;
                s.push('\n');
                out.write_all(s.as_bytes()).map_err(io)
            }
        }
    }
}

// ── Reading stored profiles ──────────────────────────────────────────────────

/// The table options of a profile, from the identity record of one of its
/// points and its stored problem: the estimated parameters, the likelihood
/// objective and the class of the log-likelihood, which every point of one
/// invocation shares.
pub(crate) fn table_options(
    store: &store::Store,
    point: &ProfilePointIdentityRecord,
    format: crate::args::ProfileTableFormat,
) -> Result<ProfileTableOptions, String> {
    use crate::fit::config_v2::LikelihoodObjective as L;
    let problem = store
        .get(point.problem.id())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("the profile's problem {} is not stored", point.problem.id().key().short12()))?;
    let objective = if problem.identity().ic_free { L::ConditionalOnFirstObservation } else { L::Full };
    let loglik_type = crate::fit::methods::parse_algorithm(&point.method.algorithm)
        .map(|a| crate::fit::loglik::LoglikType::from(a).tag())
        .unwrap_or("unknown");
    Ok(ProfileTableOptions {
        estimated: point.method.estimated.iter().map(|e| e.name.clone()).collect(),
        objective,
        loglik_type,
        format,
    })
}

/// One `camdl profile` invocation on the run store, for `camdl list`: its
/// record's sequence number and label, and the points it asked for.
pub(crate) struct ProfileInvocationRow {
    pub seq: u64,
    pub label: Option<String>,
    pub model: String,
    pub method: String,
    /// The focal parameters, by name.
    pub focal: Vec<String>,
    /// Every point the invocation asked for, by run id, once.
    pub points: Vec<store::Key>,
    /// How many of them are completed.
    pub stored: usize,
    /// When the earliest stored point's folder was made.
    pub created: std::time::SystemTime,
}

impl ProfileInvocationRow {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "profile",
            "invocation": self.seq,
            "label": self.label,
            "model": self.model,
            "method": self.method,
            "focal": self.focal,
            "points": self.points.iter().map(|k| k.to_hex()).collect::<Vec<_>>(),
            "stored": self.stored,
        })
    }
}

/// Every invocation in the run store at `root` that asked for profile
/// points, oldest first.
pub(crate) fn discover_invocations(root: &std::path::Path) -> Vec<ProfileInvocationRow> {
    let Some(store) = crate::camdl_store::open_existing(root) else { return Vec::new() };
    let mut rows = Vec::new();
    for inv in store.invocations() {
        let Ok(members) = inv.members::<ProfilePoint>() else { continue };
        let Some((first_id, first)) = members.first().cloned() else { continue };
        let mut points: Vec<store::Key> = Vec::new();
        let mut stored = 0;
        let mut created = std::time::SystemTime::now();
        let mut focal: Vec<String> = Vec::new();
        for (id, _) in &members {
            if points.contains(&id.key()) {
                continue;
            }
            points.push(id.key());
            if let Ok(Some(s)) = store.get(id) {
                stored += 1;
                if let Ok(t) = std::fs::metadata(s.folder()).and_then(|m| m.modified()) {
                    created = created.min(t);
                }
                if focal.is_empty() {
                    focal = s.identity().focal.keys().cloned().collect();
                }
            }
        }
        let _ = first_id;
        rows.push(ProfileInvocationRow {
            seq: inv.seq(),
            label: inv.label().map(str::to_string),
            model: first.model,
            method: first.method,
            focal,
            points,
            stored,
            created,
        });
    }
    rows
}

/// The profile table of the invocation a label names, when that invocation
/// asked for profile points: `None` for any other handle or invocation.
pub(crate) fn invocation_table(root: &std::path::Path, target: &str) -> Option<Result<Vec<u8>, String>> {
    let handle: store::Handle = target.parse().ok()?;
    if !matches!(handle, store::Handle::Label { .. }) {
        return None;
    }
    let store = crate::camdl_store::open_existing(root)?;
    let invocation = store.invocation(&handle).ok()?;
    let (first, _) = invocation.members::<ProfilePoint>().ok()?.into_iter().next()?;
    let point = match store.get(&first) {
        Ok(Some(p)) => p,
        Ok(None) => return Some(Err(format!("{target}: its first point {} is not stored", first.key().short12()))),
        Err(e) => return Some(Err(e.to_string())),
    };
    let options = match table_options(&store, point.identity(), crate::args::ProfileTableFormat::Tsv) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    let mut table = Vec::new();
    Some(invocation.render_to::<ProfileTableRendering>(&store, &options, &mut table).map(|()| table).map_err(|e| e.to_string()))
}

/// `camdl show` of a profile point: where it is, its focal values, its
/// method, its seed and start, and its log-likelihood and estimates.
pub(crate) fn show_point(out: &crate::out::Out, point: &store::Stored<ProfilePoint>, rel_path: &str) {
    use crate::out::paint::{self as style, Style};
    let field = |name: &str, value: &dyn std::fmt::Display| {
        out.report_line(format_args!("{}", style::stdout().paint(name, Style::Dim)));
        out.report_line(format_args!("  {value}"));
    };
    let id = point.identity();
    field("path", &style::stdout().paint(rel_path, Style::Code));
    field("kind", &ProfilePoint::KIND);
    field("method", &id.method.algorithm);
    field("seed", &id.seed.seed);
    field("run_id", &style::stdout().paint(&point.id().key().to_hex(), Style::Dim));
    for (title, values) in [("focal", &id.focal), ("start", &id.start)] {
        out.report_line(format_args!("{}", style::stdout().paint(title, Style::Dim)));
        for (name, v) in values {
            out.report_line(format_args!("  {:<16} {}", name, v.get()));
        }
    }
    let estimated: Vec<String> = id.method.estimated.iter().map(|e| e.name.clone()).collect();
    if let Ok(start) = crate::camdl_store::read_stored(point, MLE_FILE)
        .and_then(|b| crate::profile_table::read_start(&b, &estimated, String::new()))
    {
        field("final_loglik", &start.loglik);
        out.report_line(format_args!("{}", style::stdout().paint("mle", Style::Dim)));
        for (name, v) in estimated.iter().zip(&start.estimates) {
            out.report_line(format_args!("  {:<16} {}", name, v));
        }
    }
}

#[cfg(test)]
mod tests;