//! Is the model a command was given the model a fit ran on — and which
//! simulator did the fit run it on?
//!
//! The second question has one reader, [`stored_fit_simulator`] (with
//! [`fit_config_simulator`] for a fit config), and one rule for the flags over
//! it, [`replay_simulator`]; see the section near the end of this file.
//!
//! A stored fit's parameters — its posterior draws, its `[fixed]` block — mean
//! something only under the model they were estimated with. A command that
//! takes a fit handle *and* a model argument (`simulate --draws posterior
//! --fit`, `simulate --draws <file> --fit <run>`) replays those parameters
//! through whatever model file was named, so a model edited since the fit would
//! otherwise run the fit's draws through different dynamics and exit 0.
//!
//! There is one comparison, [`compare_to_fitted`], and every such check uses
//! it: `fit predict --quantities` (is the source still the fitted model, so a
//! vocabulary compiled against it may be transplanted?) and the replaying
//! `simulate` paths. It is `runid::inputs::model_ir_hash` — the
//! presentation-normalized model identity, which excludes `quantities` and
//! `contrasts` (reporting-only, `Model::hash_into`), the model's file-stem
//! name, and the gradient maps (runid SV=2). That is exactly the question a
//! replay must answer yes to: the same model apart from what it reports.
//!
//! A fit archives its model with the observation anchors resolved
//! (`simulate { to = last_obs + … }` written as a number), so a model compared
//! with a stored fit is first resolved the same way against the fit's own data
//! ([`as_fitted`]); comparing the unresolved source with the resolved archive
//! would refuse the fit's own source.

use std::path::{Path, PathBuf};

use runid::ContentHash;

/// How a given model relates to the model a fit ran on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAgreement {
    /// The same model, reporting layer included.
    Same,
    /// The same model apart from its `quantities {}` / `contrasts {}`: a run
    /// replays the fitted dynamics and reports through the given vocabulary.
    ReportingDiffers,
    /// A different model: the dynamics, observations, parameters or
    /// simulation settings changed. Both identities, for the message.
    Differs { fitted: ContentHash, given: ContentHash },
}

/// Compare `given` against the fit's model `fitted` — the one comparison
/// (module doc).
pub fn compare_to_fitted(fitted: &ir::Model, given: &ir::Model) -> ModelAgreement {
    let fitted_h = runid::inputs::model_ir_hash(fitted);
    let given_h = runid::inputs::model_ir_hash(given);
    if fitted_h != given_h {
        return ModelAgreement::Differs { fitted: fitted_h, given: given_h };
    }
    if fitted.quantities != given.quantities || fitted.contrasts != given.contrasts {
        ModelAgreement::ReportingDiffers
    } else {
        ModelAgreement::Same
    }
}

/// The short form of a model identity used in messages.
pub fn short(h: &ContentHash) -> String {
    h.to_hex()[..8].to_string()
}

/// The notice for [`ModelAgreement::ReportingDiffers`]: the run goes ahead,
/// and says whose reporting vocabulary it uses.
/// `relation` says what the reference model is to `fit` ("ran on", "names").
fn reporting_notice(fit: &str, relation: &str, given_path: &str) -> String {
    format!(
        "note: {given_path} is the model {fit} {relation} apart from its \
         quantities/contrasts; {given_path}'s quantities and contrasts apply"
    )
}

/// Same file, by canonical path. A path that cannot be canonicalized is
/// compared as written.
fn same_file(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// `model` as `fit run` would have archived it: its observation anchors
/// substituted against the window the fit `problem` resolves
/// ([`crate::obs_anchor::fit_window`]), through the one substitution
/// ([`crate::obs_anchor::substitute`]). `fit run` archives the SUBSTITUTED
/// model (horizon a number, the anchor markers cleared), so a source that still
/// carries `last_obs` must go through the same resolution before its identity
/// can be compared with the archive. An unanchored model — the archived IR
/// itself among them — comes back unchanged with no resolved anchors.
///
/// The window is read off the fit's data as it is now (a stored fit records
/// its substituted model, not the window), so data changed since the fit
/// resolves to a different model. That is the right answer: the replay would
/// run a different horizon from the one the parameters were fitted on.
pub fn as_fitted(
    model: &ir::Model,
    problem: &crate::fit::config_v2::Problem,
) -> Result<(ir::Model, Vec<crate::obs_anchor::ResolvedAnchor>), String> {
    let mut m = model.clone();
    if !crate::obs_anchor::model_is_anchored(&m) {
        return Ok((m, Vec::new()));
    }
    let window = crate::obs_anchor::fit_window(&m, problem)?;
    let resolved = crate::obs_anchor::substitute(&mut m, window)?;
    Ok((m, resolved))
}

/// Where `given`'s resolved horizons differ from the fitted model's — the
/// model's `simulate { to }` and each scenario's, by name — said for a message;
/// `None` when every horizon agrees.
fn horizons_moved(fitted: &ir::Model, given: &ir::Model) -> Option<String> {
    let mut out = Vec::new();
    if fitted.simulation.t_end != given.simulation.t_end {
        out.push(format!(
            "the model runs to t = {} where the fitted model ran to t = {}",
            given.simulation.t_end, fitted.simulation.t_end
        ));
    }
    for g in &given.presets {
        let f = fitted.presets.iter().find(|p| p.name == g.name).and_then(|p| p.t_end);
        if let (Some(f), Some(t)) = (f, g.t_end) {
            if f != t {
                out.push(format!("scenario '{}' runs to t = {t} where it ran to t = {f}", g.name));
            }
        }
    }
    (!out.is_empty()).then(|| out.join("; "))
}

/// Do the files at `a` and `b` hold the same bytes? `None` when either cannot
/// be read.
fn same_bytes(a: &Path, b: &Path) -> Option<bool> {
    Some(std::fs::read(a).ok()? == std::fs::read(b).ok()?)
}

/// Refuse to replay a STORED fit's parameters through `given` unless it is the
/// model the fit ran on, read from the fit's archived `model.ir.json`.
///
/// `fit` is the handle as the user typed it; `segment` the fit's problem
/// folder; `problem` the fit's own config — its `[model]` source path
/// (absolute, resolved at load) and the `[data]` its observation anchors
/// resolve against. `given` is the command-line model as loaded, anchors
/// unresolved; it goes through the fit's anchor substitution ([`as_fitted`])
/// before the comparison. Returns the notice to print when only the reporting
/// layer differs, `None` when the models are the same.
pub fn check_stored_fit(
    fit: &str,
    segment: &Path,
    problem: &crate::fit::config_v2::Problem,
    given: &ir::Model,
    given_path: &str,
) -> Result<Option<String>, String> {
    let archive: PathBuf = segment.join("model.ir.json");
    if !archive.is_file() {
        return Err(format!(
            "--fit {fit}: this fit has no archived model ({} is missing), so there is no \
             record of the model its parameters were estimated with to check {given_path} \
             against. Refit to replay it.",
            archive.display()
        ));
    }
    let (fitted, _) = crate::util::load_model(&archive.to_string_lossy())?;
    let (given_resolved, resolved) = as_fitted(given, problem).map_err(|e| {
        format!(
            "--fit {fit}: {given_path} is anchored to observed data, and resolving its \
             anchors against the fit's [data], to compare it with the model the fit ran \
             on, failed: {e}"
        )
    })?;
    match compare_to_fitted(&fitted, &given_resolved) {
        ModelAgreement::Same => Ok(None),
        ModelAgreement::ReportingDiffers => Ok(Some(reporting_notice(fit, "ran on", given_path))),
        ModelAgreement::Differs { fitted: fitted_h, given: given_h } => {
            let source_copy = segment.join("model.camdl.original");
            let recorded_source = problem.model.camdl.as_str();
            let stem = format!(
                "--fit {fit}: {given_path} is not the model this fit ran on (fitted model {}, \
                 {given_path} {})",
                short(&fitted_h),
                short(&given_h),
            );
            let fix = format!(
                "Fix: run with the fitted model, {} (the fit's archived IR), or refit \
                 {given_path}.",
                archive.display()
            );
            // Whether the user changed the source is a question about its bytes,
            // so it is answered by them, not by which path was named.
            if same_bytes(Path::new(given_path), &source_copy) == Some(true) {
                let now = resolved
                    .iter()
                    .map(|r| format!("{} at {} = {}", r.site, r.anchor, r.resolved))
                    .collect::<Vec<_>>()
                    .join(", ");
                let moved = horizons_moved(&fitted, &given_resolved);
                return Err(if let Some(moved) = moved.filter(|_| !resolved.is_empty()) {
                    format!(
                        "{stem}. {given_path} is byte-identical to the source this fit \
                         archived ({}); the difference is in what its observation anchors \
                         resolve to. Against the fit's observation data as it reads now, \
                         {now}, so {moved}: the fit's observation data has changed since the \
                         fit ran, and the replay would run a different horizon from the one \
                         the parameters were fitted on.\n  {fix}",
                        source_copy.display(),
                    )
                } else {
                    let anchors = if resolved.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " (Its observation anchors, resolved against the fit's data as it \
                             reads now, give {now}, the same horizons the fit ran to; a moved \
                             forcing breakpoint, if the data changed, would also differ.)"
                        )
                    };
                    format!(
                        "{stem}, yet {given_path} is byte-identical to the source this fit \
                         archived ({}). Nothing on your side differs: compiled now, the same \
                         source yields a different model from the one the fit archived. A \
                         camdl upgrade since the fit that changes what this source compiles \
                         to would do this; otherwise it is an internal inconsistency in \
                         camdl and worth reporting.{anchors}\n  {fix}",
                        source_copy.display()
                    )
                });
            }
            let source_hint = if !same_file(recorded_source, given_path) {
                format!(
                    " The fit was run from {recorded_source} (use it only if it still \
                     compiles to model {}).",
                    short(&fitted_h)
                )
            } else if source_copy.is_file() {
                format!(
                    " {given_path} is the source the fit was run from, edited since; the \
                     source as fitted is archived at {}.",
                    source_copy.display()
                )
            } else {
                format!(
                    " {given_path} is the source the fit was run from; the fit archived no \
                     copy of that source to compare it against."
                )
            };
            Err(format!(
                "{stem}, so its parameters would be replayed through different dynamics. \
                 Quantities and contrasts are excluded from this comparison: the difference \
                 is in the dynamics, observations, parameters or simulation settings.\n  \
                 {fix}{source_hint}"
            ))
        }
    }
}

/// Refuse a fit CONFIG's parameters (`[fixed]`, `[estimate]` priors) beside a
/// model other than the one its `[model]` names. No fit has run, so the
/// reference is the config's model compiled now (lean — identity is
/// gradient-independent). `config_model` is the config's resolved
/// `[model] camdl` path.
pub fn check_fit_config(
    fit_toml: &str,
    config_model: &str,
    given: &ir::Model,
    given_path: &str,
) -> Result<Option<String>, String> {
    let (ir_path, _tmp) = crate::util::resolve_ir_path(config_model, false)
        .map_err(|e| format!("--fit {fit_toml}: compiling its [model] {config_model}: {e}"))?;
    let (named, _) = crate::util::load_model(&ir_path)?;
    match compare_to_fitted(&named, given) {
        ModelAgreement::Same => Ok(None),
        ModelAgreement::ReportingDiffers => Ok(Some(reporting_notice(fit_toml, "names", given_path))),
        ModelAgreement::Differs { fitted, given: given_h } => Err(format!(
            "--fit {fit_toml}: {given_path} is not the model this fit config names (its \
             [model] is {config_model}, model {}; {given_path} is model {}), so the config's \
             [fixed] values and priors would be applied to a model they were not written \
             for. Quantities and contrasts are excluded from this comparison.\n  \
             Fix: run with {config_model}, or point the config's [model] at {given_path}.",
            short(&fitted),
            short(&given_h),
        )),
    }
}

// ── The simulator a fit ran on ──────────────────────────────────────────────
//
// The model is half of what a fit's parameters were estimated under; the
// simulator is the other half — the forward backend, its integration step, and
// (ODE) the integrator. A replay of the parameters (`simulate --draws
// posterior|prior|<file> --fit`) runs on that simulator by default, as it runs
// on the fitted model, and an explicit flag that differs runs with a notice.

use crate::args::types::{ForwardBackend, IntegratorArg};
use ir::model::Integrator;

/// The simulator a fit's parameters were estimated on.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedSimulator {
    /// The forward backend. `None` for a fit config with no `[method]` (one
    /// read only for its priors and `[fixed]`): it names no simulator.
    pub backend: Option<ForwardBackend>,
    /// The integration step (`crate::dt_rule`).
    pub dt: f64,
    /// The ODE integrator the model declares; what the ODE backend uses.
    pub integrator: Integrator,
}

fn integrator_name(i: &Integrator) -> &'static str {
    match i {
        Integrator::Rk4 => "rk4",
        Integrator::Rk45 { .. } => "rk45",
    }
}

/// `backend` at `dt`, with the integrator on ODE, for messages.
fn describe_simulator(backend: Option<ForwardBackend>, dt: f64, integrator: &Integrator) -> String {
    match backend {
        Some(ForwardBackend::Ode) => format!("ode ({}) at dt = {dt}", integrator_name(integrator)),
        Some(b) => format!("{} at dt = {dt}", b.as_str()),
        None => format!("dt = {dt}"),
    }
}

impl FittedSimulator {
    pub fn describe(&self) -> String {
        describe_simulator(self.backend, self.dt, &self.integrator)
    }
}

/// The simulator of a stored fit — the one reader.
///
/// - **backend**: the named run's own record (its method's `backend` in
///   `run.json`, as `fit predict` and `compare` read it); for a problem-folder
///   handle, the config's `[method]`.
/// - **dt**: `fit.meta.json`'s recorded step; for a fit recorded before the
///   step was, the dt rule over the archived model's `simulate { dt }` and the
///   archived config's `[config] dt` — the two inputs `fit run` resolved it
///   from.
/// - **integrator**: `fit.meta.json`'s, else the archived model's.
///
/// Read by every replay of a stored fit on a simulator the replaying command
/// chooses: `simulate --draws … --fit` and `compare`'s derived filter.
/// `fit predict` and the contrasts read the same two dt inputs through
/// `dt_rule::resolved_dt` and the run's recorded backend.
pub fn stored_fit_simulator(
    resolved: &crate::fit::handle::ResolvedFit,
) -> Result<FittedSimulator, String> {
    // The model the fit ran on: its archive, else (a fit that archived none)
    // the source its config names, as `compare` reads it.
    let archive = resolved.segment.join("model.ir.json");
    let model_src = if archive.is_file() {
        archive.to_string_lossy().into_owned()
    } else {
        resolved.config.problem.model.camdl.clone()
    };
    let (model, _) = crate::util::load_model(&model_src)
        .map_err(|e| format!("reading the model the fit ran on, {model_src}: {e}"))?;
    let sidecar = crate::run_meta::read_fit_sidecar(&resolved.segment)?;
    let backend = resolved
        .leaf
        .as_ref()
        .and_then(|leaf| crate::fit::fit_view::FitView::read(leaf))
        .and_then(|v| v.methods.first().map(|s| ForwardBackend::from(s.backend)))
        .or_else(|| {
            resolved.config.inference.method.as_ref().map(|m| ForwardBackend::from(m.algorithm.backend()))
        });
    let dt = sidecar.as_ref().and_then(|s| s.dt).unwrap_or_else(|| {
        crate::dt_rule::resolved_dt(model.simulation.dt, resolved.config.problem.config.dt)
    });
    let integrator = sidecar
        .and_then(|s| s.integrator)
        .unwrap_or_else(|| model.simulation.integrator.clone());
    Ok(FittedSimulator { backend, dt, integrator })
}

/// The simulator a fit CONFIG describes, for a replay of its priors or
/// `[fixed]` beside `model` (checked to be the model its `[model]` names,
/// [`check_fit_config`]): its `[method]` backend, the dt rule over the model's
/// step and the config's written `[config] dt` (a disagreeing one is refused,
/// as `fit run` refuses it), and the model's integrator.
pub fn fit_config_simulator(fit_toml: &str, model: &ir::Model) -> Result<FittedSimulator, String> {
    let config = crate::fit::config_v2::FitConfig::load(fit_toml)?;
    let raw = std::fs::read_to_string(fit_toml).map_err(|e| format!("cannot read {fit_toml}: {e}"))?;
    let written = crate::fit::config_v2::table_declares_key(&raw, "config", "dt")
        .then_some(config.problem.config.dt);
    let dt = crate::dt_rule::effective_dt(model.simulation.dt, written)
        .map_err(|m| format!("{fit_toml}: {}", m.message("[config] dt")))?;
    Ok(FittedSimulator {
        backend: config.inference.method.as_ref().map(|m| ForwardBackend::from(m.algorithm.backend())),
        dt,
        integrator: model.simulation.integrator.clone(),
    })
}

/// The simulator flags a replay was given (`None` = not passed).
#[derive(Debug, Clone, Copy, Default)]
pub struct SimulatorFlags {
    pub backend: Option<ForwardBackend>,
    pub dt: Option<f64>,
    pub integrator: Option<IntegratorArg>,
}

/// What a replay runs on: the fit's simulator, with each explicit flag applied.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaySimulator {
    pub backend: ForwardBackend,
    pub dt: f64,
    /// The `--integrator` override to apply; `None` runs the model's own,
    /// which is the fit's (the model was checked to be the fitted one).
    pub integrator: Option<IntegratorArg>,
    /// The one notice when a flag moved the replay off the fit's simulator.
    pub notice: Option<String>,
}

/// Resolve the simulator a replay of `fit`'s parameters runs on.
///
/// `fallback_backend` is what the command runs without a fit (its default,
/// or a backend matched from a params file's provenance), used only when the
/// fit names no backend. Each flag replaces its own setting and nothing else:
/// dt is a `[config]` setting every backend integrates at, so it carries
/// across a `--backend` change. The setting that belongs to one backend is the
/// integrator, so `--integrator` on a backend other than ODE is refused — it
/// would select nothing, yet change the run's identity. The fit's own
/// integrator lives in the model and is inert off ODE.
pub fn replay_simulator(
    fit_name: &str,
    fit: &FittedSimulator,
    flags: SimulatorFlags,
    fallback_backend: ForwardBackend,
) -> Result<ReplaySimulator, String> {
    let backend = flags.backend.or(fit.backend).unwrap_or(fallback_backend);
    if flags.integrator.is_some() && backend != ForwardBackend::Ode {
        return Err(format!(
            "--integrator selects the ODE integrator, but this replay of {fit_name} runs on \
             {}, which has none.\n  Fix: drop --integrator, or pass --backend ode.",
            backend.as_str()
        ));
    }
    let dt = flags.dt.unwrap_or(fit.dt);
    let used_integrator = match flags.integrator {
        Some(m) => crate::util::overridden_integrator(&fit.integrator, m),
        None => fit.integrator.clone(),
    };
    let backend_moved = fit.backend.is_some_and(|b| b != backend);
    let dt_moved = dt != fit.dt;
    let integrator_moved = fit.backend == Some(ForwardBackend::Ode)
        && backend == ForwardBackend::Ode
        && used_integrator != fit.integrator;
    let moved: Vec<&str> = [(backend_moved, "--backend"), (dt_moved, "--dt"), (integrator_moved, "--integrator")]
        .iter()
        .filter_map(|(on, flag)| on.then_some(*flag))
        .collect();
    let notice = (!moved.is_empty()).then(|| {
        format!(
            "note: {fit_name} was fitted on {}; this replay runs on {} ({} as given), so the \
             parameters drive a different simulator from the one they were estimated on",
            fit.describe(),
            describe_simulator(Some(backend), dt, &used_integrator),
            moved.join(", ")
        )
    });
    Ok(ReplaySimulator { backend, dt, integrator: flags.integrator, notice })
}

#[cfg(test)]
mod simulator_tests {
    use super::*;

    fn cb(dt: f64) -> FittedSimulator {
        FittedSimulator { backend: Some(ForwardBackend::ChainBinomial), dt, integrator: Integrator::Rk4 }
    }

    fn ode() -> FittedSimulator {
        FittedSimulator { backend: Some(ForwardBackend::Ode), ..cb(1.0) }
    }

    #[test]
    fn no_flag_replays_on_the_fit_simulator_silently() {
        let r = replay_simulator("F", &cb(0.5), SimulatorFlags::default(), ForwardBackend::Gillespie)
            .unwrap();
        assert_eq!(
            (r.backend, r.dt, r.integrator, r.notice),
            (ForwardBackend::ChainBinomial, 0.5, None, None)
        );
        let r = replay_simulator("F", &ode(), SimulatorFlags::default(), ForwardBackend::ChainBinomial)
            .unwrap();
        assert_eq!((r.backend, r.notice), (ForwardBackend::Ode, None));
    }

    #[test]
    fn a_flag_equal_to_the_fit_is_silent() {
        let flags = SimulatorFlags {
            backend: Some(ForwardBackend::ChainBinomial),
            dt: Some(0.5),
            integrator: None,
        };
        let r = replay_simulator("F", &cb(0.5), flags, ForwardBackend::ChainBinomial).unwrap();
        assert!(r.notice.is_none());
        let flags = SimulatorFlags { integrator: Some(IntegratorArg::Rk4), ..Default::default() };
        assert!(replay_simulator("F", &ode(), flags, ForwardBackend::ChainBinomial).unwrap().notice.is_none());
    }

    #[test]
    fn a_differing_flag_runs_with_one_notice_naming_both_simulators() {
        let flags = SimulatorFlags { backend: Some(ForwardBackend::Ode), ..Default::default() };
        let r = replay_simulator("F", &cb(0.5), flags, ForwardBackend::ChainBinomial).unwrap();
        assert_eq!((r.backend, r.dt), (ForwardBackend::Ode, 0.5));
        let n = r.notice.unwrap();
        assert!(
            n.contains("fitted on chain_binomial at dt = 0.5")
                && n.contains("runs on ode (rk4) at dt = 0.5")
                && n.contains("--backend")
                && !n.contains("--dt"),
            "{n}"
        );
        let flags = SimulatorFlags { dt: Some(1.0), ..Default::default() };
        let n = replay_simulator("F", &cb(0.5), flags, ForwardBackend::ChainBinomial)
            .unwrap()
            .notice
            .unwrap();
        assert!(n.contains("--dt") && n.contains("runs on chain_binomial at dt = 1"), "{n}");
    }

    #[test]
    fn an_integrator_off_the_ode_backend_is_refused() {
        let flags = SimulatorFlags {
            backend: Some(ForwardBackend::ChainBinomial),
            integrator: Some(IntegratorArg::Rk45),
            ..Default::default()
        };
        assert!(replay_simulator("F", &ode(), flags, ForwardBackend::ChainBinomial).is_err());
        let flags = SimulatorFlags { integrator: Some(IntegratorArg::Rk45), ..Default::default() };
        assert!(replay_simulator("F", &cb(1.0), flags, ForwardBackend::ChainBinomial).is_err());
        let n = replay_simulator("F", &ode(), flags, ForwardBackend::ChainBinomial)
            .unwrap()
            .notice
            .unwrap();
        assert!(n.contains("runs on ode (rk45)") && n.contains("--integrator"), "{n}");
    }

    #[test]
    fn a_config_without_a_method_takes_the_fallback_backend() {
        let none = FittedSimulator { backend: None, ..cb(0.25) };
        let r = replay_simulator("F", &none, SimulatorFlags::default(), ForwardBackend::Gillespie).unwrap();
        assert_eq!((r.backend, r.dt, r.notice), (ForwardBackend::Gillespie, 0.25, None));
    }
}
