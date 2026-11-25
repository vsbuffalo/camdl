//! Is the model a command was given the model a fit ran on — and which
//! simulator did the fit run it on?
//!
//! The second question has one reader, [`stored_fit_simulator`] (with
//! [`fit_config_simulator`] for a fit config); see the section near the end of
//! this file. The rule for the flags over it is `crate::simulator`.
//!
//! A stored fit's parameters — its posterior draws, its `[fixed]` block — mean
//! something only under the model they were estimated with. A command that
//! takes a fit handle *and* a model argument (`simulate --draws posterior
//! --fit`, `simulate --draws <file> --fit <run>`) replays those parameters
//! through whatever model file was named, so a model edited since the fit would
//! otherwise run the fit's draws through different dynamics and exit 0.
//!
//! There is one comparison, [`compare_to_fitted`], and every such check uses
//! it: `camdl predict --model` (may the fit be reported through the given
//! file's quantities?) and the replaying `simulate` paths. It is `ir::model_hash::model_ir_hash` — the
//! presentation-normalized model identity, which excludes `quantities` and
//! `contrasts` (reporting-only, `Model::hash_into`), the model's file-stem
//! name, and the gradient maps (model hash SV=2). That is exactly the question a
//! replay must answer yes to: the same model apart from what it reports.
//!
//! A fit archives its model with the observation anchors resolved
//! (`simulate { to = last_obs + … }` written as a number), so a model compared
//! with a stored fit is first resolved the same way against the fit's own data
//! ([`as_fitted`]); comparing the unresolved source with the resolved archive
//! would refuse the fit's own source.

use std::path::{Path, PathBuf};

use ir::model_hash::ContentHash;

/// How a given model relates to the model a fit ran on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAgreement {
    /// The same model, reporting layer included.
    Same,
    /// The same model apart from its `quantities {}` / `contrasts {}`: a run
    /// replays the fitted dynamics and reports through the given model's
    /// quantities and contrasts.
    ReportingDiffers,
    /// A different model: the dynamics, observations, parameters or
    /// simulation settings changed. Both identities, for the message.
    Differs { fitted: ContentHash, given: ContentHash },
}

/// Compare `given` against the fit's model `fitted` — the one comparison
/// (module doc).
pub fn compare_to_fitted(fitted: &ir::Model, given: &ir::Model) -> ModelAgreement {
    let fitted_h = ir::model_hash::model_ir_hash(fitted);
    let given_h = ir::model_hash::model_ir_hash(given);
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
/// and says whose quantities and contrasts it uses.
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
    // `fit run` binds the model's external inputs too, so a model that still
    // declares any is bound here from the fit's `[data.tables]` (in a stored
    // fit, the copies in its folder). One that declares none — the archived IR
    // itself among them — is left as it is.
    if crate::external_inputs::declares_any(&m) {
        crate::external_inputs::bind(&mut m, &problem.external_bindings())
            .map_err(|e| e.to_string())?;
    }
    let resolved = crate::obs_anchor::resolve_for_fit(&mut m, problem)?;
    Ok((m, resolved))
}

/// Where `given`'s resolved horizons differ from the fitted model's — the
/// model's `simulate { to }` and each scenario's, by name — said for a message;
/// `None` when every horizon agrees.
fn horizons_moved(fitted: &ir::Model, given: &ir::Model) -> Option<String> {
    let mut out = Vec::new();
    if fitted.simulation.t_start.to_bits() != given.simulation.t_start.to_bits() {
        out.push(format!(
            "the model starts at t = {} where the fitted model started at t = {}",
            given.simulation.t_start, fitted.simulation.t_start
        ));
    }
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
#[expect(clippy::disallowed_methods, reason = "compares the model file a replay was given with the copy of the stored problem's model file that `OpenedFit::model_source` read through the store, only to word the refusal of a replay on another model; nothing stored is computed from it")]
fn same_bytes(a: &Path, b: &Path) -> Option<bool> {
    Some(std::fs::read(a).ok()? == std::fs::read(b).ok()?)
}

/// The model path a stored run's fit file named, resolved against where that
/// file was: the run's `run.json` records the fit file as typed, and the run
/// keeps that file, whose `[model] camdl` is relative to it. `None` for a
/// problem handle, which names no run.
fn recorded_model_source(opened: &crate::fit_store::OpenedFit) -> Option<String> {
    let run = opened.fit.run.as_ref()?;
    let config = PathBuf::from(run.given().pointer("/config/path")?.as_str()?);
    let mut text = String::new();
    std::io::Read::read_to_string(&mut run.open_provenance(crate::fit_store::FIT_TOML_FILE).ok()?, &mut text).ok()?;
    let doc: toml::Table = text.parse().ok()?;
    let camdl = doc.get("model")?.get("camdl")?.as_str()?;
    let path = config.parent().map(|d| d.join(camdl)).unwrap_or_else(|| camdl.into());
    Some(path.to_string_lossy().into_owned())
}

/// Refuse to replay a STORED fit's parameters through `given` unless it is the
/// model the fit ran on, read from the fit's archived `model.ir.json`.
///
/// `fit` is the handle as the user typed it; `opened` the stored fit, read
/// out of the store; `problem` the fit's own config — its `[model]` source
/// path (absolute, resolved at load) and the `[data]` its observation anchors
/// resolve against. `given` is the command-line model as loaded, anchors
/// unresolved; it goes through the fit's anchor substitution ([`as_fitted`])
/// before the comparison. Returns the notice to print when only the reporting
/// layer differs, `None` when the models are the same.
pub fn check_stored_fit(
    fit: &str,
    opened: &crate::fit_store::OpenedFit,
    problem: &crate::fit::config_v2::Problem,
    given: &ir::Model,
    given_path: &str,
) -> Result<Option<String>, String> {
    let segment = opened.segment.as_path();
    let archive: PathBuf = segment.join("model.ir.json");
    if !archive.is_file() {
        return Err(format!(
            "--fit {fit}: this fit has no archived model ({} is missing), so there is no \
             record of the model its parameters were estimated with to check {given_path} \
             against. Refit to replay it.",
            archive.display()
        ));
    }
    let (stored, _) = crate::util::load_ir(&archive.to_string_lossy())?;
    // The stored compile keeps its anchored horizons unresolved; they are
    // resolved against the fit's data, as the given model's are below.
    let (fitted, _) = as_fitted(&stored, problem)
        .map_err(|e| format!("--fit {fit}: resolving the stored model's anchors against the fit's [data] failed: {e}"))?;
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
            // The model file the stored problem keeps, in the store: the
            // source (or IR) the fit was posed on.
            let source_copy = opened.model_source().unwrap_or_else(|_| segment.join("model.camdl"));
            // The model path the fit was given, as typed (the run records its
            // fit file and keeps that file); else the stored copy.
            let recorded_source_shown = recorded_model_source(opened)
                .unwrap_or_else(|| crate::fit::report::display_path(&source_copy));
            let recorded_source = recorded_source_shown.as_str();
            let stem = format!(
                "--fit {fit}: {given_path} is not the model this fit ran on (fitted model {}, \
                 {given_path} {})",
                short(&fitted_h),
                short(&given_h),
            );
            let fix = format!(
                "Fix: run with the fitted model, {} (the model file the fit stored), or refit \
                 {given_path}.",
                crate::fit::report::display_path(&archive)
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
                        crate::fit::report::display_path(&source_copy),
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
                        crate::fit::report::display_path(&source_copy)
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
                    crate::fit::report::display_path(&source_copy)
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
/// `[model] camdl` path, compiled under `compile`, the settings `given` was
/// compiled under, so the two differ only if the models do.
pub(crate) fn check_fit_config(
    fit_toml: &str,
    config_model: &str,
    given: &ir::Model,
    given_path: &str,
    compile: crate::util::CompileSettings,
) -> Result<Option<String>, String> {
    let (ir_path, _tmp) = crate::util::resolve_ir_path(config_model, false, compile)
        .map_err(|e| format!("--fit {fit_toml}: compiling its [model] {config_model}: {e}"))?;
    let (named, _) = crate::util::load_ir(&ir_path)?;
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
// simulator is the other half — the forward backend, its integration step,
// (ODE) the integrator, and the evaluator settings of its `[config]`. A replay
// of the parameters (`simulate --draws posterior|prior|<file> --fit`) runs on
// that simulator by default, as it runs on the fitted model, and an explicit
// flag that differs runs with a notice (`crate::simulator::Simulator::resolve`).

use crate::args::types::ForwardBackend;
use crate::simulator::StoredSimulator;

/// The two `[config]` settings a fit ran under that change what its model
/// evaluates to: `allow_degenerate_rates` (a collapsed rate is 0 rather than
/// an error) and `allow_forcing_extrapolation` (a forcing holds its end value
/// outside its range rather than refusing the run). A replay of the fit runs
/// under them (gh#1235).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FittedSettings {
    pub allow_degenerate_rates: bool,
    pub allow_forcing_extrapolation: bool,
}

impl FittedSettings {
    /// The settings of a fit config's `[config]`.
    pub fn of(config: &crate::fit::config_v2::FitBackendConfig) -> Self {
        FittedSettings {
            allow_degenerate_rates: config.allow_degenerate_rates,
            allow_forcing_extrapolation: config.allow_forcing_extrapolation,
        }
    }

    /// The settings a replay of the fit runs under, given the replaying
    /// command's flags: each is on when the fit ran under it or the flag is
    /// given, the fold `pfilter` and `profile` apply to a `--fit`.
    /// A flag can only turn a setting on, so a replay never drops the fit's.
    pub fn with_flags(self, allow_degenerate_rates: bool, allow_forcing_extrapolation: bool) -> Self {
        FittedSettings {
            allow_degenerate_rates: self.allow_degenerate_rates || allow_degenerate_rates,
            allow_forcing_extrapolation: self.allow_forcing_extrapolation || allow_forcing_extrapolation,
        }
    }

    /// The note a command says when a flag turned on, in `folded`, a setting
    /// this fit ran without: `{fit_name} was fitted without {flags}; {doing}
    /// under it, as given`. `None` when the flags added nothing.
    pub fn turned_on_note(self, folded: FittedSettings, fit_name: &str, doing: &str) -> Option<String> {
        let turned_on: Vec<&str> = [
            (folded.allow_degenerate_rates && !self.allow_degenerate_rates, "--allow-degenerate-rates"),
            (
                folded.allow_forcing_extrapolation && !self.allow_forcing_extrapolation,
                "--allow-forcing-extrapolation",
            ),
        ]
        .iter()
        .filter_map(|(on, flag)| on.then_some(*flag))
        .collect();
        (!turned_on.is_empty())
            .then(|| format!("note: {fit_name} was fitted without {}; {doing} under it, as given", turned_on.join(" or ")))
    }
}

/// The simulator of a stored fit, named `name` in messages.
///
/// - **backend**: the named run's own record (its method's `backend` in
///   `run.json`, as `camdl predict` and `compare` read it); for a problem-folder
///   handle, the config's `[method]`.
/// - **dt**: `fit.meta.json`'s recorded step; for a fit recorded before the
///   step was, the dt rule over the archived model's `simulate { dt }` and the
///   archived config's `[config] dt` — the two inputs `fit run` resolved it
///   from.
/// - **integrator**: `fit.meta.json`'s, else the archived model's.
/// - **settings**: the stored config's `[config]` (`fit.resolved.toml`, the
///   configuration the fit ran under, with a flag it was given written in),
///   always `Some`.
/// - **binomial**: `None`; a replay draws with the default sampler.
///
/// Read by `simulate --draws … --fit`, `fit summary --point` and `compare`.
/// `camdl predict` builds its own from the run's recorded backend and sampler.
/// `model` is the model the fit ran on, as the caller loaded it
/// ([`stored_fit_model`]).
pub(crate) fn stored_fit_simulator(
    resolved: &crate::fit_store::OpenedFit,
    model: &ir::Model,
    name: &str,
) -> Result<StoredSimulator, String> {
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
    let settings = FittedSettings::of(&resolved.config.problem.config);
    Ok(StoredSimulator { name: name.to_string(), backend, dt, integrator, binomial: None, settings: Some(settings) })
}

/// The model a stored fit ran on: its archive, else (a fit that archived
/// none) the source its config names, compiled under `compile`, as `compare`
/// reads it.
pub(crate) fn stored_fit_model(
    resolved: &crate::fit_store::OpenedFit,
    compile: crate::util::CompileSettings,
) -> Result<ir::Model, String> {
    let model_src = resolved.model_ir.to_string_lossy().into_owned();
    crate::util::load_model(&model_src, compile)
        .map(|(m, _)| m)
        .map_err(|e| format!("reading the model the fit ran on, {model_src}: {e}"))
}

/// The simulator a fit CONFIG describes, for a replay of its priors or
/// `[fixed]` beside `model` (checked to be the model its `[model]` names,
/// [`check_fit_config`]): its `[method]` backend, the dt rule over the model's
/// step and the config's written `[config] dt` (a disagreeing one is refused,
/// as `fit run` refuses it), the model's integrator, and the config's
/// `[config]` settings. Named by the file's path.
#[expect(clippy::disallowed_methods, reason = "reads the fit config that `simulate --fit` names, declared `#[input(identity)]` on `SimulateArgs`, for the backend and step a replay runs on")]
pub(crate) fn fit_config_simulator(fit_toml: &str, model: &ir::Model) -> Result<StoredSimulator, String> {
    let config = crate::fit::config_v2::FitConfig::load(fit_toml)?;
    let raw = std::fs::read_to_string(fit_toml).map_err(|e| format!("cannot read {fit_toml}: {e}"))?;
    let written = crate::fit::config_v2::table_declares_key(&raw, "config", "dt")
        .then_some(config.problem.config.dt);
    let dt = crate::dt_rule::effective_dt(model.simulation.dt, written)
        .map_err(|m| format!("{fit_toml}: {}", m.message("[config] dt")))?;
    Ok(StoredSimulator {
        name: fit_toml.to_string(),
        backend: config.inference.method.as_ref().map(|m| ForwardBackend::from(m.algorithm.backend())),
        dt,
        integrator: model.simulation.integrator.clone(),
        binomial: None,
        settings: Some(FittedSettings::of(&config.problem.config)),
    })
}
