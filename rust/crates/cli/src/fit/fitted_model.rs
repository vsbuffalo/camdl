//! Is the model a command was given the model a fit ran on?
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
