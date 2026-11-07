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

/// Refuse to replay a STORED fit's parameters through `given` unless it is the
/// model the fit ran on, read from the fit's archived `model.ir.json`.
///
/// `fit` is the handle as the user typed it; `segment` the fit's problem
/// folder; `recorded_source` the model source path the fit's config records
/// (absolute, resolved at load), when it has one. Returns the notice to print
/// when only the reporting layer differs, `None` when the models are the same.
pub fn check_stored_fit(
    fit: &str,
    segment: &Path,
    recorded_source: Option<&str>,
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
    match compare_to_fitted(&fitted, given) {
        ModelAgreement::Same => Ok(None),
        ModelAgreement::ReportingDiffers => Ok(Some(reporting_notice(fit, "ran on", given_path))),
        ModelAgreement::Differs { fitted, given: given_h } => {
            let source_copy = segment.join("model.camdl.original");
            let source_hint = match recorded_source {
                Some(src) if same_file(src, given_path) && source_copy.is_file() => format!(
                    " {given_path} is the source the fit was run from, edited since; the \
                     source as fitted is archived at {}.",
                    source_copy.display()
                ),
                Some(src) if same_file(src, given_path) => format!(
                    " {given_path} is the source the fit was run from, edited since."
                ),
                Some(src) => format!(
                    " The fit was run from {src} (use it only if it still compiles to \
                     model {}).",
                    short(&fitted)
                ),
                None => String::new(),
            };
            Err(format!(
                "--fit {fit}: {given_path} is not the model this fit ran on (fitted model {}, \
                 {given_path} {}), so its parameters would be replayed through different \
                 dynamics. Quantities and contrasts are excluded from this comparison: the \
                 difference is in the dynamics, observations, parameters or simulation \
                 settings.\n  \
                 Fix: run with the fitted model, {} (the fit's archived IR), or refit \
                 {given_path}.{source_hint}",
                short(&fitted),
                short(&given_h),
                archive.display(),
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
