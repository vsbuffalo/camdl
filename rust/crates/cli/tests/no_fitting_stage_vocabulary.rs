//! The fitting sense of "stage" stays gone from the fit, run-identity and I/O
//! crates.
//!
//! `[stages]` became one `[method]` per fit.toml, and the code, the store, the
//! run.json `kind` and the run-identity types were renamed to say method. This
//! test fails if one of the old identifiers comes back in `cli`, `runid` or
//! `io` (source and tests), so a revert or a copy from an old branch cannot
//! reintroduce the vocabulary quietly.
//!
//! It matches whole identifiers from an explicit list, not the word "stage":
//! disease stages (Erlang sub-stages, a model's `stage` dimension), the atomic
//! stage-then-rename in the store, and the migration refusals that name what a
//! user wrote (`detect_legacy_stages`, `[stages]`, the hidden `--stage` flag)
//! must never trip it.

use std::path::{Path, PathBuf};

/// Snake-case identifiers of the removed vocabulary, matched exactly.
const SNAKE: &[&str] = &[
    "stage_dir", "stage_dirs", "stage_dir_str", "run_stage", "fit_stage", "stage_name",
    "nlopt_stage", "resolve_fit_stage", "fit_stage_levels", "stage_config_hash",
    "discover_stages", "abandon_stage", "read_stage_paths", "bayesian_stage_report",
    "if2_stage_report", "render_md_stage", "render_latex_stage", "stage_label", "stage_leaf",
    "cas_stage_leaf",
];

/// CamelCase suffixes that only ever named a fitting stage.
const CAMEL_SUFFIXES: &[&str] = &[
    "StageConfig", "StageLevel", "StageBlock", "StageFacts", "StageNode", "StageAxes",
    "StageResult", "StageView", "StageCtx", "StageInput", "StageOpts", "StageOverrides",
];

/// CamelCase identifiers matched exactly.
const CAMEL: &[&str] =
    &["ResolvedStage", "ResolvedFitStage", "ProfileStage", "FilterEssStage", "StageConvergence"];

/// `(path suffix, identifier)` pairs that may name an old identifier because
/// naming it is their job: the table that explains a record written under the
/// old run.json kind, the test that plants such a record, and the two
/// run-id pin ledgers, whose comments record which renamed types moved a pin.
const ALLOWED: &[(&str, &str)] = &[
    ("runid/src/record.rs", "fit_stage"),
    ("cli/tests/cas_integration.rs", "fit_stage"),
    ("cli/src/resolve/tests.rs", "StageConfig"),
    ("cli/src/resolve/tests.rs", "StageLevel"),
    ("cli/src/resolve/tests.rs", "FitStageInput"),
    ("cli/src/resolve/tests.rs", "ProfileStage"),
    ("runid/src/inputs/tests.rs", "StageConfig"),
];

fn is_removed(ident: &str) -> bool {
    SNAKE.contains(&ident)
        || CAMEL.contains(&ident)
        || ident.starts_with("FitStage")
        || CAMEL_SUFFIXES.iter().any(|s| ident.ends_with(s))
}

fn identifiers(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|t| !t.is_empty())
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_predicate_matches_the_old_names_and_nothing_disease_shaped() {
    for old in ["stage_dir", "run_stage", "fit_stage", "FitStageCtx", "FitStageInput",
                "PgasStageResult", "CliStageOverrides", "StageConfig", "ResolvedStage"] {
        assert!(is_removed(old), "{old} must be caught");
    }
    for fine in ["stage", "stages", "Stage", "staged", "stage_per_eval", "latent_stage",
                 "erlang_stage", "detect_legacy_stages", "_removed_stage", "stages_declared",
                 "stage_in_support", "rename_staged_into_place"] {
        assert!(!is_removed(fine), "{fine} is not the fitting sense");
    }
}

#[test]
fn no_fitting_sense_stage_identifier_in_cli_runid_or_io() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let this_file = Path::new(file!()).file_name().unwrap().to_owned();
    let mut files = Vec::new();
    for krate in ["cli", "runid", "io"] {
        for sub in ["src", "tests"] {
            rust_files(&crates.join(krate).join(sub), &mut files);
        }
    }
    assert!(files.len() > 100, "the scan found only {} files; is the path right?", files.len());

    let mut hits = Vec::new();
    for file in &files {
        if file.file_name() == Some(this_file.as_os_str()) {
            continue;
        }
        let shown = file.strip_prefix(&crates).unwrap_or(file).to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            for ident in identifiers(line).filter(|i| is_removed(i)) {
                if ALLOWED.iter().any(|(p, i)| shown.ends_with(p) && *i == ident) {
                    continue;
                }
                hits.push(format!("{shown}:{}: `{ident}`", n + 1));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "the fitting sense of \"stage\" is \"method\" (a fit.toml has one `[method]`); \
         rename these, or add an `ALLOWED` entry if the line must name the old \
         identifier:\n{}",
        hits.join("\n")
    );
}
