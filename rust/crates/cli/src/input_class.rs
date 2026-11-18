//! The class of every input of a converted command, as a table per command,
//! checked against the arguments clap accepts.
//!
//! An input is a command-line field (or, in [`crate::env`], an environment
//! variable). Its class says what it may do to a stored result, and how that is
//! checked (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "Classification"):
//!
//! | Class          | Meaning                                                      | Verification                                             |
//! | -------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
//! | `keyed(level)` | can change the result; part of the resolved configuration    | differential: changed alone, that level's hash changes   |
//! | `execution`    | changes how the result is computed, not what (`--parallel`)  | differential: run id and stored bytes stay the same      |
//! | `provenance`   | how the run was asked for (`--label`)                        | structural, plus the leaf is unaffected                  |
//! | `presentation` | how output is written outside the store (`-o`)               | structural, plus the leaf is unaffected or none written  |
//! | `child`        | asks for a derived child of the leaf (`--event-log`, `--obs`) | the leaf's run id and manifest are unchanged, the child is at its address |
//! | `expands`      | defines the set of runs (`--seeds`, `--sweep`)               | differential: cells common to both keep their run ids    |
//! | `removed`      | refuses an old spelling with a migration message             | the flag fails with the message                          |
//!
//! A class belongs to a (command, input) pair, not to a flag: a group that
//! several commands flatten (`ForcingExtrapolationArg`, the global options) is
//! classified in each command's table. [`SIMULATE`] is `camdl simulate`'s and
//! [`FIT_RUN`] `camdl fit run`'s. `every_simulate_flag_is_classified` and
//! `every_fit_run_flag_is_classified` (below) check them against every argument
//! clap accepts, both ways, and every input has a case in
//! `rust/crates/cli/tests/simulate_input_classes.rs`
//! (`every_simulate_input_has_a_case`) or
//! `rust/crates/cli/tests/fit_run_input_classes.rs`
//! (`every_fit_run_input_has_a_case`).

/// The identity level a keyed input enters at. The first five are the levels of
/// a simulation leaf ([`runid::inputs::TrajectoryInput`]); `Obs` is the address
/// of the synthetic-observation subtree under a leaf
/// ([`crate::cas_sink::obs_subtree_hash`]), which a trajectory's own run id does
/// not cover. `Fit` and `Method` are the levels of a fit method leaf
/// ([`runid::inputs::FitMethodInput`]), whose third is `Seed`: the fit level
/// holds the model, the data, the problem half of the configuration and the
/// engine, so an environment variable keyed at a simulation's `Model` level
/// enters a fit at its `Fit` level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(not(test), allow(dead_code))] // the levels other than `Model` occur only in the command tables
pub enum Level {
    Model,
    Config,
    Params,
    Scenario,
    Seed,
    Obs,
    Fit,
    Method,
}

/// What an input may do to a stored result. See the module table.
///
/// `Provenance` and `Expands` occur only in the command tables, which tests
/// read; the environment table ([`crate::env::VARS`]), read at run time, has
/// neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(not(test), allow(dead_code))]
pub enum InputClass {
    Keyed(Level),
    Execution,
    Provenance,
    Presentation,
    /// Asks for a child of the leaf: an artifact derived from the leaf's
    /// result, stored under the leaf at its own address and declared in the
    /// leaf's `run.json` `children`, never one of the leaf's own files. So
    /// whether an invocation asked for it never changes what the leaf's run
    /// id holds. The observation subtree (`obs/<hash8>-<obs_seed>/`) and the
    /// event log (`event_log/`) are the two.
    Child,
    Expands,
    /// An old spelling, kept only so that it is refused with a message naming
    /// its replacement.
    Removed,
}

#[cfg(test)]
use InputClass::{Child, Execution, Expands, Keyed, Presentation, Provenance, Removed};
#[cfg(test)]
use Level::{Config, Fit, Method, Model, Obs, Params, Scenario, Seed};

/// `camdl simulate`'s inputs, by clap argument id (the field name), with their
/// classes: `SimulateArgs`, the groups it flattens and the global options.
#[cfg(test)]
pub const SIMULATE: &[(&str, InputClass)] = &[
    // SimulateArgs
    ("model", Keyed(Model)),
    ("to", Keyed(Config)),
    ("init_state", Keyed(Config)),
    ("allow_degenerate_rates", Keyed(Config)),
    ("param_vec", Keyed(Params)),
    ("scenarios", Expands),
    ("enable", Keyed(Scenario)),
    ("disable", Keyed(Scenario)),
    ("seed", Keyed(Seed)),
    ("seeds", Expands),
    ("replicates", Expands),
    ("draws", Expands),
    ("sweep", Expands),
    ("sweep_file", Expands),
    ("fit", Keyed(Params)),
    ("n_draws", Expands),
    ("chains", Expands),
    ("exclude_chains", Expands),
    ("draws_out", Presentation),
    ("output", Presentation),
    ("stdout", Presentation),
    ("emit_every", Keyed(Obs)),
    ("obs", Child),
    ("obs_dir", Child),
    ("obs_only", Child),
    ("obs_only_dir", Child),
    ("design_from", Presentation),
    ("projected_out", Presentation),
    ("dates", Presentation),
    ("quantities_out", Presentation),
    ("quantities", Presentation),
    ("dry_run", Presentation),
    ("output_dir", Execution),
    ("parallel", Execution),
    ("force", Execution),
    ("label", Provenance),
    ("event_log", Child),
    ("format", Presentation),
    ("tsv", Presentation),
    ("reactive_log", Presentation),
    // ForcingExtrapolationArg
    ("allow_forcing_extrapolation", Keyed(Config)),
    // ModelOverrides
    ("params", Keyed(Params)),
    ("param", Keyed(Params)),
    ("table", Keyed(Params)),
    // SimBackend
    ("backend", Keyed(Config)),
    ("dt", Keyed(Config)),
    ("integrator", Keyed(Model)),
    // OutputView
    ("every", Keyed(Model)),
    ("no_flows", Keyed(Config)),
    ("columns", Keyed(Config)),
    // GlobalArgs
    ("verbosity", Presentation),
    ("progress", Presentation),
    ("no_progress", Presentation),
    ("no_ir_cache", Execution),
    ("no_licm", Keyed(Model)),
];

/// `camdl fit run`'s inputs, by clap argument id (the field name), with their
/// classes: `FitRunArgs`, the group it flattens and the global options.
///
/// The fit configuration file is `keyed(fit)`: its problem half is the fit
/// level, though its `[method]` table keys the method level. Every flag that
/// writes into `[method]` (`--starts`, the sampler and dt-check flags,
/// `--binomial`, the pfilter record flags) is `keyed(method)`;
/// `--allow-forcing-extrapolation`, which writes into `[config]`, and the
/// global `--no-licm`, which changes the IR, are `keyed(fit)`. `--resume` keys
/// the method level through its target length and its dep on the base.
/// `--emit-every` keys a synthetic fit through the data it generates, in the
/// fit level. `--allow-nonconverged-source` lifts a refusal and changes
/// nothing a run that starts computes, so it is `execution`.
#[cfg(test)]
pub const FIT_RUN: &[(&str, InputClass)] = &[
    // FitRunArgs
    ("config", Keyed(Fit)),
    ("starts", Keyed(Method)),
    ("starts_chains", Keyed(Method)),
    ("allow_nonconverged_source", Execution),
    ("parallel", Execution),
    ("seed", Keyed(Seed)),
    ("force", Execution),
    ("resume", Keyed(Method)),
    ("sweep", Expands),
    ("emit_every", Keyed(Fit)),
    ("decibans_thresh", Keyed(Method)),
    ("_removed_stage", Removed),
    ("_removed_init", Removed),
    ("_removed_posterior", Removed),
    ("_removed_mle", Removed),
    ("_removed_params", Removed),
    ("_removed_survey_path", Removed),
    ("_removed_survey_top_k", Removed),
    ("_removed_allow_nonconverged_scout", Removed),
    ("_removed_init_method", Removed),
    ("_removed_starts_from", Removed),
    ("_removed_tempering", Removed),
    ("_removed_rho", Removed),
    ("label", Provenance),
    ("no_dt_check", Keyed(Method)),
    ("dt_check_strict", Keyed(Method)),
    ("binomial", Keyed(Method)),
    ("dt_check_halvings", Keyed(Method)),
    ("cooling_target_iters", Keyed(Method)),
    ("max_tree_depth", Keyed(Method)),
    ("trajectory_warmup", Keyed(Method)),
    ("csmc_sweeps_per_nuts", Keyed(Method)),
    ("n_trajectories", Keyed(Method)),
    ("diagonal_mass", Keyed(Method)),
    ("no_nuts", Keyed(Method)),
    ("no_ancestor_sampling", Keyed(Method)),
    ("no_adapt", Keyed(Method)),
    ("adapt_start", Keyed(Method)),
    ("record_ancestry", Keyed(Method)),
    ("record_prequential", Keyed(Method)),
    // ForcingExtrapolationArg
    ("allow_forcing_extrapolation", Keyed(Fit)),
    // GlobalArgs
    ("verbosity", Presentation),
    ("progress", Presentation),
    ("no_progress", Presentation),
    ("no_ir_cache", Execution),
    ("no_licm", Keyed(Fit)),
];

/// `camdl mre simulate`'s own input. It flattens `SimulateArgs` and writes a
/// bundle that reproduces that simulate run, so every other input keeps the
/// class it has in [`SIMULATE`].
#[cfg(test)]
pub const MRE_SIMULATE_OWN: &[(&str, InputClass)] = &[("bundle", Presentation)];

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::collections::BTreeSet;

    /// Every argument clap accepts for `path` (a subcommand path from the root),
    /// by id, without clap's own `help` / `version`.
    fn clap_args(path: &[&str]) -> BTreeSet<String> {
        let mut cmd = crate::Cli::command();
        cmd.build();
        let mut cur = &cmd;
        for name in path {
            cur = cur
                .find_subcommand(name)
                .unwrap_or_else(|| panic!("no subcommand `{name}`"));
        }
        cur.get_arguments()
            .map(|a| a.get_id().to_string())
            .filter(|id| id != "help" && id != "version")
            .collect()
    }

    fn assert_covers<'a>(what: &str, clap: BTreeSet<String>, table: impl IntoIterator<Item = &'a (&'a str, InputClass)>) {
        let mut seen = BTreeSet::new();
        for (field, _) in table {
            assert!(seen.insert(field.to_string()), "{what}: `{field}` is classified twice");
        }
        let missing: Vec<&String> = clap.difference(&seen).collect();
        let stale: Vec<&String> = seen.difference(&clap).collect();
        assert!(
            missing.is_empty() && stale.is_empty(),
            "{what}: the class table and the arguments clap accepts disagree.\n  \
             accepted by clap, not classified: {missing:?}\n  \
             classified, not accepted by clap: {stale:?}"
        );
    }

    #[test]
    fn every_simulate_flag_is_classified() {
        assert_covers("camdl simulate", clap_args(&["simulate"]), SIMULATE);
    }

    #[test]
    fn every_fit_run_flag_is_classified() {
        assert_covers("camdl fit run", clap_args(&["fit", "run"]), FIT_RUN);
    }

    #[test]
    fn every_mre_simulate_flag_is_classified() {
        assert_covers("camdl mre simulate", clap_args(&["mre", "simulate"]), SIMULATE.iter().chain(MRE_SIMULATE_OWN));
    }

    /// Environment inputs with no `simulate` case, each with the reason.
    const NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("output_dir", "not read by simulate: its store root is `--output-dir`, whose clap `env` reads CAMDL_OUTPUT_DIR"),
        ("unlabeled_threshold", "not read by simulate: `fit table` and `camdl list` read it"),
        ("ci", "not read by simulate: `fit summary` reads it"),
    ];

    /// Environment inputs with no `fit run` case, each with the reason.
    const FIT_RUN_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by fit run: `fit table` and `camdl list` read it"),
        ("ci", "not read by fit run: `fit summary` reads it"),
    ];

    fn case_name(class: InputClass, field: &str, env: bool) -> String {
        let class = match class {
            InputClass::Keyed(_) => "keyed",
            InputClass::Execution => "execution",
            InputClass::Expands => "expands",
            InputClass::Presentation => "presentation",
            InputClass::Provenance => "provenance",
            InputClass::Child => "child",
            InputClass::Removed => "removed",
        };
        // A removed spelling's field is `_removed_<old name>`; its case is
        // `removed_<old name>`.
        let field = field.strip_prefix("_removed_").unwrap_or(field);
        format!("{class}_{}{field}", if env { "env_" } else { "" })
    }

    /// The test functions of a case file whose names begin with a class.
    fn cases(file: &str) -> BTreeSet<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        text.lines()
            .filter_map(|l| l.trim_start().strip_prefix("fn "))
            .filter_map(|l| l.split('(').next())
            .filter(|n| {
                ["keyed_", "execution_", "expands_", "presentation_", "provenance_", "child_", "removed_"]
                    .iter()
                    .any(|p| n.starts_with(p))
            })
            .map(str::to_string)
            .collect()
    }

    /// The cases `table` and the environment table (less `no_case`) call for,
    /// against those `file` holds.
    fn assert_every_input_has_a_case(what: &str, file: &str, table: &[(&str, InputClass)], no_case: &[(&str, &str)]) {
        let mut wanted = BTreeSet::new();
        for (field, class) in table {
            wanted.insert(case_name(*class, field, false));
        }
        for v in crate::env::VARS {
            if no_case.iter().all(|(f, _)| *f != v.field) {
                wanted.insert(case_name(v.class, v.field, true));
            }
        }
        let cases = cases(file);
        let missing: Vec<&String> = wanted.difference(&cases).collect();
        let stray: Vec<&String> = cases.difference(&wanted).collect();
        assert!(
            missing.is_empty() && stray.is_empty(),
            "the cases and {what}'s class table disagree.\n  \
             inputs without a case: {missing:?}\n  \
             cases naming no such input (or the wrong class): {stray:?}"
        );
    }

    /// Every input of simulate — on its command line or in its environment —
    /// has a case in `tests/simulate_input_classes.rs` named
    /// `<class>_<input>` (or `<class>_env_<field>`), and every case there names
    /// such an input with its class. A `keyed`, `execution` or `expands` case
    /// runs camdl twice and compares; a `presentation` or `provenance` case
    /// checks, on a fixture where the input is live, that the leaf is
    /// unaffected or that none is written; a `child` case, that the leaf's
    /// run id and manifest are unchanged and the child is at its address.
    #[test]
    fn every_simulate_input_has_a_case() {
        assert_every_input_has_a_case("simulate", "simulate_input_classes.rs", SIMULATE, NO_CASE);
    }

    /// The same for `fit run`, in `tests/fit_run_input_classes.rs`, where a
    /// `removed` case checks that the old spelling is refused with its
    /// message.
    #[test]
    fn every_fit_run_input_has_a_case() {
        assert_every_input_has_a_case("fit run", "fit_run_input_classes.rs", FIT_RUN, FIT_RUN_NO_CASE);
    }
}
