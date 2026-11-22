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
//! | `expands`      | defines the set of runs (`--seeds`, `--sweep`)               | differential: cells common to both keep their run ids    |
//! | `removed`      | refuses an old spelling with a migration message             | the flag fails with the message                          |
//!
//! A class belongs to a (command, input) pair, not to a flag: a group that
//! several commands flatten (`ForcingExtrapolationArg`, the global options) is
//! classified in each command's table: [`FIT_RUN`], [`PROFILE`].
//! `camdl pfilter` and `camdl simulate` declare their inputs' classes in their
//! argument types instead (`#[derive(store::Inputs)]`,
//! `crate::args::PfilterInvocation`, `crate::args::SimulateInvocation`).
//! `every_<command>_flag_is_classified` (below) checks each against every
//! argument clap accepts, both ways, and `every_<command>_input_has_a_case`
//! that every input has a case in `rust/crates/cli/tests/<command>_input_classes.rs`.

/// The identity level a keyed input enters at. `Model` is the compiled model,
/// which the environment table keys. `Fit` and `Method` are the levels of a
/// fit method leaf ([`runid::inputs::FitDigest`],
/// [`runid::inputs::MethodLevel`]), whose third is `Seed`: the fit level holds
/// the model, the data, the problem half of the configuration and the engine,
/// so an environment variable keyed at `Model` enters a fit at its `Fit`
/// level. A profile point's are `profile`, `point`,
/// `method`, `seed` and `start` ([`crate::profile_cas::resolve_profile_point`]);
/// its inputs key `Profile`, `Method` or `Seed`. The profile level (the base)
/// holds the model, the data, the problem, the sweep grid and the
/// engine, so an environment variable keyed at `Model` enters a profile at
/// its `Profile` level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(not(test), allow(dead_code))] // the levels other than `Model` occur only in the command tables
pub enum Level {
    Model,
    Seed,
    Fit,
    Method,
    Profile,
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
    Expands,
    /// An old spelling, kept only so that it is refused with a message naming
    /// its replacement.
    Removed,
}

#[cfg(test)]
use InputClass::{Execution, Expands, Keyed, Presentation, Provenance, Removed};
#[cfg(test)]
use Level::{Fit, Method, Profile, Seed};

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

/// `camdl profile`'s inputs, by clap argument id (the field name), with their
/// classes: `ProfileArgs`, the groups it flattens and the global options.
///
/// A profile point's levels are `profile` (the base: the resolved model, the
/// data, the problem, the sweep grid, the engine), `point` (the cell's focal
/// values), `method` (how a cell is fitted), `seed` and `start`. `--sweep` is
/// keyed, not `expands`: the grid is folded into every point's base, so a
/// different grid is a different profile and no cell is shared with it.
/// `--seeds` expands: each seed is its own level. Each algorithm runs on one
/// backend, so `--backend` decides only whether the run is refused, as
/// `--time-format` does; `--allow-nonconverged-source` lifts a refusal. All
/// three are `execution`.
#[cfg(test)]
pub const PROFILE: &[(&str, InputClass)] = &[
    // ProfileArgs
    ("model", Keyed(Profile)),
    ("data", Keyed(Profile)),
    ("fit", Keyed(Profile)),
    ("sweep", Keyed(Profile)),
    ("particles", Keyed(Method)),
    ("iterations", Keyed(Method)),
    ("n_starts", Keyed(Method)),
    ("starts", Keyed(Method)),
    ("starts_chains", Keyed(Method)),
    ("allow_nonconverged_source", Execution),
    ("_removed_init", Removed),
    ("_removed_posterior", Removed),
    ("_removed_mle", Removed),
    ("_removed_params", Removed),
    ("_removed_pmmh_steps", Removed),
    ("_removed_pmmh_particles", Removed),
    ("_removed_pmmh_rho", Removed),
    ("cooling", Keyed(Method)),
    ("cooling_target_iters", Keyed(Method)),
    ("rw_sd", Keyed(Method)),
    ("output", Presentation),
    ("format", Presentation),
    ("seeds", Expands),
    ("label", Provenance),
    ("algorithm", Keyed(Method)),
    ("backend", Execution),
    // InferenceModelOverrides
    ("fixed_cli", Keyed(Profile)),
    ("fixed_files", Keyed(Profile)),
    ("table", Keyed(Profile)),
    ("_removed_param", Removed),
    // ScenarioArgs
    ("scenario", Keyed(Profile)),
    ("enable", Keyed(Profile)),
    ("disable", Keyed(Profile)),
    // InferenceCore
    ("dt", Keyed(Method)),
    ("seed", Keyed(Seed)),
    ("parallel", Execution),
    ("allow_degenerate_rates", Keyed(Profile)),
    ("time_format", Execution),
    ("pf_max_substeps", Keyed(Method)),
    ("binomial", Keyed(Method)),
    // ForcingExtrapolationArg
    ("allow_forcing_extrapolation", Keyed(Profile)),
    // StreamSelection
    ("obs", Keyed(Profile)),
    // GlobalArgs
    ("verbosity", Presentation),
    ("progress", Presentation),
    ("no_progress", Presentation),
    ("no_ir_cache", Execution),
    ("no_licm", Keyed(Profile)),
];

/// `camdl mre simulate`'s own input. It flattens `SimulateArgs` and writes a
/// bundle that reproduces that simulate run, so every other input keeps the
/// class `SimulateArgs` declares.
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
    fn every_fit_run_flag_is_classified() {
        assert_covers("camdl fit run", clap_args(&["fit", "run"]), FIT_RUN);
    }

    #[test]
    fn every_profile_flag_is_classified() {
        assert_covers("camdl profile", clap_args(&["profile"]), PROFILE);
    }

    /// `mre simulate` takes every input of `simulate` (declared on
    /// `SimulateArgs`) and its own.
    #[test]
    fn every_mre_simulate_flag_is_classified() {
        use store::Inputs;
        let simulate: BTreeSet<String> = crate::args::SimulateArgs::fields()
            .iter()
            .map(|f| f.name.rsplit('.').next().unwrap().to_string())
            .collect();
        let mut accepted = clap_args(&["mre", "simulate"]);
        for (own, _) in MRE_SIMULATE_OWN {
            assert!(accepted.remove(*own), "`{own}` is not an argument of `camdl mre simulate`");
        }
        let globals: BTreeSet<String> = ["verbosity", "progress", "no_progress", "no_ir_cache", "no_licm"].map(String::from).into();
        let accepted: BTreeSet<String> = accepted.difference(&globals).cloned().collect();
        assert_eq!(accepted, simulate, "`mre simulate` takes simulate's arguments");
    }

    /// Environment inputs with no `fit run` case, each with the reason.
    const FIT_RUN_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by fit run: `fit table` and `camdl list` read it"),
        ("ci", "not read by fit run: `fit summary` reads it"),
        ("store", "not read by fit run: the commands on the run store read it"),
    ];

    /// Environment inputs with no `profile` case, each with the reason.
    const PROFILE_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by profile: `fit table` and `camdl list` read it"),
        ("ci", "not read by profile: `fit summary` reads it"),
        ("store", "not read by profile: the commands on the run store read it"),
    ];

    fn case_name(class: InputClass, field: &str, env: bool) -> String {
        let class = match class {
            InputClass::Keyed(_) => "keyed",
            InputClass::Execution => "execution",
            InputClass::Expands => "expands",
            InputClass::Presentation => "presentation",
            InputClass::Provenance => "provenance",
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
                ["keyed_", "execution_", "expands_", "presentation_", "provenance_", "removed_"]
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

    /// Every input of `fit run`, on its command line or in its environment,
    /// has a case in `tests/fit_run_input_classes.rs`, where a
    /// `removed` case checks that the old spelling is refused with its
    /// message.
    #[test]
    fn every_fit_run_input_has_a_case() {
        assert_every_input_has_a_case("fit run", "fit_run_input_classes.rs", FIT_RUN, FIT_RUN_NO_CASE);
    }

    /// The same for `profile`, in `tests/profile_input_classes.rs`.
    #[test]
    fn every_profile_input_has_a_case() {
        assert_every_input_has_a_case("profile", "profile_input_classes.rs", PROFILE, PROFILE_NO_CASE);
    }
}
