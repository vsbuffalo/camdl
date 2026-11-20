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
//! classified in each command's table: [`SIMULATE`], [`FIT_RUN`], [`PFILTER`],
//! [`PROFILE`], [`SURVEY`].
//! `every_<command>_flag_is_classified` (below) checks each against every
//! argument clap accepts, both ways, and `every_<command>_input_has_a_case`
//! that every input has a case in `rust/crates/cli/tests/<command>_input_classes.rs`.

/// The identity level a keyed input enters at. The first five are the levels of
/// a simulation leaf ([`crate::resolve::resolve_trajectory`]); `Obs` is the
/// address of the synthetic-observation subtree under a leaf
/// ([`crate::cas_sink::obs_subtree_hash`]), which a trajectory's own run id does
/// not cover. `Fit` and `Method` are the levels of a fit method leaf
/// ([`runid::inputs::FitDigest`], [`runid::inputs::MethodLevel`]), whose third
/// is `Seed`: the fit level holds the model, the data, the problem half of the
/// configuration and the engine, so an environment variable keyed at a
/// simulation's `Model` level enters a fit at its `Fit` level. A `pfilter` leaf's levels are `Model`,
/// `Config`, `Params` and `Seed`. A profile point's are `profile`, `point`,
/// `method`, `seed` and `start` ([`crate::profile_cas::resolve_profile_point`]);
/// its inputs key `Profile`, `Method` or `Seed`. The profile level (the base)
/// holds the model, the data, the problem, the sweep grid and the
/// engine, so an environment variable keyed at `Model` enters a profile at
/// its `Profile` level. A survey leaf's are `Model`, `Config`, `Box` (the
/// sampled bounds and the number of points) and `Seed`.
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
    Profile,
    Box,
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

/// Inputs by clap argument id (the field name), with their classes.
pub type Inventory = &'static [(&'static str, InputClass)];

/// A clap `Args` struct whose every argument carries a class, derived with
/// `#[derive(Inputs)]` (`rust/crates/input-class-derive`).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is flattened into a classified command, but its fields carry no input classes",
    label = "`{Self}` does not derive `Inputs`",
    note = "add `#[derive(Inputs)]` to `{Self}` and an `#[input(..)]` class to each of its fields"
)]
pub trait InputFields {
    /// Every argument of the struct, its own and its flattened groups', with
    /// its class: the class the group declares, unless the flatten site
    /// overrides it.
    const FIELDS: Inventory;
}

/// The constant functions `#[derive(Inputs)]` expands to. Each runs at
/// compile time; a failed check is a compile error.
pub mod derive_support {
    use super::{InputClass, Inventory, Level};

    type Entry = (&'static str, InputClass);

    pub const fn total_len(parts: &[Inventory]) -> usize {
        let (mut n, mut i) = (0, 0);
        while i < parts.len() {
            n += parts[i].len();
            i += 1;
        }
        n
    }

    /// `parts`, end to end. `N` is [`total_len`] of `parts`.
    pub const fn concat<const N: usize>(parts: &[Inventory]) -> [Entry; N] {
        let mut out = [("", InputClass::Removed); N];
        let (mut k, mut i) = (0, 0);
        while i < parts.len() {
            let mut j = 0;
            while j < parts[i].len() {
                out[k] = parts[i][j];
                k += 1;
                j += 1;
            }
            i += 1;
        }
        assert!(k == N, "concat: N is not the parts' total length");
        out
    }

    const fn str_eq(a: &str, b: &str) -> bool {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    const fn position(inv: &[Entry], name: &str) -> Option<usize> {
        let mut i = 0;
        while i < inv.len() {
            if str_eq(inv[i].0, name) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    pub const fn has_field(inv: Inventory, name: &str) -> bool {
        position(inv, name).is_some()
    }

    /// `base` with each `overrides` entry's class in place of the class its
    /// field has in `base`. `N` is `base.len()`.
    pub const fn overridden<const N: usize>(base: Inventory, overrides: &[Entry]) -> [Entry; N] {
        let mut out = [("", InputClass::Removed); N];
        let mut i = 0;
        while i < N {
            out[i] = base[i];
            i += 1;
        }
        let mut o = 0;
        while o < overrides.len() {
            match position(base, overrides[o].0) {
                Some(at) => out[at].1 = overrides[o].1,
                None => panic!("#[input(override)] names a field the group does not have"),
            }
            o += 1;
        }
        out
    }

    const fn level_name(level: Level) -> &'static str {
        match level {
            Level::Model => "model",
            Level::Config => "config",
            Level::Params => "params",
            Level::Scenario => "scenario",
            Level::Seed => "seed",
            Level::Obs => "obs",
            Level::Fit => "fit",
            Level::Method => "method",
            Level::Profile => "profile",
            Level::Box => "box",
        }
    }

    /// A compile-time message: string pieces copied into a fixed buffer.
    struct Message {
        buf: [u8; 512],
        len: usize,
    }

    impl Message {
        const fn new() -> Self {
            Message { buf: [0; 512], len: 0 }
        }
        const fn push(mut self, s: &str) -> Self {
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() && self.len < self.buf.len() {
                self.buf[self.len] = b[i];
                self.len += 1;
                i += 1;
            }
            self
        }
        const fn as_str(&self) -> &str {
            match core::str::from_utf8(self.buf.split_at(self.len).0) {
                Ok(s) => s,
                Err(_) => "an input-class check failed (the message was cut inside a character)",
            }
        }
    }

    /// No argument id occurs twice in `command`'s inventory, and every `keyed`
    /// class names one of its levels.
    pub const fn check_command(command: &str, inputs: Inventory, levels: &[Level]) {
        let mut i = 0;
        while i < inputs.len() {
            let (field, class) = inputs[i];
            if position(inputs.split_at(i + 1).1, field).is_some() {
                let m = Message::new().push("`").push(command).push("`: the input `").push(field).push(
                    "` is classified twice (a field and a flattened group, or two groups, share a clap id)",
                );
                panic!("{}", m.as_str());
            }
            if let InputClass::Keyed(level) = class {
                let mut found = false;
                let mut l = 0;
                while l < levels.len() {
                    if level as u8 == levels[l] as u8 {
                        found = true;
                    }
                    l += 1;
                }
                if !found {
                    let m = Message::new()
                        .push("`")
                        .push(command)
                        .push("`: the input `")
                        .push(field)
                        .push("` is keyed(")
                        .push(level_name(level))
                        .push("), a level this command's leaf does not have (its #[input(command(levels(..)))]); ")
                        .push("override the class where the command flattens the group that declares it");
                    panic!("{}", m.as_str());
                }
            }
            i += 1;
        }
    }
}

#[cfg(test)]
use InputClass::{Child, Execution, Expands, Keyed, Presentation, Provenance, Removed};
#[cfg(test)]
use Level::{Config, Fit, Method, Model, Obs, Params, Profile, Scenario, Seed};

/// `camdl simulate`'s inputs, by clap argument id (the field name), with their
/// classes: `SimulateArgs`, the groups it flattens and the global options.
#[cfg(test)]
pub const SIMULATE: &[(&str, InputClass)] = &[
    // SimulateArgs
    ("model", Keyed(Model)),
    ("to", Keyed(Config)),
    // `--from` moves the model's start (and the starts derived from it) in the
    // one load the run and its identity share, so it keys the model level.
    ("from", Keyed(Model)),
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

/// `camdl pfilter`'s inputs, by clap argument id (the field name), with their
/// classes: `PfilterArgs`, the groups it flattens and the global options.
///
/// A pfilter leaf's levels are `model` (the resolved model, with the engine),
/// `config` (the scoring setup and the data), `params` (the scored point) and
/// `seed`. `--fit` keys the config level through what its problem supplies:
/// the objective, the data, the two policies. `--table` and a scenario change
/// the resolved model. `--save-final-state` writes a file, and under the fit's
/// `holdout_after` cuts the bound rows to the training rows, which the config
/// level keys. `--pf-max-substeps` and `--time-format` decide whether a run is
/// refused, never what a run that completes stores, so they are `execution`.
#[cfg(test)]
pub const PFILTER: &[(&str, InputClass)] = &[
    // PfilterArgs
    ("model", Keyed(Model)),
    ("particles", Keyed(Config)),
    ("data", Keyed(Config)),
    ("fit", Keyed(Config)),
    ("replicates", Keyed(Config)),
    ("trace", Presentation),
    ("pf_health", Presentation),
    ("output", Presentation),
    ("save_final_state", Keyed(Config)),
    ("save_paths", Presentation),
    ("n_paths", Presentation),
    ("save_filtering", Presentation),
    ("save_filtered_streams", Presentation),
    ("save_prequential", Presentation),
    ("score_from", Presentation),
    ("no_save_samples", Presentation),
    ("no_store", Presentation),
    // ModelOverrides
    ("params", Keyed(Params)),
    ("param", Keyed(Params)),
    ("table", Keyed(Model)),
    // ScenarioArgs
    ("scenario", Keyed(Model)),
    ("enable", Keyed(Model)),
    ("disable", Keyed(Model)),
    // InferenceCore
    ("dt", Keyed(Config)),
    ("seed", Keyed(Seed)),
    ("parallel", Execution),
    ("allow_degenerate_rates", Keyed(Config)),
    ("time_format", Execution),
    ("pf_max_substeps", Execution),
    ("binomial", Keyed(Config)),
    // ForcingExtrapolationArg
    ("allow_forcing_extrapolation", Keyed(Config)),
    // StreamSelection
    ("obs", Keyed(Config)),
    // GlobalArgs
    ("verbosity", Presentation),
    ("progress", Presentation),
    ("no_progress", Presentation),
    ("no_ir_cache", Execution),
    ("no_licm", Keyed(Model)),
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

/// `camdl survey`'s inputs, by clap argument id (the field name), with their
/// classes: `SurveyArgs`, the group it flattens and the global options.
///
/// A survey leaf's levels are `model` (the resolved model and the engine),
/// `config` (the evaluation, the data, the fixed values, the scenario, the
/// objective, the two policies), `box` (the sampled bounds and the number of
/// points) and `seed`. `--fit` keys the config level through what its
/// problem supplies. `--output` is the store root, so it is `execution`, with
/// `--force` and `--parallel`. `--render` asks for the leaf's child
/// `landscape_html/`.
#[cfg(test)]
pub const SURVEY: &[(&str, InputClass)] = &[
    // SurveyArgs
    ("model", Keyed(Model)),
    ("allow_degenerate_rates", Keyed(Config)),
    ("fit", Keyed(Config)),
    ("estimate", Keyed(Level::Box)),
    ("data", Keyed(Config)),
    ("fixed", Keyed(Config)),
    ("scenario", Keyed(Config)),
    ("n_points", Keyed(Level::Box)),
    ("eval", Keyed(Config)),
    ("eval_particles", Keyed(Config)),
    ("eval_replicates", Keyed(Config)),
    ("seed", Keyed(Seed)),
    ("render", Child),
    ("output", Execution),
    ("label", Provenance),
    ("force", Execution),
    ("parallel", Execution),
    // ForcingExtrapolationArg
    ("allow_forcing_extrapolation", Keyed(Config)),
    // GlobalArgs
    ("verbosity", Presentation),
    ("progress", Presentation),
    ("no_progress", Presentation),
    ("no_ir_cache", Execution),
    ("no_licm", Keyed(Model)),
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

    /// The inventory `#[derive(Inputs)]` generates for each command is its
    /// hand-written table, row for row, as a set.
    #[test]
    fn the_derived_inventory_is_each_commands_table() {
        use crate::args::{FitRunArgs, MreSimulateArgs, PfilterArgs, ProfileArgs, SimulateArgs, SurveyArgs};
        fn set<'a>(rows: impl IntoIterator<Item = &'a (&'a str, InputClass)>) -> BTreeSet<(String, InputClass)> {
            rows.into_iter().map(|(f, c)| (f.to_string(), *c)).collect()
        }
        let mre_table: Vec<(&str, InputClass)> = SIMULATE.iter().chain(MRE_SIMULATE_OWN).copied().collect();
        for (what, derived, table) in [
            ("simulate", SimulateArgs::INPUTS, SIMULATE),
            ("fit run", FitRunArgs::INPUTS, FIT_RUN),
            ("pfilter", PfilterArgs::INPUTS, PFILTER),
            ("profile", ProfileArgs::INPUTS, PROFILE),
            ("survey", SurveyArgs::INPUTS, SURVEY),
            ("mre simulate", MreSimulateArgs::INPUTS, &mre_table[..]),
        ] {
            assert_eq!(derived.len(), table.len(), "{what}: as many derived inputs as table rows");
            let (d, t) = (set(derived), set(table));
            let only_derived: Vec<_> = d.difference(&t).collect();
            let only_table: Vec<_> = t.difference(&d).collect();
            assert!(
                only_derived.is_empty() && only_table.is_empty(),
                "{what}: the derived inventory and the table disagree.\n  \
                 derived only: {only_derived:?}\n  table only: {only_table:?}"
            );
        }
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
    fn every_pfilter_flag_is_classified() {
        assert_covers("camdl pfilter", clap_args(&["pfilter"]), PFILTER);
    }

    #[test]
    fn every_profile_flag_is_classified() {
        assert_covers("camdl profile", clap_args(&["profile"]), PROFILE);
    }

    #[test]
    fn every_survey_flag_is_classified() {
        assert_covers("camdl survey", clap_args(&["survey"]), SURVEY);
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

    /// Environment inputs with no `pfilter` case, each with the reason.
    const PFILTER_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by pfilter: `fit table` and `camdl list` read it"),
        ("ci", "not read by pfilter: `fit summary` reads it"),
    ];

    /// Environment inputs with no `profile` case, each with the reason.
    const PROFILE_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by profile: `fit table` and `camdl list` read it"),
        ("ci", "not read by profile: `fit summary` reads it"),
    ];

    /// Environment inputs with no `survey` case, each with the reason.
    const SURVEY_NO_CASE: &[(&str, &str)] = &[
        ("skip_version_check", "every test run sets it: without it the handshake refuses a test build's camdlc"),
        ("path", "the harness needs PATH to start camdl; `execution_env_camdlc` varies the camdlc found on it"),
        ("unlabeled_threshold", "not read by survey: `fit table` and `camdl list` read it"),
        ("ci", "not read by survey: `fit summary` reads it"),
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

    /// The same for `survey`, in `tests/survey_input_classes.rs`.
    #[test]
    fn every_survey_input_has_a_case() {
        assert_every_input_has_a_case("survey", "survey_input_classes.rs", SURVEY, SURVEY_NO_CASE);
    }

    /// The same for `profile`, in `tests/profile_input_classes.rs`.
    #[test]
    fn every_profile_input_has_a_case() {
        assert_every_input_has_a_case("profile", "profile_input_classes.rs", PROFILE, PROFILE_NO_CASE);
    }

    /// The same for `pfilter`, in `tests/pfilter_input_classes.rs`.
    #[test]
    fn every_pfilter_input_has_a_case() {
        assert_every_input_has_a_case("pfilter", "pfilter_input_classes.rs", PFILTER, PFILTER_NO_CASE);
    }
}
