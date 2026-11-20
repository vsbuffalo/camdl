//! The class of every input of a converted command, declared on its clap field
//! with `#[derive(Inputs)]` and checked against the arguments clap accepts.
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
//! Each field of a command's clap struct, of every group it flattens and of
//! the global options carries its class, `#[input(keyed(config))]` or
//! `#[input(execution)]`; a field without one does not compile
//! (`rust/crates/input-class-derive`). A class belongs to a (command, input)
//! pair, not to a flag: a group that several commands flatten
//! (`ForcingExtrapolationArg`, the global options) declares a class for each
//! field, and a command that needs another says so where it flattens the group,
//! `#[input(override(allow_forcing_extrapolation = keyed(fit)))]`. A command
//! declares the levels of its leaf, `#[input(command(levels(..), global(..)))]`,
//! and a `keyed` class naming a level it does not have does not compile.
//!
//! The derive generates each command's inventory, `SimulateArgs::INPUTS`, and
//! [`COMMANDS`] lists them. `every_classified_command_matches_its_clap_arguments`
//! (below) checks each against the arguments clap accepts, both ways;
//! `every_<command>_input_has_a_case`, that every input has a case in
//! `rust/crates/cli/tests/<command>_input_classes.rs`. Those suites read a keyed
//! input's level from the inventory through `camdl __input-classes`
//! ([`print_classes`]).

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
/// `Provenance` and `Expands` occur only in the command inventories; the
/// environment table ([`crate::env::VARS`]) has neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

    pub const fn level_name(level: Level) -> &'static str {
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

/// Every command whose inputs are classified, by its subcommand path, with its
/// inventory (`#[derive(Inputs)]`).
pub const COMMANDS: &[(&[&str], Inventory)] = &[
    (&["simulate"], crate::args::SimulateArgs::INPUTS),
    (&["fit", "run"], crate::args::FitRunArgs::INPUTS),
    (&["pfilter"], crate::args::PfilterArgs::INPUTS),
    (&["profile"], crate::args::ProfileArgs::INPUTS),
    (&["survey"], crate::args::SurveyArgs::INPUTS),
    (&["mre", "simulate"], crate::args::MreSimulateArgs::INPUTS),
];

/// The spelling of a class in `#[input(..)]`: `keyed(config)`, `execution`.
pub fn spelling(class: InputClass) -> String {
    match class {
        InputClass::Keyed(level) => format!("keyed({})", derive_support::level_name(level)),
        InputClass::Execution => "execution".into(),
        InputClass::Provenance => "provenance".into(),
        InputClass::Presentation => "presentation".into(),
        InputClass::Child => "child".into(),
        InputClass::Expands => "expands".into(),
        InputClass::Removed => "removed".into(),
    }
}

/// `camdl __input-classes <command...>`, a hidden mode: every input of the
/// classified command `path` names, then every environment variable, one per
/// line as `<id>\t<class>` (`env:<field>` for a variable), in the spelling of
/// `#[input(..)]`. The differential tests in `tests/<command>_input_classes.rs`
/// read a keyed input's level from it, since an integration test cannot import
/// this crate. Fails, naming the classified commands, when `path` is none of them.
pub fn print_classes(path: &[String]) -> i32 {
    let out = crate::out::detached();
    let Some((_, inputs)) = COMMANDS.iter().find(|(p, _)| p.iter().copied().eq(path.iter().map(String::as_str))) else {
        let known: Vec<String> = COMMANDS.iter().map(|(p, _)| p.join(" ")).collect();
        out.fail(format!("`{}` is not a classified command; one of: {}", path.join(" "), known.join(", ")));
    };
    for (id, class) in *inputs {
        out.data_line(format!("{id}\t{}", spelling(*class)));
    }
    for v in crate::env::VARS {
        out.data_line(format!("env:{}\t{}", v.field, spelling(v.class)));
    }
    0
}

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

    /// Every classified command's inventory names exactly the arguments clap
    /// accepts for it. The derive sees one struct and the groups it flattens;
    /// this sees the parser `main.rs` builds. So it catches what the derive
    /// cannot: a subcommand wired to a struct other than the one classified,
    /// global options that reach a command from anywhere but the group its
    /// `global(..)` names, and a field whose clap id is not its name
    /// (`#[arg(id = ..)]`) or that is not an argument at all (`#[arg(skip)]`).
    #[test]
    fn every_classified_command_matches_its_clap_arguments() {
        for (path, inputs) in COMMANDS {
            let what = format!("camdl {}", path.join(" "));
            let clap = clap_args(path);
            let classified: BTreeSet<String> = inputs.iter().map(|(f, _)| f.to_string()).collect();
            let missing: Vec<&String> = clap.difference(&classified).collect();
            let stale: Vec<&String> = classified.difference(&clap).collect();
            assert!(
                missing.is_empty() && stale.is_empty(),
                "{what}: the derived inventory and the arguments clap accepts disagree.\n  \
                 accepted by clap, not classified: {missing:?}\n  \
                 classified, not accepted by clap: {stale:?}"
            );
        }
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

    /// The cases `inputs` and the environment table (less `no_case`) call for,
    /// against those `file` holds.
    fn assert_every_input_has_a_case(what: &str, file: &str, inputs: Inventory, no_case: &[(&str, &str)]) {
        let mut wanted = BTreeSet::new();
        for (field, class) in inputs {
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
            "the cases and {what}'s inventory disagree.\n  \
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
        assert_every_input_has_a_case("simulate", "simulate_input_classes.rs", crate::args::SimulateArgs::INPUTS, NO_CASE);
    }

    /// The same for `fit run`, in `tests/fit_run_input_classes.rs`, where a
    /// `removed` case checks that the old spelling is refused with its
    /// message.
    #[test]
    fn every_fit_run_input_has_a_case() {
        assert_every_input_has_a_case("fit run", "fit_run_input_classes.rs", crate::args::FitRunArgs::INPUTS, FIT_RUN_NO_CASE);
    }

    /// The same for `survey`, in `tests/survey_input_classes.rs`.
    #[test]
    fn every_survey_input_has_a_case() {
        assert_every_input_has_a_case("survey", "survey_input_classes.rs", crate::args::SurveyArgs::INPUTS, SURVEY_NO_CASE);
    }

    /// The same for `profile`, in `tests/profile_input_classes.rs`.
    #[test]
    fn every_profile_input_has_a_case() {
        assert_every_input_has_a_case("profile", "profile_input_classes.rs", crate::args::ProfileArgs::INPUTS, PROFILE_NO_CASE);
    }

    /// The same for `pfilter`, in `tests/pfilter_input_classes.rs`.
    #[test]
    fn every_pfilter_input_has_a_case() {
        assert_every_input_has_a_case("pfilter", "pfilter_input_classes.rs", crate::args::PfilterArgs::INPUTS, PFILTER_NO_CASE);
    }
}
