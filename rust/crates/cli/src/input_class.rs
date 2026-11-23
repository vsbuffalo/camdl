//! The class of each environment variable camdl reads ([`crate::env::VARS`]),
//! and the check that `camdl mre simulate` takes `simulate`'s arguments.
//!
//! A variable's class says what it may do to a stored result:
//!
//! | Class          | Meaning                                                      |
//! | -------------- | ------------------------------------------------------------ |
//! | `keyed(level)` | can change the result; part of the resolved configuration    |
//! | `execution`    | changes how the result is computed, not what (`--parallel`)  |
//! | `presentation` | how output is written outside the store (`-o`)               |
//!
//! The commands on the run store declare their command-line inputs' classes
//! in their argument types instead (`#[derive(store::Inputs)]`), and check
//! them with `store::check_input_classes`.

/// The identity level a keyed input enters at: `Model`, the compiled model,
/// which every result on the run store depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Model,
}

/// What an environment variable may do to a stored result. See the module
/// table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputClass {
    Keyed(Level),
    Execution,
    Presentation,
}

/// `camdl mre simulate`'s own input. It flattens `SimulateArgs` and writes a
/// bundle that reproduces that simulate run, so every other input keeps the
/// class `SimulateArgs` declares.
#[cfg(test)]
pub const MRE_SIMULATE_OWN: &[(&str, InputClass)] = &[("bundle", InputClass::Presentation)];

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
}
