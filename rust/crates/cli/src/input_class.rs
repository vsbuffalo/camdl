//! The class of every input of a converted command, and the inventory the
//! `#[derive(Classify)]` of `classify-derive` emits from its surface structs.
//!
//! An input is a command-line field (or, in [`crate::env`], an environment
//! variable). Its class says what it may do to a stored result, and how that is
//! checked (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "Classification"):
//!
//! | Class          | Meaning                                                      | Verification                                             |
//! | -------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
//! | `keyed(level)` | can change the result; part of the resolved configuration    | differential: changed alone, the run id changes          |
//! | `execution`    | changes how the result is computed, not what (`--parallel`)  | differential: run id and stored bytes stay the same      |
//! | `provenance`   | how the run was asked for (`--label`)                        | structural: not a field of the resolved configuration    |
//! | `presentation` | how output is written outside the store (`-o`)               | structural, plus the leaf is unaffected                  |
//! | `expands`      | defines the set of runs (`--seeds`, `--sweep`)               | differential: cells common to both keep their run ids    |
//! | `removed`      | refuses an old spelling with a migration message             | the flag fails with the message                          |
//!
//! The differential cases are `rust/crates/cli/tests/simulate_input_classes.rs`;
//! `every_simulate_flag_is_classified` (below) checks the inventory covers every
//! argument clap accepts.

// Read only by the tests below until the environment module reads input
// classes at run time (phase 3 step 1, part E), which removes this.
#![cfg_attr(not(test), allow(dead_code))]

/// The identity level a keyed input enters at. The first five are the levels of
/// a simulation leaf ([`runid::inputs::TrajectoryInput`]); `Obs` is the address
/// of the synthetic-observation subtree under a leaf
/// ([`crate::cas_sink::obs_subtree_hash`]), which a trajectory's own run id does
/// not cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Model,
    Config,
    Params,
    Scenario,
    Seed,
    Obs,
}

/// What an input may do to a stored result. See the module table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputClass {
    Keyed(Level),
    Execution,
    Provenance,
    Presentation,
    Expands,
    // `Removed` (`#[class(removed)]`) joins with the first converted command
    // that has a removed-spelling trap: `simulate` has none, and a variant
    // nothing constructs is dead code. The derive already emits it.
}

/// One input and its class. `field` is the surface struct's field name, which
/// is clap's argument id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    pub field: &'static str,
    pub class: InputClass,
}

/// A surface struct whose every input is classified: implemented only by
/// `#[derive(Classify)]`.
pub trait Classify {
    /// Every input of the struct, flattened groups included, in declaration
    /// order.
    fn inventory() -> Vec<Input>;
}

/// The inventory of `camdl simulate`: its own arguments and the global ones
/// every subcommand accepts.
pub fn simulate_inventory() -> Vec<Input> {
    let mut v = crate::args::SimulateArgs::inventory();
    v.extend(crate::GlobalArgs::inventory());
    v
}

/// The inventory of `camdl mre simulate`, which flattens [`crate::args::SimulateArgs`].
pub fn mre_simulate_inventory() -> Vec<Input> {
    let mut v = crate::args::MreSimulateArgs::inventory();
    v.extend(crate::GlobalArgs::inventory());
    v
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

    fn assert_covers(what: &str, clap: BTreeSet<String>, inventory: Vec<Input>) {
        let mut seen = BTreeSet::new();
        for i in &inventory {
            assert!(seen.insert(i.field.to_string()), "{what}: `{}` is classified twice", i.field);
        }
        let missing: Vec<&String> = clap.difference(&seen).collect();
        let stale: Vec<&String> = seen.difference(&clap).collect();
        assert!(
            missing.is_empty() && stale.is_empty(),
            "{what}: the inventory and the arguments clap accepts disagree.\n  \
             accepted by clap, not classified: {missing:?}\n  \
             classified, not accepted by clap: {stale:?}"
        );
    }

    #[test]
    fn every_simulate_flag_is_classified() {
        assert_covers("camdl simulate", clap_args(&["simulate"]), simulate_inventory());
    }

    #[test]
    fn every_mre_simulate_flag_is_classified() {
        assert_covers("camdl mre simulate", clap_args(&["mre", "simulate"]), mre_simulate_inventory());
    }
}
