//! Pins the contract of the hidden `camdl __check-args -- <argv...>`
//! parse-only mode that backs `make test-cli-docs`.
//!
//! The mode runs ONLY clap parsing against the real `Cli` command tree —
//! no file I/O, no compilation, no simulation — and exits:
//!   0  the surface parses (subcommand + flags + arg shape are real), OR
//!      clap wants help/version, OR a required positional is merely missing
//!      (an input concern, not surface drift).
//!   2  surface DRIFT: unknown subcommand / unrecognized flag / unexpected
//!      positional / bad arg count / invalid enum value.
//!
//! This is the binary-level half of the non-vacuous drift gate: it proves the
//! exit-code contract directly, independent of the shell extractor in
//! `scripts/check_cli_docs.sh` (whose `--selftest` proves the same contract
//! through the doc-extraction path).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

/// Run `camdl __check-args -- <args>` and return the exit code.
fn check(args: &[&str]) -> i32 {
    let out = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .arg("__check-args")
        .arg("--")
        .args(args)
        .output()
        .expect("spawn camdl __check-args");
    out.status.code().expect("process exited with a code")
}

// ── Valid surface → exit 0 (EXPECTED input failures are NOT drift) ───────────

#[test]
fn valid_surface_with_missing_inputs_is_ok() {
    // Every one of these references a real subcommand + real flags; the only
    // thing "wrong" is that the files don't exist. The parse-only check never
    // touches the filesystem, so these MUST parse clean.
    assert_eq!(check(&["simulate", "/no/such/model.camdl", "--seed", "1"]), 0);
    assert_eq!(check(&["fit", "run", "/no/such/fit.toml", "--seed", "1"]), 0);
    assert_eq!(check(&["fit", "summary", "/no/such/dir"]), 0);
    assert_eq!(check(&["pfilter", "m.camdl", "--params", "p.toml",
                       "--data", "d.tsv", "--particles", "5000"]), 0);
    assert_eq!(check(&["compare", "a", "b"]), 0);
    assert_eq!(check(&["survey", "m.camdl", "--fit", "f.toml", "--render"]), 0);
    assert_eq!(check(&["simulate", "m.camdl", "--draws", "prior",
                       "--fit", "f.toml", "-n", "200", "--obs", "ppc.tsv"]), 0);
}

#[test]
fn missing_required_positional_is_not_drift() {
    // The subcommand is recognized; a missing required positional is an input
    // concern, not surface drift. (E.g. a doc snippet that omits the path.)
    assert_eq!(check(&["fit", "summary"]), 0);
}

#[test]
fn camdlc_passthrough_subcommands_accept_their_own_flags() {
    // check/inspect (and `camdl dev compile`/`dev doctest`) forward verbatim to
    // camdlc, so clap accepts any tail after them. Documented scope limit: those
    // flags are camdlc's surface, not camdl's. Must parse OK here.
    assert_eq!(check(&["check", "model.camdl"]), 0);
    assert_eq!(check(&["dev", "compile", "model.camdl", "--set", "beta=0.3",
                       "--json-errors"]), 0);
    assert_eq!(check(&["inspect", "model.camdl", "--tables"]), 0);
}

#[test]
fn subcommand_alias_resolves() {
    // `sim` is an alias for `simulate`; aliases are real surface.
    assert_eq!(check(&["sim", "m.camdl", "--seed", "1"]), 0);
}

// ── Surface DRIFT → exit 2 (the NON-VACUOUS negative test) ───────────────────

#[test]
fn unknown_subcommand_is_drift() {
    assert_eq!(check(&["frobnicate", "foo"]), 2);
}

#[test]
fn unrecognized_flag_is_drift() {
    assert_eq!(check(&["simulate", "model.camdl", "--no-such-flag"]), 2);
}

#[test]
fn unknown_fit_subcommand_is_drift() {
    assert_eq!(check(&["fit", "bogus", "fit.toml"]), 2);
}

#[test]
fn invalid_enum_value_is_drift() {
    // --backend takes a typed enum; a bogus value is a parse-layer rejection.
    assert_eq!(check(&["simulate", "model.camdl", "--backend", "not_a_backend"]), 2);
}

#[test]
fn flag_on_wrong_subcommand_is_drift() {
    // `--particles` belongs to pfilter/profile/survey, not simulate.
    assert_eq!(check(&["simulate", "model.camdl", "--particles", "10"]), 2);
}

// ── gh#194: pfilter `--scenario` ⊥ `--params` / `--param` ─────────────────────
//
// On pfilter a scenario's `set`/`scale` block resolves at a higher precedence
// than `--params`, so combining them would silently score the likelihood at
// the scenario's θ rather than the user's pinned θ. The combination is a hard
// clap-level conflict; clap rejects it at the parse layer (exit 2), so the
// silent-wrong-θ run can never start.

#[test]
fn pfilter_scenario_with_params_file_is_conflict() {
    assert_eq!(
        check(&["pfilter", "m.camdl", "--scenario", "baseline",
                "--params", "p.toml", "--data", "d.tsv", "--particles", "100"]),
        2,
        "`pfilter --scenario S --params FILE` must be a parse-layer conflict \
         (gh#194): the scenario silently overrides the pinned θ otherwise."
    );
}

#[test]
fn pfilter_scenario_with_param_cli_is_conflict() {
    assert_eq!(
        check(&["pfilter", "m.camdl", "--scenario", "baseline",
                "--param", "beta=0.3", "--data", "d.tsv", "--particles", "100"]),
        2,
        "`pfilter --scenario S --param NAME=VALUE` must be a parse-layer \
         conflict (gh#194), same root cause as --params."
    );
}

#[test]
fn pfilter_scenario_without_explicit_theta_is_ok() {
    // Scenario alone (θ comes from the scenario's set/scale) is the
    // intended way to score a named scenario — must still parse clean.
    assert_eq!(
        check(&["pfilter", "m.camdl", "--scenario", "baseline",
                "--data", "d.tsv", "--particles", "100"]),
        0
    );
}

#[test]
fn pfilter_params_without_scenario_is_ok() {
    // Pinning θ via --params with no scenario is the canonical pfilter
    // invocation — must parse clean.
    assert_eq!(
        check(&["pfilter", "m.camdl", "--params", "p.toml",
                "--data", "d.tsv", "--particles", "100"]),
        0
    );
}

#[test]
fn pfilter_params_with_enable_is_ok() {
    // `--enable`/`--disable` toggle interventions, not parameter values, so
    // "pin θ + toggle an intervention" stays coherent and must NOT conflict
    // with --params. Only --scenario (which sets θ) conflicts.
    assert_eq!(
        check(&["pfilter", "m.camdl", "--params", "p.toml", "--enable", "sia",
                "--data", "d.tsv", "--particles", "100"]),
        0
    );
}

#[test]
fn pfilter_params_and_param_together_is_ok() {
    // The conflict is scenario-vs-explicit-θ, NOT params-vs-param. A file plus
    // a singular override is a legitimate layering and must stay accepted.
    assert_eq!(
        check(&["pfilter", "m.camdl", "--params", "p.toml", "--param", "beta=0.3",
                "--data", "d.tsv", "--particles", "100"]),
        0
    );
}
