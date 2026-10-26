//! `camdl profile` has no per-cell PMMH. A profile likelihood is a maximum
//! over the nuisance parameters at each grid value; a per-cell chain's
//! highest-posterior sample is neither that maximum nor the marginal
//! posterior of the swept parameter. `--algorithm pmmh` is refused on every
//! backend, and the `--pmmh-*` flags are trapped, each with one message that
//! names the maximizers to use instead and the `fit run` route to a
//! posterior. The model path is never opened: the refusal answers first.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

/// `profile` on a model that does not exist, with the required arguments
/// supplied, plus `extra`.
fn profile(extra: &[&str]) -> Output {
    let mut argv = vec![
        "profile", "/no/such/model.ir.json",
        "--sweep", "beta=lin(0.1,0.2,2)",
        "--particles", "10",
        "--rw-sd", "auto",
    ];
    argv.extend_from_slice(extra);
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(&argv)
        .output()
        .expect("spawn camdl")
}

/// The refusal names why (a profile is a maximization), what to run per cell
/// instead, and where a posterior comes from.
fn assert_refusal(out: &Output, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{what}: the refusal exits 1; stderr: {stderr}");
    for needle in [
        "per-cell PMMH was removed",
        "profile likelihood",
        "--algorithm if2",
        "nl-sbplx",
        "camdl fit run",
    ] {
        assert!(stderr.contains(needle), "{what}: refusal must contain {needle:?}; got: {stderr}");
    }
}

#[test]
fn profile_algorithm_pmmh_is_refused() {
    let out = profile(&["--algorithm", "pmmh"]);
    assert_refusal(&out, "--algorithm pmmh");
    assert!(String::from_utf8_lossy(&out.stderr).contains("--algorithm pmmh"));
}

#[test]
fn profile_algorithm_pmmh_on_ode_gets_the_same_refusal() {
    // Without the profile check first, the registry's own (pmmh, ode) error
    // would answer instead — true, but silent about profile.
    assert_refusal(&profile(&["--algorithm", "pmmh", "--backend", "ode"]), "pmmh on ode");
}

#[test]
fn profile_pmmh_flags_are_refused_with_the_same_message() {
    for flag in ["--pmmh-steps", "--pmmh-particles"] {
        let out = profile(&["--algorithm", "if2", flag, "500"]);
        assert_refusal(&out, flag);
        assert!(String::from_utf8_lossy(&out.stderr).contains(flag),
            "{flag}: the message names the flag given");
    }
}

#[test]
fn profile_if2_is_not_refused() {
    // Control: the same invocation with a maximizer gets past the refusal
    // and fails on the missing model instead.
    let out = profile(&["--algorithm", "if2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("per-cell PMMH"), "if2 must not be refused; got: {stderr}");
}
