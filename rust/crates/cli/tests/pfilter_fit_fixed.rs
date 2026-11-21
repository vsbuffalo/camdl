//! `camdl pfilter --fit` applies the fit config's `[fixed]` table at tier 2 of
//! the parameter precedence chain (`docs/camdl-run-spec.md` §1.3, "fit.toml
//! [fixed] ... and --fit on others"). A parameter that only `[fixed]` sets is
//! scored at that value, and the evaluation is the one the same value passed
//! by `--param` gives: the same log-likelihood bit for bit, and the same
//! stored leaf, since the leaf's identity hashes resolved values, not where
//! they came from.
//!
//! End-to-end via the release `camdl` binary, with `camdlc.exe` on PATH.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin")
}

/// A two-parameter SIR. A `.camdl` compiles every parameter without a value,
/// so a parameter is scored only at a value some tier supplies.
const MODEL: &str = r#"time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 5.0]
  gamma : rate in [0.01, 3.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

const DATA: &str = "time\tcases\n2\t3\n4\t1\n6\t0\n8\t0\n10\t0\n";

/// `beta` is estimated and `gamma` is set only by `[fixed]`.
const FIT: &str = r#"[model]
camdl = "model.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
beta = { bounds = [0.1, 3.0], start = 0.8 }

[fixed]
gamma = 1.5

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 200
"#;

fn fixture(dir: &Path) {
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT).unwrap();
}

/// `camdl pfilter model.camdl ARGS` in `dir`, storing under `dir/store`.
fn pfilter(dir: &Path, args: &[&str]) -> Output {
    // `pfilter` compiles the `.camdl` with the `camdlc` first on PATH: put the
    // freshly built one there.
    let path = std::env::var("PATH").unwrap_or_default();
    let shim = dir.join("bin");
    std::fs::create_dir_all(&shim).unwrap();
    let camdlc = shim.join("camdlc");
    if !camdlc.exists() {
        std::os::unix::fs::symlink(camdlc_dir().join("camdlc.exe"), &camdlc).unwrap();
    }
    Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("store"))
        .env("PATH", format!("{}:{path}", shim.display()))
        .current_dir(dir)
        .arg("pfilter")
        .arg("model.camdl")
        .args(args)
        .output()
        .unwrap()
}

/// The text without ANSI escape sequences (the status lines are coloured).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() { break; }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The stored leaf of a successful call: its `run_id` and the full-precision
/// `loglik` of its `loglik.toml`.
fn stored(dir: &Path, out: &Output) -> (String, f64) {
    let stderr = strip_ansi(&String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pfilter failed:\n{stderr}");
    let run_id = stderr.lines()
        .find_map(|l| l.trim_start().strip_prefix("camdl cat "))
        .unwrap_or_else(|| panic!("no `camdl cat` hint:\n{stderr}"))
        .trim().to_string();
    let leaf = stderr.lines()
        .find_map(|l| l.trim_start().strip_prefix("stored "))
        .unwrap_or_else(|| panic!("no `stored` line:\n{stderr}"))
        .trim().to_string();
    let toml = std::fs::read_to_string(dir.join(&leaf).join("loglik.toml"))
        .unwrap_or_else(|e| panic!("{leaf}/loglik.toml: {e}"));
    let loglik = toml.lines()
        .find_map(|l| l.strip_prefix("loglik = "))
        .unwrap_or_else(|| panic!("no loglik in {toml}"))
        .parse::<f64>().unwrap();
    (run_id, loglik)
}

#[test]
fn pfilter_fit_scores_a_parameter_set_only_in_fixed() {
    let bin = camdl_bin();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    assert!(camdlc_dir().join("camdlc.exe").exists(), "camdlc.exe missing — run `make build-ocaml`");

    let from_fit = tempfile::tempdir().unwrap();
    fixture(from_fit.path());
    let out = pfilter(from_fit.path(), &[
        "--fit", "fit.toml", "--param", "beta=0.8", "--particles", "200", "--seed", "1",
    ]);
    let (fit_id, fit_ll) = stored(from_fit.path(), &out);
    assert!(fit_ll.is_finite(), "the fixture must score a finite log-likelihood: {fit_ll}");

    let by_flag = tempfile::tempdir().unwrap();
    fixture(by_flag.path());
    let out = pfilter(by_flag.path(), &[
        "--data", "cases.tsv", "--param", "beta=0.8", "--param", "gamma=1.5",
        "--particles", "200", "--seed", "1",
    ]);
    let (flag_id, flag_ll) = stored(by_flag.path(), &out);

    assert_eq!(fit_ll.to_bits(), flag_ll.to_bits(),
        "gamma from [fixed] must score as gamma from --param: {fit_ll} vs {flag_ll}");
    assert_eq!(fit_id, flag_id, "the two evaluations are one stored leaf");
}

/// A lower tier still wins over `[fixed]`: `--param gamma` replaces the
/// fit's value, and the run says so.
#[test]
fn a_param_flag_overrides_fixed_and_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let out = pfilter(dir.path(), &[
        "--fit", "fit.toml", "--param", "beta=0.8", "--param", "gamma=1.2",
        "--particles", "200", "--seed", "1", "--no-store",
    ]);
    let stderr = strip_ansi(&String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pfilter failed:\n{stderr}");
    assert!(stderr.contains("replaced gamma = 1.5 (from fit.toml [fixed]) with 1.2"),
        "the override of [fixed] gamma = 1.5 by --param gamma=1.2 must be reported:\n{stderr}");
}
