//! A typed `let` with a constant body (`let omega : rate = 0.01`) is a model
//! constant, not a parameter: it compiles to the IR's `constants` section and
//! is read through a `const_ref` node (proposal
//! `docs/dev/proposals/2026-10-26-parameter-roles.md`, D2 and D9).
//!
//! End to end through the compiler and the release binary: the constant is
//! absent from the IR's parameters, a value flag that names it is refused as an
//! unknown parameter with a hint naming the constant, and an expression over it
//! evaluates to exactly the value it names.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

/// An SIRS model whose waning rate reads the constant through a `let`:
/// `w = 2 * omega`, used as `w / 2`.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]
  gamma : rate        in [0.01, 0.5]
  rho   : probability in [0.05, 0.95]
  N0    : count
  I0    : count
}
let omega : rate = 0.01
let w = 2 * omega
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  waning    : R --> S  @ w * R
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = 10.0)
  }
}
simulate { from = 0 'days  to = 56 'days }
"#;

const PARAMS: &str = "beta = 0.4\ngamma = 0.15\nrho = 0.6\nN0 = 10000\nI0 = 10\n";

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The hint every external naming of `omega` must carry.
const HINT: &str = "`omega` is a constant of the model (`let omega : rate = 0.01`), not a \
                    parameter; declare it in `parameters {}` to set it from outside";

fn fixture() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("params.toml"), PARAMS).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    tmp
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin())
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn refused_with_hint(out: &Output, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{what} must be refused; it exited 0:\n{stderr}");
    assert!(stderr.contains(HINT), "{what}: the refusal must carry the constant's hint:\n{stderr}");
}

#[test]
fn a_typed_constant_is_in_constants_not_parameters() {
    let tmp = fixture();
    let out = Command::new(camdlc_bin()).arg(tmp.path().join("model.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let env: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let model = &env["model"];
    let params: Vec<&str> = model["parameters"].as_array().unwrap()
        .iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert!(!params.contains(&"omega"), "a constant is not a parameter: {params:?}");
    assert_eq!(model["constants"], serde_json::json!([
        { "name": "omega", "value": 0.01, "kind": "rate" }
    ]));
    // `w = 2 * omega` is param-free, so it is a hoisted binding over the
    // constant reference.
    let w = model["bindings"].as_array().unwrap().iter()
        .find(|b| b["name"] == "w").expect("w is hoisted into bindings");
    assert_eq!(w["expr"]["bin_op"]["right"], serde_json::json!({ "const_ref": "omega" }));
}

#[test]
fn every_value_flag_that_names_a_constant_is_refused_with_its_hint() {
    let tmp = fixture();
    let dir = tmp.path();
    refused_with_hint(
        &camdl(dir, &["simulate", "model.camdl", "--params", "params.toml",
                      "--param", "omega=0.02"]),
        "simulate --param omega",
    );
    std::fs::write(dir.join("omega.toml"), "omega = 0.02\n").unwrap();
    refused_with_hint(
        &camdl(dir, &["simulate", "model.camdl", "--params", "params.toml",
                      "--params", "omega.toml"]),
        "simulate --params naming omega",
    );
    refused_with_hint(
        &camdl(dir, &["simulate", "model.camdl", "--params", "params.toml",
                      "--sweep", "omega=0.01,0.02"]),
        "simulate --sweep omega",
    );
    // Even at its declared value (D9).
    std::fs::write(dir.join("fit.toml"), "output_dir = \"res\"\n\
        [model]\ncamdl = \"model.camdl\"\n\
        [data.observations]\nweekly_cases = \"weekly_cases.tsv\"\n\
        [estimate]\nbeta = { start = 0.4 }\n\
        [fixed]\ngamma = 0.15\nrho = 0.6\nN0 = 10000\nI0 = 10\nomega = 0.01\n\
        [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
        particles = 20\niterations = 1\ncooling = 0.7\n").unwrap();
    refused_with_hint(&camdl(dir, &["fit", "run", "fit.toml"]), "fit run [fixed] omega");
    refused_with_hint(
        &camdl(dir, &["profile", "model.camdl", "--data", "weekly_cases.tsv",
                      "--sweep", "beta=0.4", "--fixed", "gamma=0.15", "--fixed", "rho=0.6",
                      "--fixed", "N0=10000", "--fixed", "I0=10", "--fixed", "omega=0.02",
                      "--rw-sd", "auto", "--starts", "from_params=params.toml",
                      "--particles", "20", "--iterations", "1"]),
        "profile --fixed omega",
    );
}

/// `let w = 2 * omega` evaluates to 0.02, bit for bit: the waning flow of the
/// model over the constant equals that of the same model with `0.02` written
/// in. `2 * 0.01` is exactly `0.02` in binary floating point (a doubling), so
/// any difference is a wrong value for the constant, not rounding.
#[test]
fn an_expression_over_a_constant_evaluates_to_its_value() {
    let tmp = fixture();
    let dir = tmp.path();
    std::fs::write(dir.join("literal.camdl"),
        MODEL.replace("waning    : R --> S  @ w * R", "waning    : R --> S  @ 0.02 'per_day * R"))
        .unwrap();
    for (model, out) in [("model.camdl", "via_constant.tsv"), ("literal.camdl", "literal.tsv")] {
        let o = camdl(dir, &["simulate", model, "--params", "params.toml",
                             "--backend", "ode", "-o", out]);
        assert!(o.status.success(), "simulate {model}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let body = |f: &str| -> String {
        std::fs::read_to_string(dir.join(f)).unwrap()
            .lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>().join("\n")
    };
    let (a, b) = (body("via_constant.tsv"), body("literal.tsv"));
    assert!(a.lines().count() > 50, "the trajectory is non-trivial:\n{a}");
    assert_eq!(a, b, "w = 2 * omega must evaluate to 0.02");
}
