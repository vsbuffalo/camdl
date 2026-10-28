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

// ── Fit artifacts written before constants left the parameters ─────────────
//
// A fit run before IR 0.41 wrote the model's constants into its
// `mle_params.toml` beside the parameters. Such a file, when its values still
// hash to its `[provenance] content_hash`, has the constant's entry set aside
// if it equals the model's declared value; anything else naming a constant is
// refused.

/// The tamper hash `write_mle_params` writes into `content_hash`.
fn tamper_hash(values: &[(&str, f64)]) -> String {
    use sha2::{Digest, Sha256};
    let mut v: Vec<&(&str, f64)> = values.iter().collect();
    v.sort_by_key(|(k, _)| *k);
    let mut h = Sha256::new();
    for (name, value) in v {
        h.update(format!("{name}={value:.12}\x00").as_bytes());
    }
    hex::encode(&h.finalize()[..4])
}

/// An `mle_params.toml` as a pre-0.41 fit wrote it, declaring `content_hash`
/// (the true tamper hash of `values` unless `declared` overrides it).
fn artifact(values: &[(&str, f64)], declared: Option<&str>) -> String {
    let hash = declared.map(str::to_string).unwrap_or_else(|| tamper_hash(values));
    let mut s = String::from("# camdl fit MLE\n");
    for (k, v) in values {
        s.push_str(&format!("{k} = {v:?}\n"));
    }
    s.push_str(&format!("\n[provenance]\ncamdl_version = \"0.1.0+ed634245\"\n\
        timestamp = \"2026-10-27T23:58:12Z\"\ncontent_hash = \"{hash}\"\n\
        fit_hash = \"97690198bb9a0062e570096cbc8f9c6bad93abe90a241633e2ee1b1ff119ddd0\"\n\
        backend = \"chain_binomial\"\ndt = 1.0\nmodel = \"model.camdl\"\n\
        model_identity = \"ea325bdbe922e0d43235fb7996d0ec0dd47b3335e9daac4cc4a17603c6f349e1\"\n\
        seed = 3\nmethod = \"if2\"\nchain = 1\nlog_likelihood = -49.29\nloglik_sd = 0.0\n\
        n_particles = 100\n"));
    s
}

const FIT_VALUES: &[(&str, f64)] =
    &[("I0", 10.0), ("N0", 10000.0), ("beta", 0.4), ("gamma", 0.15), ("omega", 0.01), ("rho", 0.6)];

/// `profile` with one `--fixed-file`: a path that reads the file without
/// `simulate --params`'s own tamper refusal in front of it.
fn profile_with_file(dir: &Path, file: &str) -> Output {
    camdl(dir, &["profile", "model.camdl", "--data", "weekly_cases.tsv",
                 "--sweep", "beta=0.4", "--fixed", "N0=10000", "--fixed", "I0=10",
                 "--fixed", "rho=0.6", "--fixed-file", file, "--rw-sd", "gamma=0.02",
                 "--starts", "from_params=params.toml", "--particles", "20",
                 "--iterations", "1"])
}

#[test]
fn a_verified_legacy_artifact_at_the_declared_value_runs_with_a_notice() {
    let tmp = fixture();
    let dir = tmp.path();
    std::fs::write(dir.join("mle_params.toml"), artifact(FIT_VALUES, None)).unwrap();
    let notice = "mle_params.toml: ignored omega";
    let o = camdl(dir, &["simulate", "model.camdl", "--params", "mle_params.toml"]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "simulate --params of a verified legacy artifact: {stderr}");
    assert!(stderr.contains(notice), "the set-aside entry is reported: {stderr}");
    // `--fixed-file` reaches the same rule, and `profile`'s partition agrees.
    std::fs::write(dir.join("held.toml"), artifact(&[("gamma", 0.15), ("omega", 0.01)], None))
        .unwrap();
    let o = profile_with_file(dir, "held.toml");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "profile --fixed-file of a verified legacy artifact: {stderr}");
    assert!(stderr.contains("held.toml: ignored omega"), "{stderr}");
}

#[test]
fn a_verified_artifact_whose_constant_differs_is_refused() {
    let tmp = fixture();
    let dir = tmp.path();
    let mut values = FIT_VALUES.to_vec();
    values.iter_mut().find(|(k, _)| *k == "omega").unwrap().1 = 0.02;
    std::fs::write(dir.join("mle_params.toml"), artifact(&values, None)).unwrap();
    refused_with_hint(&camdl(dir, &["simulate", "model.camdl", "--params", "mle_params.toml"]),
        "a verified artifact recording omega = 0.02 against the model's 0.01");
}

#[test]
fn a_hand_written_file_naming_a_constant_is_refused_even_at_its_value() {
    let tmp = fixture();
    let dir = tmp.path();
    std::fs::write(dir.join("hand.toml"), "gamma = 0.15\nomega = 0.01\n").unwrap();
    refused_with_hint(&profile_with_file(dir, "hand.toml"), "a hand-written --fixed-file");
    std::fs::write(dir.join("hand_params.toml"), format!("{PARAMS}omega = 0.01\n")).unwrap();
    refused_with_hint(&camdl(dir, &["simulate", "model.camdl", "--params", "hand_params.toml"]),
        "a hand-written --params");
}

/// A `[provenance]` block proves nothing by being there: one whose
/// `content_hash` the values do not match — forged outright, or edited — is
/// a hand-written file, and so is a genuine artifact whose values were edited.
#[test]
fn a_forged_or_edited_provenance_block_does_not_unlock_a_constant() {
    let tmp = fixture();
    let dir = tmp.path();
    let held = [("gamma", 0.15), ("omega", 0.01)];
    std::fs::write(dir.join("forged.toml"), artifact(&held, Some("deadbeef"))).unwrap();
    refused_with_hint(&profile_with_file(dir, "forged.toml"), "a forged content_hash");
    // A genuine artifact with its block's hash edited.
    let genuine = artifact(&held, None);
    let edited_block = genuine.replace(&tamper_hash(&held), "0badc0de");
    assert_ne!(genuine, edited_block);
    std::fs::write(dir.join("edited_block.toml"), edited_block).unwrap();
    refused_with_hint(&profile_with_file(dir, "edited_block.toml"), "an edited content_hash");
    // A genuine artifact with a value edited after the fit wrote it.
    let edited_value = genuine.replace("gamma = 0.15", "gamma = 0.2");
    std::fs::write(dir.join("edited_value.toml"), edited_value).unwrap();
    refused_with_hint(&profile_with_file(dir, "edited_value.toml"), "an edited value");
}
