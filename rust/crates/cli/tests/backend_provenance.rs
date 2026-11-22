//! End-to-end tests for the backend-provenance guardrail.
//!
//! See `docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`
//! and the originating incident
//! `docs/dev/incidents/2026-04-19-backend-default-mismatch.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

mod sim_store_cells;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing() -> PathBuf {
    let b = binary();
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        b.display()
    );
    b
}

fn golden_pure_death() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/pure_death.ir.json")
}

/// The tamper hash `camdl fit` declares in `[provenance] content_hash`:
/// SHA-256 over `{name}={value:.12}\0` in name order, first 8 hex characters
/// (`fit::provenance::mle_params_tamper_hash`; run-spec §10.6). The fixtures
/// compute it so each is a fit's unedited file, which `simulate --params`
/// checks before it reads the block.
fn tamper_hash(params: &[(&str, f64)]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted = params.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = Sha256::new();
    for (name, value) in sorted {
        h.update(format!("{}={:.12}\x00", name, value).as_bytes());
    }
    hex::encode(&h.finalize()[..4])
}

/// Write a minimal mle_params.toml with a `[provenance]` block claiming
/// the fit used `backend` at `dt=1.0`. Top-level params come first
/// (required by TOML so they stay at the file's top level, not scoped
/// under the [provenance] table) — matches what `write_mle_params`
/// produces in production, down to the content hash over `mu`.
fn write_fake_mle(path: &Path, backend: &str, dt: f64) {
    let contents = format!(r#"mu = 0.05

[provenance]
camdl_version = "test"
timestamp = "2026-04-19T00:00:00Z"
content_hash = "{hash}"
backend = "{}"
dt = {}
source_model = "../../../pure_death.ir.json"
model_identity = "f00d"
seed = 1
method = "refine"
chain = 1
log_likelihood = -22.0
loglik_sd = 0.0
n_particles = 500
"#, backend, dt, hash = tamper_hash(&[("mu", 0.05)]));
    std::fs::write(path, contents).unwrap();
}

/// A fit-written params file whose values were edited afterwards is refused
/// before anything runs: the block's `content_hash` no longer matches the
/// values, and the message names the file and both ways forward.
#[test]
fn simulate_refuses_a_params_file_edited_after_its_fit() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let text = std::fs::read_to_string(&mle).unwrap();
    assert!(text.contains("mu = 0.05\n"), "{text}");
    std::fs::write(&mle, text.replace("mu = 0.05\n", "mu = 0.07\n")).unwrap();

    let output = tmp.path().join("out");
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--seed", "1", "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an edited fit params file must be refused; stderr: {stderr}");
    let mle_s = mle.to_string_lossy();
    for needle in [
        &format!("params file {mle_s} was edited after the fit that produced it") as &str,
        "content_hash = \"",
        "re-run the fit",
        "remove the [provenance] block",
    ] {
        assert!(stderr.contains(needle), "diagnostic lacks {needle:?}; stderr: {stderr}");
    }
    assert!(sim_store_cells::sim_cells(&output).is_empty(), "a refused run writes no cell");
    assert!(!tmp.path().join("t.tsv").exists(), "a refused run writes no mirror");
}

/// The unedited file a fit writes runs, silently as far as the check goes.
#[test]
fn simulate_runs_an_unedited_fit_params_file() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--seed", "1", "--store", &tmp.path().join("out").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert!(!stderr.contains("edited after the fit"), "stderr: {stderr}");
    assert!(tmp.path().join("t.tsv").is_file());
}

#[test]
fn simulate_auto_matches_fit_backend_when_backend_not_passed() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--store", &tmp.path().join("out").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "expected success; stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("backend auto-matched"),
        "expected auto-match info log, got: {}", stderr);
    assert!(stderr.contains("chain_binomial"),
        "info log should name the fit's backend: {}", stderr);
}

#[test]
fn simulate_warns_on_explicit_backend_mismatch() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--backend", "gillespie",
               "--store", &tmp.path().join("out").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(),
        "simulate should proceed despite mismatch warning; stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("backend mismatch"),
        "expected warning on mismatch, got: {}", stderr);
    assert!(stderr.contains("chain_binomial") && stderr.contains("gillespie"),
        "warning should name both backends: {}", stderr);
    assert!(stderr.contains("incidents/2026-04-19-backend-default-mismatch"),
        "warning should cite the incident file: {}", stderr);
}

#[test]
fn simulate_silent_on_matching_explicit_backend() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--backend", "chain_binomial", "--dt", "1.0",
               "--store", &tmp.path().join("out").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("backend auto-matched"),
        "matching explicit backend should not emit auto-match log: {}", stderr);
    assert!(!stderr.contains("backend mismatch"),
        "matching explicit backend should not emit warning: {}", stderr);
}

#[test]
fn simulate_unchanged_on_standalone_params_file() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let standalone = tmp.path().join("p.toml");
    // No [provenance] block — should behave as pre-guardrail.
    std::fs::write(&standalone, "mu = 0.05\n").unwrap();

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &standalone.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--store", &tmp.path().join("out").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("backend auto-matched"),
        "standalone params should not trigger auto-match log");
    assert!(!stderr.contains("backend mismatch"),
        "standalone params should not trigger warning");
}

#[test]
fn simulate_from_mle_params_writes_a_valid_sim_record() {
    // simulate `--params <mle.toml>` must run and write one valid sim
    // RunRecord. (The sim → fit *lineage* edge — recording which fit produced
    // the params as a `dep` ArtifactRef on the fit's run_id — lands in M3,
    // when fits migrate to RunRecord. In M2 the consumed params are captured
    // by value in the params-level digest, not by a from_fit pointer.)
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    // Use a distinctive fit_hash so we can grep for it.
    let fit_hash = "abc12345deadbeef00000000000000000000000000000000000000000000abcd";
    let contents = format!(r#"mu = 0.05

[provenance]
camdl_version = "test"
timestamp = "2026-04-19T00:00:00Z"
content_hash = "{hash}"
fit_hash = "{}"
backend = "chain_binomial"
dt = 1.0
source_model = "../../../pure_death.ir.json"
model_identity = "f00d"
seed = 1
method = "refine"
chain = 1
log_likelihood = -22.0
loglik_sd = 0.0
n_particles = 500
"#, fit_hash, hash = tamper_hash(&[("mu", 0.05)]));
    std::fs::write(&mle, contents).unwrap();

    let output = tmp.path().join("out");
    let status = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--seed", "1",
               "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success());

    // The one stored sim cell, found by its identity.json.
    let cell = sim_store_cells::only_sim_cell(&output);
    assert_eq!(cell.identity["kind"], "sim", "a valid sim cell");
    assert_eq!(cell.run_id.len(), 64);
    assert_eq!(cell.run_json()["status"], "completed");
    // The mle's backend (chain_binomial) is auto-matched into the config.
    assert_eq!(cell.config()["backend"], "ChainBinomial", "config: {}", cell.config());
}

// ── Behavior tests: read run.json and assert the *resolved* backend ──
//
// The tests above (auto-match, warn, silent) check the stderr text of
// each code path, which is useful as a UX regression guard but not
// sufficient: a bug where the info log says "auto-matched to
// chain_binomial" but the code silently proceeds with Gillespie would
// pass those tests. The tests below fix that by running `simulate`,
// reading the resulting run.json, and asserting the
// `kind.backend` (+ `kind.dt`) fields match what we expect under the
// three-way matching rule.
//
// Two fixtures (one chain_binomial, one gillespie in provenance) so a
// bug hard-coding chain_binomial in the auto-match path is caught —
// checking only one backend direction is insufficient.

/// Find the single sim cell under the store `output_root` and return the
/// resolved `(backend, dt)` recorded in its identity record's `config`
/// (`{"backend": "ChainBinomial", "dt": 1.0}` → `("ChainBinomial", 1.0)`).
fn read_sim_config(output_root: &Path) -> (String, f64) {
    let cell = sim_store_cells::only_sim_cell(output_root);
    let cfg = cell.config();
    let backend = cfg["backend"].as_str()
        .unwrap_or_else(|| panic!("config.backend should be a string: {cfg}")).to_string();
    let dt = cfg["dt"].as_f64()
        .unwrap_or_else(|| panic!("config.dt should be a number: {cfg}"));
    (backend, dt)
}

/// Run `simulate --params <mle> [extra args...]`, return the
/// output root so the caller can inspect run.json.
fn run_simulate_cas(
    bin: &Path, tmp: &Path, mle: &Path, extra: &[&str],
) -> PathBuf {
    let output = tmp.join("out");
    let mut args: Vec<String> = vec![
        "simulate".into(), golden_pure_death().to_string_lossy().to_string(),
        "--params".into(), mle.to_string_lossy().to_string(),
        "--seed".into(), "1".into(),
        "--store".into(), output.to_string_lossy().to_string(),
        "-o".into(), tmp.join("t.tsv").to_string_lossy().to_string(),
    ];
    for a in extra { args.push(a.to_string()); }
    let status = Command::new(bin).env_remove("CLICOLOR_FORCE").args(&args)
        .status().expect("spawn");
    assert!(status.success(), "simulate should succeed");
    output
}

#[test]
fn run_json_records_chain_binomial_when_provenance_says_chain_binomial() {
    // Fixture A: provenance says chain_binomial. No --backend passed.
    // Expected: run.json records backend=chain_binomial (auto-match
    // actually took effect, not just the log text).
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_cbin.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let (backend, dt) = read_sim_config(&output);
    assert_eq!(backend, "ChainBinomial", "config must record auto-matched backend");
    assert_eq!(dt, 1.0);
}

#[test]
fn run_json_records_gillespie_when_provenance_says_gillespie() {
    // Fixture B: provenance says gillespie (tests the opposite
    // direction — guards against a bug where we hardcoded
    // chain_binomial in the auto-match path).
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_gill.toml");
    write_fake_mle(&mle, "gillespie", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let (backend, _dt) = read_sim_config(&output);
    assert_eq!(backend, "Gillespie", "config must record gillespie (opposite direction)");
}

#[test]
fn run_json_records_auto_matched_dt_not_just_backend() {
    // Provenance has dt=0.5 (non-default). Regression guard: a bug
    // where we auto-match backend but NOT dt would make run.json
    // record dt=1.0 (CLI default) despite the fit having used 0.5.
    // Subtly wrong behavior, silent, hard to spot without this check.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_dt05.toml");
    write_fake_mle(&mle, "chain_binomial", 0.5);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let (_backend, dt) = read_sim_config(&output);
    assert_eq!(dt, 0.5,
        "config must record auto-matched dt 0.5, not the 1.0 default");
}

#[test]
fn explicit_backend_overrides_provenance_in_run_json() {
    // Provenance says chain_binomial; user explicitly passes
    // --backend gillespie. The user's choice must win (the warning
    // fires, but the run proceeds with gillespie). Regression
    // guard: a bug where the auto-match path always overrides
    // --backend would silently ignore the user's explicit choice.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle,
        &["--backend", "gillespie"]);
    let (backend, _dt) = read_sim_config(&output);
    assert_eq!(backend, "Gillespie", "explicit --backend must override provenance");
}

#[test]
fn standalone_params_use_chain_binomial_default_in_run_json() {
    // Standalone params file (no [provenance] block) + no --backend:
    // run.json must record chain_binomial (the CLI default, matching
    // `camdl fit`'s default — see the 2026-04-19 incident). Regression
    // guard on the "no leakage from prior runs" invariant.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let standalone = tmp.path().join("p.toml");
    std::fs::write(&standalone, "mu = 0.05\n").unwrap();
    let output = run_simulate_cas(&bin, tmp.path(), &standalone, &[]);
    let (backend, _dt) = read_sim_config(&output);
    assert_eq!(backend, "ChainBinomial", "standalone params must use chain_binomial default");
}

#[test]
fn explicit_dt_overrides_provenance_in_run_json() {
    // Provenance has dt=0.5; user explicitly passes --dt 0.25.
    // User's choice wins. Companion to the backend-override test —
    // dt and backend are paired fields and the override semantics
    // should be symmetric.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 0.5);
    let output = run_simulate_cas(&bin, tmp.path(), &mle,
        &["--dt", "0.25"]);
    let (backend, dt) = read_sim_config(&output);
    assert_eq!(dt, 0.25, "explicit --dt must override provenance dt");
    // And backend still auto-matches (we didn't pass --backend).
    assert_eq!(backend, "ChainBinomial");
}

// ── Producer side (gh#241 C3): `camdl fit` must record the METHOD's backend
//    in mle_params.toml, not the global `[config].backend`. The tests above
//    check the consumer (simulate reading a hand-written mle); this checks the
//    fit WRITES the right backend, so a `simulate --params` replay matches the
//    dynamics that produced θ̂.

fn seed_timing_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim/tests/fixtures/seed_timing_dated.ir.json")
}

#[test]
fn fit_records_method_backend_in_mle() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, "time\tcases\n\
        2020-03-15\t3\n2020-03-16\t6\n2020-03-17\t11\n2020-03-18\t18\n\
        2020-03-19\t27\n2020-03-20\t31\n2020-03-21\t28\n2020-03-22\t20\n\
        2020-03-23\t13\n2020-03-24\t8\n").unwrap();

    // The provenance must record the backend the METHOD actually fit on. (Before
    // gh#241 the producer copied the global `[config].backend` instead — a bug;
    // gh#241 PR E then relocated that field to `[synthetic].backend`, so a
    // global forward backend no longer exists to mis-record. This pins that the
    // recorded backend tracks the method.)
    let fit_toml = tmp.path().join("fit.toml");
    let body = format!(
        r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.1, 2.0], start = 0.6 }}
tau  = {{ bounds = [0.0, 60.0], start = 20.0 }}

[fixed]
gamma = 0.2
lambda = 2.0
w = 3.0
N0 = 1000
rho = 0.6
k = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 3
cooling = 0.7
"#,
        out = out.display(), ir = seed_timing_model().display(), data = data.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();

    let output = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("run").arg(&fit_toml)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn camdl fit run");
    assert!(output.status.success(),
        "fit run should succeed; stderr: {}", String::from_utf8_lossy(&output.stderr));

    let mle = walkdir(&out).into_iter()
        .find(|p| p.file_name().map(|s| s == "mle_params.toml").unwrap_or(false))
        .expect("fit must write an mle_params.toml");
    let contents = std::fs::read_to_string(&mle).unwrap();
    // The only `backend =` line in mle_params.toml is the [provenance] one.
    let line = contents.lines()
        .find(|l| l.trim_start().starts_with("backend ="))
        .unwrap_or_else(|| panic!("no [provenance].backend line in:\n{contents}"));
    assert!(line.contains("\"chain_binomial\""),
        "mle_params must record the method backend (chain_binomial); got: {line}");

    // The round trip the hand-written fixtures above cannot make: the file a
    // real fit writes passes `simulate --params`' tamper check. Its estimates
    // carry ten decimals, so this pins that the declared `content_hash` is
    // over the values as written (a hash over the unrounded estimate refused
    // every real fit's file).
    let sim = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &seed_timing_model().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--seed", "1", "--store", &tmp.path().join("sim").to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn camdl simulate");
    let stderr = String::from_utf8_lossy(&sim.stderr);
    assert!(sim.status.success(), "simulate --params on the fit's own file must run; stderr: {stderr}");
    assert!(!stderr.contains("edited after the fit"), "stderr: {stderr}");
    assert!(stderr.contains("backend auto-matched"), "the block was read: {stderr}");
}

fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p.clone()); }
                else { out.push(p); }
            }
        }
    }
    out
}
