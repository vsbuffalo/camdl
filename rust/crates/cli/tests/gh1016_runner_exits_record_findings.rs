//! gh#1016. A method run that fails inside its runner leaves the record every
//! other failure leaves: its findings in `diagnostics.json`, and a `failed`
//! `method_report` as the last line of `events.jsonl`, carrying the error the
//! command printed.
//!
//! Two runner exits are exercised. IF2's watchdog exit, when every chain's
//! particle filter degenerated (two particles on a ten-day SIR, the fixture of
//! `if2_lost_chains.rs` pushed past losing some chains to losing all), wrote
//! neither: its log ended on the last chain's finding. NLopt's error return,
//! when no chain reached a finite log-likelihood (an observed count above any
//! the model can reach, under a binomial whose `n` is the projection), wrote
//! the report but no `diagnostics.json`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 3.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ LIKELIHOOD
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

/// Run `fit run` on a fit of beta and gamma under `method`, with the
/// observation `likelihood` and `cases` rows; return the stderr and the
/// one leaf the run claimed.
fn failing_run(tag: &str, likelihood: &str, cases: &str, method: &str) -> (String, PathBuf, tempfile::TempDir) {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), MODEL.replace("LIKELIHOOD", likelihood)).unwrap();
    std::fs::write(dir.join("cases.tsv"), cases).unwrap();
    std::fs::write(dir.join(format!("{tag}.toml")), format!(
        "[model]\ncamdl = \"m.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.01, 3.0], start = 0.3 }}\n\
         gamma = {{ bounds = [0.01, 1.0], start = 0.1 }}\n[fixed]\nN0 = 1000\n[method]\n{method}"
    )).unwrap();
    let out = Command::new(&camdl)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
        .current_dir(dir)
        .args(["fit", "run", &format!("{tag}.toml"), "--seed", "1", "--progress", "none"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(1), "the run fails with status 1:\n{stderr}");
    let subdirs = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()
    };
    let leaves: Vec<PathBuf> = subdirs(&dir.join("results/fits")).iter()
        .flat_map(|fit| subdirs(fit))
        .flat_map(|m| subdirs(&m))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("seed_"))
        .collect();
    assert_eq!(leaves.len(), 1, "one claimed leaf: {leaves:?}");
    (stderr, leaves[0].clone(), tmp)
}

/// The leaf's `diagnostics.json` holds the findings its log recorded, in
/// order, and the log ends with a `failed` report whose error is the one
/// printed, which names `expect`.
fn assert_failure_recorded(leaf: &Path, stderr: &str, expect: &str) -> Vec<serde_json::Value> {
    let log = std::fs::read_to_string(leaf.join("events.jsonl")).unwrap();
    let events: Vec<serde_json::Value> = log.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let last = events.last().unwrap();
    assert_eq!(last["event"], "method_report", "the log ends with the method's report:\n{log}");
    assert_eq!(last["outcome"], "failed", "{last}");
    let error = last["error"].as_str().unwrap();
    assert!(error.contains(expect), "the recorded error names `{expect}`: {error}");
    assert!(stderr.contains(error), "the recorded error is the one printed:\n{error}\n---\n{stderr}");

    let path = leaf.join("diagnostics.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let findings: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let logged: Vec<&serde_json::Value> = events.iter()
        .filter(|e| e["event"] == "finding").map(|e| &e["finding"]).collect();
    assert_eq!(findings.iter().collect::<Vec<_>>(), logged,
        "diagnostics.json holds the findings the log recorded");
    findings
}

#[test]
fn if2s_every_chain_bailed_exit_records_its_findings_and_report() {
    let cases = "time\tcases\n1\t12\n2\t15\n3\t19\n4\t24\n5\t30\n6\t36\n7\t42\n8\t48\n9\t52\n10\t55\n";
    let (stderr, leaf, _tmp) = failing_run("if2", "poisson(rate = projected)", cases,
        "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 2\n\
         iterations = 5\ncooling = 0.7\nstarts = \"lhs\"\n");
    let findings = assert_failure_recorded(&leaf, &stderr, "bailed via the PF degeneracy watchdog");
    let bad_init = findings.iter().filter(|f| f["kind"]["type"] == "bad_init").count();
    assert_eq!(bad_init, 2, "one bad_init per lost chain: {findings:?}");
}

#[test]
fn an_nlopt_runner_error_records_its_findings_and_report() {
    let (stderr, leaf, _tmp) = failing_run("nl", "binomial(n = projected, p = 0.5)",
        "time\tcases\n1\t1000000\n2\t1000000\n",
        "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\nstarts = \"single\"\n");
    assert_failure_recorded(&leaf, &stderr, "reached no finite log-likelihood");
}
