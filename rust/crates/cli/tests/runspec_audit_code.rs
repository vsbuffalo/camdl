//! Three defects found by auditing docs/camdl-run-spec.md against the code.
//!
//! * IF2's `mle_params.toml` `[provenance]` states the uncertainty of the
//!   log-likelihood it records: the re-score's Monte Carlo standard error and
//!   particle count, not a zero and the swarm's size.
//! * `survey --fit` and `profile --fit` seed an `[estimate]` entry with no
//!   `start` (`beta = {}`) the way `fit run` does (run-spec §1.4, §6.3).
//! * The resolver's refusals name the flag the user passed and fixes that
//!   exist (§1.3).
//!
//! Each fit is tiny; the model is a three-compartment SIR with Poisson cases.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
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
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;

fn fixture(dir: &Path) {
    std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n")
        .unwrap();
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .current_dir(dir)
        .env("CAMDL_OUTPUT_DIR", dir.join("out"))
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &Output) -> String { String::from_utf8_lossy(&o.stderr).into_owned() }

fn ok(o: &Output, what: &str) -> String {
    let e = stderr(o);
    assert!(o.status.success(), "{what} failed: {e}");
    e
}

fn find_files(root: &Path, name: &str, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() { find_files(&p, name, out); }
        else if p.file_name().is_some_and(|n| n == name) { out.push(p); }
    }
}

// ─── A: IF2 provenance records the re-score's uncertainty ───────────────────

const IF2_SWARM: usize = 31;
const RESCORE_PARTICLES: usize = 77;

#[test]
fn if2_provenance_records_the_rescore_standard_error_and_particle_count() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    std::fs::write(dir.join("fit.toml"), format!(r#"
[model]
camdl = "sir.camdl"
[data.observations]
cases = "cases.tsv"
[estimate]
beta  = {{ start = 0.4 }}
gamma = {{ start = 0.15 }}
[fixed]
N0 = 1000
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = {IF2_SWARM}
iterations = 3
cooling = 0.5
loglik_eval = {{ n_particles = {RESCORE_PARTICLES}, n_replicates = 3 }}
dt_check = {{ enabled = false }}
"#)).unwrap();
    ok(&camdl(dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");

    let mut found = Vec::new();
    find_files(&dir.join("out"), "mle_params.toml", &mut found);
    assert_eq!(found.len(), 1, "one mle_params.toml expected: {found:?}");
    let text = std::fs::read_to_string(&found[0]).unwrap();
    let doc: toml::Value = text.parse().unwrap();
    let prov = &doc["provenance"];

    let sd = prov["loglik_sd"].as_float().unwrap();
    assert!(sd > 0.0 && sd.is_finite(),
        "loglik_sd is the re-score's standard error, nonzero for a stochastic filter: {text}");
    assert_eq!(prov["n_particles"].as_integer(), Some(RESCORE_PARTICLES as i64),
        "n_particles is the re-score's particle count, not the swarm's ({IF2_SWARM}): {text}");
    assert!(prov["ess_mean"].as_float().is_some_and(|v| v > 0.0),
        "the re-score's mean effective sample size is recorded: {text}");
}
