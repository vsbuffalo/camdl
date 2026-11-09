//! An IF2 fit that lost started chains is recorded as `incomplete`, and a warm
//! start refuses it.
//!
//! IF2 skips a chain whose particle filter degenerates mid-run (the PF
//! degeneracy watchdog) and carries on with the rest. The fit used to record
//! nothing of it: `fit_state.toml` had no `n_good_chains`, `chain_starts.tsv`
//! still marked the lost chain `accepted`, and a fit left with one chain read
//! as `not_assessed` ("one chain: nothing to compare between chains") — which
//! a `from_mle` warm start accepts. A deliberately one-chain fit is still
//! `not_assessed` and still accepted.
//!
//! The fixture is chosen to lose chains: four particles on a ten-day SIR, six
//! chains under `starts = "lhs"`. With so few particles the filter's ESS sits
//! below the watchdog's floor for three windows in a row on some chains and not
//! others. Which chains are lost is a function of the seed; the test reads the
//! count off stderr and checks every artifact against it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test"))
}

fn camdl_bin() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
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
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

const CASES_TSV: &str =
    "time\tcases\n1\t12\n2\t15\n3\t19\n4\t24\n5\t30\n6\t36\n7\t42\n8\t48\n9\t52\n10\t55\n";

fn setup(dir: &Path) {
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES_TSV).unwrap();
}

/// `<name>.toml`: an IF2 fit of beta and gamma, `chains` chains of
/// `particles` particles, with `starts` as written.
fn fit_toml(dir: &Path, name: &str, chains: usize, particles: usize, starts: &str) {
    std::fs::write(dir.join(format!("{name}.toml")), format!(
        "output_dir = \"res_{name}\"\n[model]\ncamdl = \"m.ir.json\"\n\
         [data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.01, 3.0] }}\ngamma = {{ bounds = [0.01, 1.0] }}\n\
         [fixed]\nN0 = 1000\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = {chains}\n\
         particles = {particles}\niterations = 5\ncooling = 0.7\nstarts = {starts}\n"
    )).unwrap();
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The one leaf (`fit_state.toml`'s directory) under `root`.
fn leaf(root: &Path) -> PathBuf {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.file_name().is_some_and(|n| n == "fit_state.toml") {
                out.push(p.parent().unwrap().to_path_buf());
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    assert_eq!(found.len(), 1, "want one leaf under {}: {found:?}", root.display());
    found.pop().unwrap()
}

/// The value of `key = <int>` in `fit_state.toml`, `None` when absent.
fn state_int(leaf: &Path, key: &str) -> Option<usize> {
    let text = std::fs::read_to_string(leaf.join("fit_state.toml")).unwrap();
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key} = ")))
        .map(|v| v.trim().parse().unwrap())
}

/// The final (last-attempt) `status` of every chain in `chain_starts.tsv`.
fn final_statuses(leaf: &Path) -> Vec<(usize, String)> {
    let text = std::fs::read_to_string(leaf.join("chain_starts.tsv")).unwrap();
    let mut rows = text.lines().filter(|l| !l.starts_with('#')).skip(1);
    let mut last: std::collections::BTreeMap<usize, String> = Default::default();
    for l in rows.by_ref() {
        let cols: Vec<&str> = l.split('\t').collect();
        last.insert(cols[0].parse().unwrap(), cols[2].to_string());
    }
    last.into_iter().collect()
}

#[test]
fn an_if2_fit_that_lost_chains_is_incomplete_and_refused_as_a_warm_start() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "lossy", 6, 4, "\"lhs\"");
    let out = camdl(dir, &["fit", "run", "lossy.toml", "--seed", "6", "--progress", "none"]);
    let err = stderr(&out);
    assert!(out.status.success(), "the lossy fit itself completes: {err}");
    let lost = err.lines().filter(|l| l.contains("skipped") && l.contains("PF degenerate")).count();
    assert!((1..6).contains(&lost),
        "fixture precondition: seed 6 must lose some but not all of the 6 chains \
         (lost {lost}); pick another seed\n{err}");

    let leaf = leaf(&dir.join("res_lossy"));
    assert_eq!(state_int(&leaf, "n_chains"), Some(6));
    assert_eq!(state_int(&leaf, "n_good_chains"), Some(6 - lost),
        "fit_state.toml must record how many chains finished");
    let statuses = final_statuses(&leaf);
    assert_eq!(statuses.len(), 6, "{statuses:?}");
    let skipped = statuses.iter().filter(|(_, s)| s == "skipped").count();
    assert_eq!(skipped, lost,
        "chain_starts.tsv must mark each lost chain `skipped`, not `accepted`: {statuses:?}");

    let summary = camdl(dir, &["fit", "summary", &leaf.to_string_lossy()]);
    let text = String::from_utf8_lossy(&summary.stdout).into_owned() + &stderr(&summary);
    assert!(!text.contains("NOT ASSESSED"),
        "a fit that lost chains is not 'not assessed':\n{text}");
    if lost == 5 {
        assert!(text.contains("INCOMPLETE"), "one survivor of six is incomplete:\n{text}");
    }

    let starts = format!("{{ from_mle = \"{}\" }}", leaf.to_string_lossy());
    fit_toml(dir, "warm", 2, 50, &starts);
    let out = camdl(dir, &["fit", "run", "warm.toml", "--seed", "1", "--progress", "none"]);
    let err = stderr(&out);
    assert!(!out.status.success(), "a warm start from a fit that lost chains is refused:\n{err}");
    let line = format!("{lost} of its 6 IF2 chains were lost");
    assert!(err.contains("error: starts source") && err.contains(&line),
        "the refusal names how many chains were lost ({line:?}):\n{err}");
    assert!(err.contains("--allow-nonconverged-source"), "the refusal names the override:\n{err}");
    assert_eq!(leaf_count(&dir.join("res_warm")), 0, "a refused warm start writes no leaf");

    let out = camdl(dir, &["fit", "run", "warm.toml", "--seed", "1", "--progress", "none",
        "--allow-nonconverged-source"]);
    let err = stderr(&out);
    assert!(out.status.success(), "the override starts from it anyway:\n{err}");
    assert!(err.contains(&line) && err.contains("starting from it anyway"), "{err}");
}

/// A fit that asked for one chain lost nothing: it is `not_assessed`, and a
/// warm start accepts it as before.
#[test]
fn a_deliberate_one_chain_if2_fit_is_not_assessed_and_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "one", 1, 50, "\"single\"");
    let out = camdl(dir, &["fit", "run", "one.toml", "--seed", "1", "--progress", "none"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let leaf = leaf(&dir.join("res_one"));
    assert_eq!(state_int(&leaf, "n_good_chains"), Some(1));
    assert_eq!(final_statuses(&leaf), vec![(0, "accepted".to_string())]);
    let summary = camdl(dir, &["fit", "summary", &leaf.to_string_lossy()]);
    let text = String::from_utf8_lossy(&summary.stdout).into_owned() + &stderr(&summary);
    assert!(text.contains("NOT ASSESSED"), "{text}");

    let starts = format!("{{ from_mle = \"{}\" }}", leaf.to_string_lossy());
    fit_toml(dir, "warm", 2, 50, &starts);
    let out = camdl(dir, &["fit", "run", "warm.toml", "--seed", "1", "--progress", "none"]);
    assert!(out.status.success(), "a one-chain source is accepted: {}", stderr(&out));
}

fn leaf_count(root: &Path) -> usize {
    fn walk(d: &Path) -> usize {
        std::fs::read_dir(d).map(|it| it.flatten().map(|e| {
            let p = e.path();
            if p.is_dir() { walk(&p) } else { usize::from(p.ends_with("fit_state.toml")) }
        }).sum()).unwrap_or(0)
    }
    walk(root)
}
