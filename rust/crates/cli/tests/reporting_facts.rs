//! Nothing a fixture's output shows is dropped.
//!
//! `fixtures/reporting/facts.tsv` records, for each command below, the facts
//! its output shows as (label, value, stream). This test runs every command in
//! a fresh copy of `fixtures/reporting/` and fails naming each fact it cannot
//! find, with the stream it searched. The file's header gives the matching
//! rule.
//!
//! The point is the reporting rework (proposal
//! `docs/dev/proposals/2026-11-08-cli-reporting.md`, "How we know nothing was
//! dropped"): output moves between printers, merges and gains labels, and a
//! fact that silently stops being printed is the failure it guards against. A
//! change to what is shown is an edit to `facts.tsv` in the same commit, so it
//! is reviewed as one.
//!
//! The runs are small (about ten seconds in all) and fixed-seed, so every
//! recorded value is deterministic.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust` (gh#105)", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The single directory under `dir` whose name starts with `prefix`.
fn only_child(dir: &Path, prefix: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    assert_eq!(hits.len(), 1, "expected one `{prefix}*` under {}: {hits:?}", dir.display());
    hits.into_iter().next().unwrap()
}

/// The leaf `results/fits/<fit>-*/<method>-*/seed_*`.
fn leaf(work: &Path, fit: &str, method: &str) -> PathBuf {
    let fit_dir = only_child(&work.join("results/fits"), &format!("{fit}-"));
    let method_dir = only_child(&fit_dir, &format!("{method}-"));
    only_child(&method_dir, "seed_")
}

struct Output {
    stdout: String,
    stderr: String,
}

struct Fact {
    line_no: usize,
    run: String,
    stream: String,
    block: bool,
    label: String,
    value: String,
}

fn read_facts() -> Vec<Fact> {
    let path = manifest().join("tests/fixtures/reporting/facts.tsv");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut facts = Vec::new();
    let mut header_seen = false;
    for (i, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5, "facts.tsv:{}: expected 5 tab-separated columns: {line}", i + 1);
        if !header_seen {
            assert_eq!(cols, ["run", "stream", "scope", "label", "value"], "facts.tsv header");
            header_seen = true;
            continue;
        }
        assert!(cols[1] == "stdout" || cols[1] == "stderr", "facts.tsv:{}: stream {}", i + 1, cols[1]);
        assert!(cols[2] == "line" || cols[2] == "block", "facts.tsv:{}: scope {}", i + 1, cols[2]);
        facts.push(Fact {
            line_no: i + 1,
            run: cols[0].to_string(),
            stream: cols[1].to_string(),
            block: cols[2] == "block",
            label: cols[3].to_string(),
            value: cols[4].to_string(),
        });
    }
    facts
}

/// Whether `text` shows `value` under `label`: on a line containing the
/// label, or for a block fact on a later line before the next blank line.
fn holds(text: &str, label: &str, value: &str, block: bool) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    lines.iter().enumerate().filter(|(_, l)| l.contains(label)).any(|(i, l)| {
        if l.contains(value) {
            return true;
        }
        block && lines[i + 1..].iter().take_while(|l| !l.trim().is_empty()).any(|l| l.contains(value))
    })
}

#[test]
fn every_recorded_fact_is_shown() {
    let camdl = camdl();
    let camdlc = camdlc();
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_reporting_facts_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for entry in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        std::fs::copy(entry.path(), work.join(entry.file_name())).unwrap();
    }

    let run = |args: &[String]| -> Output {
        let out = Command::new(&camdl)
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        Output {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    };
    let a = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let fit = |toml: &str, seed: &str, extra: &[&str]| {
        let mut v = a(&["fit", "run", toml, "--seed", seed, "--progress", "none"]);
        v.extend(a(extra));
        v
    };

    // In order: the resumed run reads the converged run's leaf, `compare`
    // and `pfilter` read the IF2 and converged fits. Each fit's `fit summary`
    // follows its run.
    let mut runs: Vec<(&str, Output)> = Vec::new();
    runs.push(("stuck", run(&fit("pmmh_stuck.toml", "8", &[]))));
    let stuck_fit = only_child(&work.join("results/fits"), "pmmh_stuck-");
    runs.push(("stuck.summary", run(&a(&["fit", "summary", stuck_fit.to_str().unwrap()]))));
    runs.push(("converged", run(&fit("pmmh_converged.toml", "1", &["--label", "converged"]))));
    let converged_fit = only_child(&work.join("results/fits"), "pmmh_converged-");
    runs.push(("converged.summary", run(&a(&["fit", "summary", converged_fit.to_str().unwrap()]))));
    let base = leaf(&work, "pmmh_converged", "pmmh");
    runs.push(("resumed", run(&fit("pmmh_resumed.toml", "1", &["--resume", base.to_str().unwrap()]))));
    runs.push(("stuck.excluded", run(&a(&["fit", "summary", stuck_fit.to_str().unwrap(), "--exclude-chains", "2"]))));
    runs.push(("pgas", run(&fit("pgas.toml", "1", &[]))));
    let pgas_fit = only_child(&work.join("results/fits"), "pgas-");
    runs.push(("pgas.summary", run(&a(&["fit", "summary", pgas_fit.to_str().unwrap()]))));
    runs.push(("pgas.explain", run(&a(&["fit", "summary", pgas_fit.to_str().unwrap(), "--explain", "--parameters"]))));
    runs.push(("if2", run(&fit("if2.toml", "1", &["--label", "if2"]))));
    runs.push(("if2.summary", run(&a(&["fit", "summary", "@if2"]))));
    runs.push(("nlopt", run(&fit("nlopt.toml", "1", &[]))));
    let nlopt_fit = only_child(&work.join("results/fits"), "nlopt-");
    runs.push(("nlopt.summary", run(&a(&["fit", "summary", nlopt_fit.to_str().unwrap()]))));
    runs.push(("fail", run(&fit("fail.toml", "1", &[]))));
    let fail_fit = only_child(&work.join("results/fits"), "fail-");
    runs.push(("fail.summary", run(&a(&["fit", "summary", fail_fit.to_str().unwrap()]))));
    runs.push(("pfilter_fit", run(&fit("pfilter.toml", "1", &[]))));
    let pfilter_fit = only_child(&work.join("results/fits"), "pfilter-");
    runs.push(("pfilter_fit.summary", run(&a(&["fit", "summary", pfilter_fit.to_str().unwrap()]))));
    runs.push(("compare", run(&a(&["compare", "@if2", "@converged", "--draws", "1"]))));
    let mle = leaf(&work, "if2", "if2").join("mle_params.toml");
    runs.push(("pfilter", run(&a(&[
        "pfilter", "sir.camdl", "--fit", "if2.toml", "--params", mle.to_str().unwrap(),
        "--particles", "200", "--seed", "1",
    ]))));

    let facts = read_facts();
    assert!(facts.len() > 100, "facts.tsv holds {} facts; the reader is broken", facts.len());
    let mut missing = Vec::new();
    let mut failed_runs = std::collections::BTreeSet::new();
    for f in &facts {
        let Some((_, out)) = runs.iter().find(|(name, _)| *name == f.run) else {
            panic!("facts.tsv:{}: no run named `{}`", f.line_no, f.run);
        };
        let text = if f.stream == "stdout" { &out.stdout } else { &out.stderr };
        if !holds(text, &f.label, &f.value, f.block) {
            missing.push(format!(
                "facts.tsv:{}: {} {}: `{}` under `{}` ({})",
                f.line_no, f.run, f.stream, f.value, f.label, if f.block { "block" } else { "line" }));
            failed_runs.insert(f.run.clone());
        }
    }
    for (name, _) in &runs {
        assert!(facts.iter().any(|f| f.run == *name), "run `{name}` has no recorded fact");
    }
    if !missing.is_empty() {
        let shown: Vec<String> = runs.iter()
            .filter(|(name, _)| failed_runs.contains(*name))
            .map(|(name, o)| format!("──── {name} stdout ────\n{}\n──── {name} stderr ────\n{}", o.stdout, o.stderr))
            .collect();
        panic!("{} recorded fact(s) not shown:\n{}\n\n{}", missing.len(), missing.join("\n"), shown.join("\n"));
    }
}

/// The matching rule, on its own: a line fact needs label and value on one
/// line; a block fact may find the value later in the label's section, but
/// not past a blank line.
#[test]
fn the_matching_rule() {
    let text = "pfilter noise check (20 particles)\n  ⚠ ceiling below target\n\nchain 1: 0.0%\n";
    assert!(holds(text, "chain 1:", "0.0%", false));
    assert!(!holds(text, "pfilter noise check", "ceiling below target", false));
    assert!(holds(text, "pfilter noise check", "ceiling below target", true));
    assert!(!holds(text, "pfilter noise check", "chain 1:", true), "a block ends at a blank line");
    assert!(!holds(text, "chain 2:", "0.0%", true));
}
