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

mod fit_store_support;

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

/// The run store every command here writes to: `camdl-store` in the work
/// directory.
fn store(work: &Path) -> PathBuf {
    work.join("camdl-store")
}

/// The one stored fit run of the fit file `toml`: the run whose `fit.toml`
/// (the file as given) is that file's bytes.
fn leaf(work: &Path, toml: &str) -> PathBuf {
    let text = std::fs::read(work.join(toml)).unwrap();
    let hits: Vec<PathBuf> = fit_store_support::runs(&store(work))
        .into_iter()
        .filter(|r| std::fs::read(r.join("fit.toml")).ok().as_deref() == Some(&text[..]))
        .collect();
    assert_eq!(hits.len(), 1, "expected one fit run of {toml}: {hits:?}");
    hits.into_iter().next().unwrap()
}

struct Output {
    stdout: String,
    stderr: String,
    /// The `events.jsonl` of the leaf a fit run wrote, for the runs whose
    /// recorded facts include their events; empty otherwise.
    events: String,
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
        assert!(matches!(cols[1], "stdout" | "stderr" | "events"), "facts.tsv:{}: stream {}", i + 1, cols[1]);
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
    // Files only: the fixture is flat, and a directory beside it (an editor's
    // or tool's state folder) is not part of it.
    for entry in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        if !entry.path().is_file() {
            continue;
        }
        std::fs::copy(entry.path(), work.join(entry.file_name())).unwrap();
    }

    let run = |args: &[String]| -> Output {
        let out = Command::new(&camdl)
            .env_remove("CLICOLOR_FORCE")
            // The compiled-IR cache is the work directory's own, so whether a
            // run prints `compiling` or `cached IR` depends on the runs before
            // it here, not on what other processes left in `~/.cache`.
            .env("CAMDL_IR_CACHE_DIR", work.join(".ir-cache"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            // The work directory's store, named relative to it, so output
            // names it as `camdl-store`.
            .env("CAMDL_STORE", "camdl-store")
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        Output {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            events: String::new(),
        }
    };
    // A fit run's output with its leaf's event log beside it.
    let with_events = |mut o: Output, leaf: &Path| -> Output {
        o.events = std::fs::read_to_string(leaf.join("events.jsonl"))
            .unwrap_or_else(|e| panic!("{}/events.jsonl: {e}", leaf.display()));
        o
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
    let stuck = run(&fit("pmmh_stuck.toml", "8", &[]));
    let stuck_leaf = leaf(&work, "pmmh_stuck.toml");
    runs.push(("stuck", with_events(stuck, &stuck_leaf)));
    // The same run again is a cache hit: no method runs, and the stored
    // leaf's event log is the one the run that computed it wrote.
    let log_before = std::fs::read(stuck_leaf.join("events.jsonl")).unwrap();
    runs.push(("stuck.cached", run(&fit("pmmh_stuck.toml", "8", &[]))));
    assert_eq!(std::fs::read(stuck_leaf.join("events.jsonl")).unwrap(), log_before,
        "a cache hit rewrote the stored leaf's events.jsonl");
    let stuck_fit = stuck_leaf.clone();
    runs.push(("stuck.summary", run(&a(&["fit", "summary", stuck_fit.to_str().unwrap()]))));
    runs.push(("converged", run(&fit("pmmh_converged.toml", "1", &["--label", "converged"]))));
    let base = leaf(&work, "pmmh_converged.toml");
    runs.push(("converged.summary", run(&a(&["fit", "summary", base.to_str().unwrap()]))));
    runs.push(("resumed", run(&fit("pmmh_resumed.toml", "1", &["--resume", base.to_str().unwrap()]))));
    runs.push(("stuck.excluded", run(&a(&["fit", "summary", stuck_fit.to_str().unwrap(), "--exclude-chains", "2"]))));
    runs.push(("pgas", run(&fit("pgas.toml", "1", &[]))));
    let pgas_fit = leaf(&work, "pgas.toml");
    runs.push(("pgas.summary", run(&a(&["fit", "summary", pgas_fit.to_str().unwrap()]))));
    runs.push(("pgas.explain", run(&a(&["fit", "summary", pgas_fit.to_str().unwrap(), "--explain", "--parameters"]))));
    runs.push(("if2", run(&fit("if2.toml", "1", &["--label", "if2"]))));
    runs.push(("if2.summary", run(&a(&["fit", "summary", "@if2"]))));
    runs.push(("nlopt", run(&fit("nlopt.toml", "1", &[]))));
    let nlopt_fit = leaf(&work, "nlopt.toml");
    runs.push(("nlopt.summary", run(&a(&["fit", "summary", nlopt_fit.to_str().unwrap()]))));
    let failed = run(&fit("fail.toml", "1", &[]));
    let fail_fit = leaf(&work, "fail.toml");
    runs.push(("fail", with_events(failed, &fail_fit)));
    runs.push(("fail.summary", run(&a(&["fit", "summary", fail_fit.to_str().unwrap()]))));
    runs.push(("pfilter_fit", run(&fit("pfilter.toml", "1", &[]))));
    let pfilter_fit = leaf(&work, "pfilter.toml");
    runs.push(("pfilter_fit.summary", run(&a(&["fit", "summary", pfilter_fit.to_str().unwrap()]))));
    runs.push(("compare", run(&a(&["compare", "@if2", "@converged", "--draws", "1", "--allow-mixed-provenance"]))));
    let mle = leaf(&work, "if2.toml").join("mle_params.toml");
    runs.push(("pfilter", run(&a(&[
        "pfilter", "sir.camdl", "--fit", "if2.toml", "--params", mle.to_str().unwrap(),
        "--particles", "200", "--seed", "1",
    ]))));

    // The other commands, each with one run that refuses its input, so an
    // error line is a recorded fact too.
    let none = ["--progress", "none"];
    let mle_s = mle.to_str().unwrap();
    let pf = |extra: &[&str]| {
        let mut v = a(&["pfilter", "sir.camdl", "--fit", "if2.toml", "--params", mle_s]);
        v.extend(a(extra));
        v.extend(a(&none));
        v
    };
    runs.push(("pfilter.replicates", run(&pf(&["--particles", "100", "--seed", "2", "--replicates", "3"]))));
    runs.push(("pfilter.files", run(&pf(&[
        "--particles", "100", "--seed", "3", "--trace", "tr.tsv", "--pf-health", "h.tsv",
        "--save-final-state", "fs.tsv", "--save-paths", "p.tsv", "--n-paths", "2",
        "--save-filtering", "f.tsv", "--save-prequential", "preq", "-o", "ll.toml",
    ]))));
    runs.push(("pfilter.no_data", run(&a(&["pfilter", "sir.camdl", "--params", mle_s, "--particles", "100"]))));
    // Seed 12: one start at the lowest beta scores -inf, so the run shows the
    // non-finite warning and a cell completed by one of its two starts.
    let profile = a(&[
        "profile", "sir.camdl", "--fit", "if2.toml", "--sweep", "beta=lin(0.3,0.6,3)", "--particles", "100",
        "--iterations", "3", "--n-starts", "2", "--seed", "12", "--rw-sd", "auto", "--progress", "none",
    ]);
    runs.push(("profile", run(&profile)));
    runs.push(("profile.cached", run(&profile)));
    runs.push(("profile.refused", run(&a(&["profile", "sir.camdl", "--fit", "if2.toml", "--sweep", "beta=lin(0.3,0.6,3)"]))));
    runs.push(("compare.refused", run(&a(&["compare", "@if2"]))));
    runs.push(("predict", run(&a(&["predict", "@converged", "--mode", "free_forward", "--n-draws", "5"]))));
    runs.push(("predict.refused", run(&a(&["predict", "@nope", "--mode", "free_forward"]))));
    let sim = |extra: &[&str]| {
        let mut v = a(&["simulate", "sir.camdl", "--params", "truth.toml"]);
        v.extend(a(extra));
        v
    };
    runs.push(("simulate", run(&sim(&["--seed", "1"]))));
    runs.push(("simulate.stdout", run(&sim(&["--seed", "1", "--stdout"]))));
    runs.push(("simulate.seeds", run(&sim(&["--seeds", "1:3", "--obs-dir", "obsd", "-o", "traj.tsv", "--progress", "none"]))));
    runs.push(("simulate.dry", run(&sim(&["--seed", "1", "--dry-run"]))));
    runs.push(("simulate.refused", run(&a(&["simulate", "sir.camdl", "--param", "zzz=1"]))));
    runs.push(("simulate.events", run(&a(&[
        "simulate", "lineage.camdl", "--param", "beta=0.8", "--param", "gamma=0.1", "--seed", "3",
        "--event-log", "ev.tsv", "--tsv", "--backend", "gillespie",
    ]))));
    let sim_id = runs.iter().find(|(n, _)| *n == "simulate").unwrap().1.stderr.lines()
        .find_map(|l| l.trim().strip_prefix("camdl cat ").map(str::to_string))
        .expect("simulate names its leaf as `camdl cat <id>`");
    runs.push(("lineage.realize", run(&a(&["lineage", "realize", "ev.tsv", "--identity-seed", "7", "-o", "ll.tsv"]))));
    runs.push(("lineage.tree", run(&a(&["lineage", "tree", "ll.tsv", "--scheme", "flat:1.0", "--output", "tree.nwk"]))));
    runs.push(("lineage.sojourn", run(&a(&["lineage", "sojourn", "ll.tsv", "--compartment", "1", "--output", "soj.tsv"]))));
    runs.push(("lineage.cohort", run(&a(&["lineage", "cohort", "ll.tsv", "--event", "infection", "--window", "7", "--output", "coh.tsv"]))));
    runs.push(("lineage.refused", run(&a(&["lineage", "tree", "nope.tsv", "--scheme", "flat:1.0"]))));
    runs.push(("list", run(&a(&["list"]))));
    let if2_leaf = leaf(&work, "if2.toml");
    runs.push(("show", run(&a(&["show", if2_leaf.to_str().unwrap()]))));
    runs.push(("cat", run(&a(&["cat", &sim_id]))));
    runs.push(("show.refused", run(&a(&["show", "zzzzzzzz"]))));
    runs.push(("fit.diff", run(&a(&["fit", "diff", "if2.toml", "pmmh_converged.toml"]))));
    runs.push(("fit.diff.refused", run(&a(&["fit", "diff", "if2.toml", "nope.toml"]))));
    runs.push(("fit.table", run(&a(&["fit", "table", "camdl-store"]))));
    runs.push(("fit.table.md", run(&a(&["fit", "table", "camdl-store", "--format", "md"]))));

    let facts = read_facts();
    assert!(facts.len() > 100, "facts.tsv holds {} facts; the reader is broken", facts.len());
    let mut missing = Vec::new();
    let mut failed_runs = std::collections::BTreeSet::new();
    for f in &facts {
        let Some((_, out)) = runs.iter().find(|(name, _)| *name == f.run) else {
            panic!("facts.tsv:{}: no run named `{}`", f.line_no, f.run);
        };
        let text = match f.stream.as_str() {
            "stdout" => &out.stdout,
            "stderr" => &out.stderr,
            _ => &out.events,
        };
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
            .map(|(name, o)| format!("──── {name} stdout ────\n{}\n──── {name} stderr ────\n{}\n──── {name} events ────\n{}",
                o.stdout, o.stderr, o.events))
            .collect();
        panic!("{} recorded fact(s) not shown:\n{}\n\n{}", missing.len(), missing.join("\n"), shown.join("\n"));
    }

    run_equals_loader(&work, |args| run(&a(args)));
}

/// The report `fit run` built from memory equals the one `fit summary` builds
/// from the leaf the run wrote (proposal
/// `docs/dev/proposals/2026-11-08-cli-reporting.md`, "the run-equals-loader
/// test"). The run's is the `method_report` line of the leaf's
/// `events.jsonl`; the loader's is `fit summary --format json`'s. Both
/// spell non-finite numbers out, so NaN cannot equal ∞ or a missing value;
/// equality is exact. Every completed method the fixtures run is compared,
/// the resumed PMMH leaf included. The failed fit's log ends in a `failed`
/// outcome, with no report on either side.
fn run_equals_loader(work: &Path, run: impl Fn(&[&str]) -> Output) {
    let mut from_runs: Vec<(PathBuf, serde_json::Value)> = Vec::new();
    let mut failed = 0;
    for leaf in fit_store_support::runs(&store(work)) {
        let log = leaf.join("events.jsonl");
        let text = std::fs::read_to_string(&log).unwrap_or_else(|e| panic!("{}: {e}", log.display()));
        // Each stored line is `{at, event, format, n}`, the event itself in `event`.
        let line: serde_json::Value = serde_json::from_str(text.lines().last().expect("a non-empty log"))
            .expect("one JSON event per line");
        let last = &line["event"];
        assert_eq!(last["event"], "method_report", "{} ends with its report", log.display());
        match last["outcome"].as_str() {
            Some("completed") => from_runs.push((leaf, last["report"].clone())),
            Some("failed") => failed += 1,
            other => panic!("{}: outcome {other:?}", log.display()),
        }
    }
    let mut methods: Vec<String> = from_runs.iter()
        .map(|(_, r)| r["method"].as_str().unwrap().to_string()).collect();
    methods.sort();
    assert_eq!(methods, ["if2", "nl-sbplx", "pfilter", "pgas", "pmmh", "pmmh", "pmmh"],
        "every fixture method's run recorded its report");
    assert_eq!(failed, 1, "the failed fit's log ends in its failure");

    let mut mismatched = Vec::new();
    for (path, a) in &from_runs {
        // The leaf as the run's report names it (relative to the work
        // directory), so `stored.leaf` is spelled alike on both sides.
        let leaf = a["stored"]["leaf"].as_str()
            .unwrap_or_else(|| panic!("{}: a completed report names its leaf", path.display()));
        assert!(work.join(leaf) == *path, "{} names its leaf {leaf}", path.display());
        let out = run(&["fit", "summary", leaf, "--format", "json"]);
        let doc: serde_json::Value = serde_json::from_str(&out.stdout)
            .unwrap_or_else(|e| panic!("fit summary {leaf} --format json: {e}\n{}", out.stderr));
        let from_leaf = doc["methods"].as_array().expect("the document's methods");
        assert_eq!(from_leaf.len(), 1, "one report from {leaf}: {}", out.stderr);
        let b = &from_leaf[0];
        if a != b {
            let mut diffs = Vec::new();
            json_diff("", a, b, &mut diffs);
            mismatched.push(format!("{leaf}:\n  {}", diffs.join("\n  ")));
        }
    }
    assert!(mismatched.is_empty(),
        "the report `fit run` built differs from the one `fit summary` builds from its leaf \
         (run ≠ leaf):\n{}", mismatched.join("\n"));
}

/// Each JSON pointer where `a` and `b` differ, with both values.
fn json_diff(at: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            for k in keys {
                json_diff(&format!("{at}/{k}"), x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null), out);
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                json_diff(&format!("{at}/{i}"), p, q, out);
            }
        }
        _ if a != b => out.push(format!("{at}: run {a} ≠ leaf {b}")),
        _ => {}
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
