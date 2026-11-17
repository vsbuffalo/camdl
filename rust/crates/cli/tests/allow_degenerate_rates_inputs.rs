//! `--allow-degenerate-rates` (and its fit-config spelling, `[config]
//! allow_degenerate_rates = true`) changes what every simulating command
//! computes: a rate that divides by an empty compartment is `0` under it and a
//! `NumericalCollapse` error without it. So each command must both *apply* it
//! and *key* it (proposal 2026-11-17, phase 1c):
//!
//! - `profile` takes the flag (through the shared `InferenceCore`) and the
//!   fit-config field, and must apply them to every cell;
//! - `pfilter` applies the flag and must put it in its run identity, or a run
//!   with it and a run without it share one stored leaf;
//! - commands that replay a stored fit (`fit predict`, `compare`) must replay
//!   it under the fit's own setting, as they already do for forcing
//!   extrapolation.
//!
//! The fixture's infection rate is `beta * S * S / (S + I)`: once the epidemic
//! has run its course `S + I` is 0, and the rate divides by zero.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build`", bin.display());
    bin
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir()
        .join(format!("camdl_degenerate_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// The collapsing model over `days` days, with `beta`/`gamma` bounded by
/// `[lo, 5]` (a high `lo` makes the collapse certain at every θ in the box).
fn model(days: u32, lo: f64) -> String {
    format!(
        r#"time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [{lo}, 5.0] ~ log_normal(mu = 1.0, sigma = 0.3)
  gamma : rate  in [{lo}, 5.0] ~ log_normal(mu = 1.0, sigma = 0.3)
}}
transitions {{
  infection : S --> I @ beta * S * S / (S + I)
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected + 0.5)
  }}
}}
init {{ S = 10  I = 1 }}
simulate {{ from = 0 'days  to = {days} 'days }}
"#
    )
}

/// Two cases a day for five days, then none, through day `days`.
fn cases(days: u32) -> String {
    let mut s = String::from("time\tcases\n");
    for t in 1..=days {
        s.push_str(&format!("{t}\t{}\n", if t <= 5 { 2 } else { 0 }));
    }
    s
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("out"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn text(o: &Output) -> String {
    format!(
        "status={:?}\nstdout:\n{}\nstderr:\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

// ── profile ─────────────────────────────────────────────────────────────────

/// The `loglik` column of the profile table on stdout, one per cell.
fn profile_logliks(o: &Output) -> Vec<f64> {
    let stdout = String::from_utf8_lossy(&o.stdout);
    let mut lines = stdout.lines().skip_while(|l| !l.starts_with("beta\tloglik"));
    let header: Vec<&str> = lines.next().unwrap_or_else(|| panic!("no profile table:\n{}", text(o)))
        .split('\t').collect();
    let col = header.iter().position(|c| *c == "loglik").unwrap();
    lines.map(|l| l.split('\t').nth(col).unwrap().parse::<f64>().unwrap()).collect()
}

fn run_profile(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "profile", "model.camdl", "--data", "cases.tsv", "--obs", "cases",
        "--sweep", "beta=1.0", "--starts", "from_params=start.toml",
        "--algorithm", "if2", "--particles", "20", "--iterations", "2",
        "--n-starts", "1", "--rw-sd", "gamma=0.02", "--seed", "1",
    ];
    args.extend_from_slice(extra);
    camdl(dir, &args)
}

fn profile_fixture(tag: &str) -> Tmp {
    let tmp = tempdir(tag);
    std::fs::write(tmp.0.join("model.camdl"), model(40, 0.1)).unwrap();
    std::fs::write(tmp.0.join("cases.tsv"), cases(40)).unwrap();
    std::fs::write(tmp.0.join("start.toml"), "gamma = 1.0\n").unwrap();
    tmp
}

/// Without the setting every particle of every cell reaches `S + I = 0` and
/// dies, so the cell has no finite log-likelihood; with it, the rate is 0 there
/// and the cell scores. Before the fix `profile` read the flag and never
/// applied it, so both runs printed `NaN`.
#[test]
fn profile_applies_allow_degenerate_rates() {
    let tmp = profile_fixture("profile_flag");
    let without = run_profile(&tmp.0, &[]);
    assert!(without.status.success(), "{}", text(&without));
    assert!(profile_logliks(&without).iter().all(|l| l.is_nan()),
        "the control: without the flag the collapsing model has no finite cell\n{}", text(&without));

    let with = run_profile(&tmp.0, &["--allow-degenerate-rates"]);
    assert!(with.status.success(), "{}", text(&with));
    let ll = profile_logliks(&with);
    assert!(!ll.is_empty() && ll.iter().all(|l| l.is_finite()),
        "with --allow-degenerate-rates every cell must score a finite log-likelihood, got {ll:?}\n{}",
        text(&with));
}

/// The same setting read from the `--fit` config's `[config]` table.
#[test]
fn profile_applies_the_fit_configs_allow_degenerate_rates() {
    let tmp = profile_fixture("profile_config");
    let toml = |flag: bool| format!(
        "[model]\ncamdl = \"model.camdl\"\n\n[config]\nallow_degenerate_rates = {flag}\n\n\
         [estimate]\ngamma = {{ bounds = [0.1, 5.0], start = 1.0 }}\nbeta = {{ bounds = [0.1, 5.0] }}\n\n\
         [fixed]\n");
    std::fs::write(tmp.0.join("off.toml"), toml(false)).unwrap();
    std::fs::write(tmp.0.join("on.toml"), toml(true)).unwrap();

    let off = run_profile(&tmp.0, &["--fit", "off.toml"]);
    assert!(off.status.success(), "{}", text(&off));
    assert!(profile_logliks(&off).iter().all(|l| l.is_nan()), "{}", text(&off));

    let on = run_profile(&tmp.0, &["--fit", "on.toml"]);
    assert!(on.status.success(), "{}", text(&on));
    let ll = profile_logliks(&on);
    assert!(!ll.is_empty() && ll.iter().all(|l| l.is_finite()),
        "`[config] allow_degenerate_rates = true` must reach every cell, got {ll:?}\n{}", text(&on));
}

// ── pfilter ─────────────────────────────────────────────────────────────────

/// Every stored pfilter leaf under `<dir>/out/pfilters/`: its run id and the
/// loglik its record holds.
fn pfilter_leaves(dir: &Path) -> Vec<(String, f64)> {
    fn walk(d: &Path, out: &mut Vec<(String, f64)>) {
        if let Ok(b) = std::fs::read_to_string(d.join("run.json")) {
            let v: serde_json::Value = serde_json::from_str(&b).unwrap();
            if v["kind"] == "pfilter" {
                out.push((v["run_id"].as_str().unwrap().to_string(),
                          v["inputs"]["loglik"].as_f64().unwrap()));
            }
        }
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            if e.path().is_dir() { walk(&e.path(), out) }
        }
    }
    let mut out = Vec::new();
    walk(&dir.join("out/pfilters"), &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Over ten days some particles reach `S + I = 0` and some do not, so the
/// filter scores both with and without the flag, to different values. They are
/// two computations and must be two leaves. Before the fix the flag was not in
/// pfilter's identity: the second run printed its own loglik and reported the
/// first's leaf as "cached", so the store held one value for both.
#[test]
fn pfilter_keys_allow_degenerate_rates() {
    let tmp = tempdir("pfilter");
    std::fs::write(tmp.0.join("model.camdl"), model(10, 0.1)).unwrap();
    std::fs::write(tmp.0.join("cases.tsv"), cases(10)).unwrap();
    let run = |extra: &[&str]| {
        let mut args = vec![
            "pfilter", "model.camdl", "--param", "beta=1", "--param", "gamma=1",
            "--data", "cases.tsv", "--particles", "50", "--seed", "1",
        ];
        args.extend_from_slice(extra);
        let o = camdl(&tmp.0, &args);
        assert!(o.status.success(), "{}", text(&o));
        String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().unwrap()
    };
    let without = run(&[]);
    let with = run(&["--allow-degenerate-rates"]);
    assert!((without - with).abs() > 1e-6,
        "the control: the flag changes this filter's loglik ({without} vs {with})");

    let leaves = pfilter_leaves(&tmp.0);
    assert_eq!(leaves.len(), 2,
        "a run with --allow-degenerate-rates and one without must be two leaves, got {leaves:?}");
    let mut stored: Vec<f64> = leaves.iter().map(|l| l.1).collect();
    stored.sort_by(f64::total_cmp);
    let mut printed = vec![without, with];
    printed.sort_by(f64::total_cmp);
    for (s, p) in stored.iter().zip(&printed) {
        assert!((s - p).abs() < 1e-3,
            "each leaf holds its own run's loglik: stored {stored:?}, printed {printed:?}");
    }
}
