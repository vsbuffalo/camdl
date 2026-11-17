//! D72, end to end: `camdl pfilter` with a second stream bound to a file whose
//! only row is `NA` gives the same filter as the run without it.
//!
//! The `daily` stream's single row (`11  NA`) declares the period [10, 11), so
//! binding it adds two stops to the filter's axis: the period's start at
//! t = 10 and its close at t = 11, neither carrying an observation. With the
//! same seed, the printed log-likelihood, the trace's per-step
//! log-likelihood increments, ESS and latent-state summaries at the weekly
//! times, the saved final particle states and the prequential log scores are
//! the ones the weekly-only run gives. The trace's predictive quantiles and
//! the prequential CRPS and PIT are not compared: they come from the
//! diagnostic draw stream, which the extra stops' own predictive draws
//! advance.
//!
//! End-to-end via the release `camdl` binary and the built `camdlc.exe`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let bin = Path::new(manifest).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

fn camdlc() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let p = Path::new(manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  weekly {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases ~ poisson(rate = projected)
  }
  daily {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 35 'days }
"#;

const WEEKLY: &str = "time\tcases\n7\t40\n14\t90\n21\t150\n28\t140\n35\t60\n";
const DAILY_NA: &str = "time\tcases\n11\tNA\n";

fn fixture(dir: &Path) {
    std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
    let out = Command::new(camdlc()).env_remove("CLICOLOR_FORCE")
        .arg(dir.join("sir.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("weekly.tsv"), WEEKLY).unwrap();
    std::fs::write(dir.join("daily.tsv"), DAILY_NA).unwrap();
}

fn pfilter(dir: &Path, tag: &str, seed: u64, data: &[&str]) -> Output {
    let mut cmd = Command::new(camdl_bin());
    cmd.env_remove("CLICOLOR_FORCE")
        .arg("pfilter").arg("sir.ir.json")
        .args(["--param", "beta=0.4", "--param", "gamma=0.2"])
        .args(["--particles", "1000", "--seed", &seed.to_string()]);
    for d in data {
        cmd.args(["--data", d]);
    }
    let out = cmd
        .args(["--trace", &format!("{tag}.trace.tsv")])
        .args(["--save-final-state", &format!("{tag}.state.tsv")])
        .args(["--save-prequential", &format!("{tag}.preq")])
        .args(["--no-store", "--no-progress"])
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(), "pfilter {tag}: {}", String::from_utf8_lossy(&out.stderr));
    out
}

/// A TSV as rows keyed by its first column, each row a map column → cell.
fn table(path: &Path) -> HashMap<String, HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    lines.map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        let row = header.iter().zip(&cells).map(|(h, c)| (h.to_string(), c.to_string())).collect();
        (cells[0].to_string(), row)
    }).collect()
}

#[test]
fn an_all_na_stream_leaves_the_pfilter_run_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    for seed in [1u64, 2, 3, 4, 5, 6] {
        let base = pfilter(dir, &format!("base{seed}"), seed, &["weekly=weekly.tsv"]);
        let with = pfilter(dir, &format!("with{seed}"), seed, &["weekly=weekly.tsv", "daily=daily.tsv"]);
        assert_eq!(String::from_utf8_lossy(&with.stdout), String::from_utf8_lossy(&base.stdout),
            "seed {seed}: the printed log-likelihood differs");

        let tb = table(&dir.join(format!("base{seed}.trace.tsv")));
        let tw = table(&dir.join(format!("with{seed}.trace.tsv")));
        assert!(tw.contains_key("10") && tw.contains_key("11"), "the NA row adds stops at 10 and 11");
        assert_eq!(tw["10"]["ll_increment"], "0.0000");
        assert_eq!(tw["11"]["ll_increment"], "0.0000");
        for (t, rb) in &tb {
            for col in ["ll_increment", "ESS", "state_mean", "state_q05", "state_q50", "state_q95"] {
                assert_eq!(tw[t][col], rb[col], "seed {seed}, t = {t}: trace column {col}");
            }
        }

        let sb = std::fs::read_to_string(dir.join(format!("base{seed}.state.tsv"))).unwrap();
        let sw = std::fs::read_to_string(dir.join(format!("with{seed}.state.tsv"))).unwrap();
        assert_eq!(sw, sb, "seed {seed}: the saved final particle states differ");

        let pb = table(&dir.join(format!("base{seed}.preq.tsv")));
        let pw = table(&dir.join(format!("with{seed}.preq.tsv")));
        assert_eq!(pw.len(), pb.len(), "seed {seed}: prequential steps");
        for (t, rb) in &pb {
            assert_eq!(pw[t]["log_score"], rb["log_score"], "seed {seed}, t = {t}: log score");
            assert_eq!(pw[t]["ess"], rb["ess"], "seed {seed}, t = {t}: ESS");
        }
    }
}
