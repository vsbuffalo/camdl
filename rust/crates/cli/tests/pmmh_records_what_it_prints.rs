//! A PMMH run records the facts it prints.
//!
//! `fit summary` can only show what a run showed if the run wrote it down
//! (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`, "Every fact the
//! run-end view prints is recorded"). This runs the stuck-chain fixture and
//! its resumed continuation and checks each new record against the line the
//! same run printed: the opening block (`run.json` `inputs.header`), the
//! noise check's mean (`fit_state.toml` `pf_noise.ll_mean`), the per-chain
//! warm-up, the sampling plan and the cost (`pmmh_summary.json`).

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

fn only_child(dir: &Path, prefix: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    assert_eq!(hits.len(), 1, "expected one `{prefix}*` under {}: {hits:?}", dir.display());
    hits.into_iter().next().unwrap()
}

fn leaf(work: &Path, fit: &str) -> PathBuf {
    let fit_dir = only_child(&work.join("results/fits"), &format!("{fit}-"));
    only_child(&only_child(&fit_dir, "pmmh-"), "seed_")
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The text after `label` on the first line of `stderr` that contains it.
fn after<'a>(stderr: &'a str, label: &str) -> &'a str {
    let line = stderr.lines().find(|l| l.contains(label))
        .unwrap_or_else(|| panic!("no `{label}` line in:\n{stderr}"));
    line[line.find(label).unwrap() + label.len()..].trim()
}

#[test]
fn every_new_pmmh_record_matches_the_line_the_run_printed() {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_pmmh_records_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for e in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        std::fs::copy(e.path(), work.join(e.file_name())).unwrap();
    }
    let run = |args: &[&str]| -> String {
        let out = Command::new(camdl())
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc())
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{args:?} failed:\n{stderr}");
        stderr
    };

    let stderr = run(&["fit", "run", "pmmh_stuck.toml", "--seed", "8", "--progress", "none"]);
    let dir = leaf(&work, "pmmh_stuck");

    // The opening block.
    let header = &json(&dir.join("run.json"))["inputs"]["header"];
    assert_eq!(header["config"], "pmmh_stuck.toml");
    assert_eq!(header["model"], after(&stderr, "model:"));
    let names = |v: &serde_json::Value| v.as_array().unwrap().iter()
        .map(|x| x.as_str().unwrap().to_string()).collect::<Vec<_>>().join(", ");
    assert_eq!(names(&header["estimate"]), after(&stderr, "estimate:"));
    assert_eq!(names(&header["fixed"]), after(&stderr, "fixed:"));
    assert_eq!(header["starts"], after(&stderr, "starts:"));
    let output = header["output"].as_str().unwrap();
    assert!(output.starts_with("results/fits/pmmh_stuck-") && after(&stderr, "output:").ends_with(output),
        "relative output {output} is the printed path's tail");

    // The noise check's mean.
    let state: toml::Value = toml::from_str(&std::fs::read_to_string(dir.join("fit_state.toml")).unwrap()).unwrap();
    let ll_mean = state["pf_noise"]["ll_mean"].as_float().unwrap();
    assert!(after(&stderr, "log L̂ mean =").starts_with(&format!("{ll_mean:.1},")), "{ll_mean}");

    // Warm-up: one record per chain; the two chains that printed a floor
    // warning are the two with floor steps, with the printed counts.
    let summary = json(&dir.join("pmmh_summary.json"));
    let warmup = summary["warmup"].as_array().unwrap();
    assert_eq!(warmup.len(), 4);
    let mut floors: Vec<u64> = warmup.iter()
        .map(|w| w["steps_at_scale_floor"].as_u64().unwrap())
        .filter(|&n| n > 0)
        .collect();
    floors.sort();
    let mut printed: Vec<u64> = stderr.lines()
        .filter(|l| l.contains("warm-up steps: no scale reached the"))
        .map(|l| l.trim().strip_prefix("for ").unwrap().split(' ').next().unwrap().parse().unwrap())
        .collect();
    printed.sort();
    assert_eq!(floors, printed, "{stderr}");
    let mut lambdas: Vec<String> = warmup.iter()
        .map(|w| format!("lambda = {:.3e}", w["final_scale"].as_f64().unwrap())).collect();
    let mut printed_lambdas: Vec<String> = stderr.lines()
        .filter(|l| l.contains("adaptive proposal scale at end of warm-up:"))
        .map(|l| after(l, "warm-up:").to_string()).collect();
    lambdas.sort();
    printed_lambdas.sort();
    assert_eq!(lambdas, printed_lambdas);
    for w in warmup {
        assert_eq!(w["warmup_steps"], 1000);
        assert!(stderr.contains(&format!("{:.1}% target acceptance", w["target_accept"].as_f64().unwrap() * 100.0))
            || w["steps_at_scale_floor"] == 0);
    }

    // Plan and cost.
    let plan = &summary["plan"];
    assert_eq!(
        format!("pmmh: {} chains × {} steps × {} particles, burn_in={}, thin={}, adapt={}",
            plan["n_chains"], plan["n_steps"], plan["n_particles"], plan["burn_in"], plan["thin"], plan["adapt"]),
        stderr.lines().find(|l| l.starts_with("pmmh: ")).unwrap());
    assert_eq!(plan["proposal_sd"]["beta"], 0.25);
    let cost = &summary["cost"];
    assert!(after(&stderr, "pmmh complete in").contains(&format!("({} PF evaluations,", cost["evaluations"])));
    assert!(after(&stderr, "pmmh complete in")
        .starts_with(&format!("{:.1}s", cost["sampling_seconds"].as_f64().unwrap())));

    // A resumed run continues its base's warm-up rather than recording the
    // empty warm-up its own call saw.
    let base = leaf(&work, "pmmh_stuck");
    std::fs::copy(work.join("pmmh_stuck.toml"), work.join("pmmh_stuck_more.toml")).unwrap();
    let more = std::fs::read_to_string(work.join("pmmh_stuck_more.toml")).unwrap()
        .replace("iterations = 1500", "iterations = 1600");
    std::fs::write(work.join("pmmh_stuck_more.toml"), more).unwrap();
    run(&["fit", "run", "pmmh_stuck_more.toml", "--seed", "8", "--progress", "none",
          "--resume", base.to_str().unwrap()]);
    let resumed = json(&leaf(&work, "pmmh_stuck_more").join("pmmh_summary.json"));
    assert_eq!(resumed["warmup"], summary["warmup"]);
}
