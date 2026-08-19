//! `camdl simulate --to SPEC` — the obs-anchored horizon override (gh#626).
//!
//! Pins the proposal's contract
//! (docs/dev/proposals/2026-08-19-obs-anchored-horizon-cli.md):
//! an absolute `--to` moves the run horizon (trajectory rows appear past the
//! model's baked `simulate { to }`); an anchored `--to` without `--fit` is a
//! named refusal (a forward simulation binds no observed data); with
//! `--fit <toml>` the anchor resolves against the fit's [data.observations]
//! and the run ends at exactly `last_obs + offset`; a scenario carrying its
//! own DIFFERENT horizon is a named conflict (never silently discard a
//! declared horizon, gh#561), while an equal horizon is the allowed no-op;
//! an inverted horizon is refused up front (no validator used to check
//! `t_end > t_start` — the failure was a silent header-only TSV).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn golden(name: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden").join(name)
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).expect("write fixture");
}

/// Long-form `(time, patch, cases)` for `sir_two_patch_long_obs`: weekly
/// cadence ending at t = 42 — `last_obs` = 42.
fn write_obs_data(path: &Path) {
    write(path,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\t6\n\
         21\turban\t25\n21\trural\t9\n\
         28\turban\t30\n28\trural\t11\n\
         35\turban\t28\n35\trural\t10\n\
         42\turban\t22\n42\trural\t8\n");
}

/// The four estimated params, pinned for forward simulation.
const PARAMS: [&str; 8] = [
    "--param", "beta=0.3", "--param", "gamma=0.1",
    "--param", "rho=0.6", "--param", "k=5.0",
];

/// Max `time` value in a trajectory TSV (first column, tab-separated,
/// one header line; tolerate replicate/scenario lead columns by scanning for
/// the column named `time`).
fn max_time(tsv: &str) -> f64 {
    let mut lines = tsv.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header = lines.next().expect("header");
    let ti = header.split('\t').position(|c| c == "time" || c == "t")
        .unwrap_or_else(|| panic!("time column in header: {header:?}"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').nth(ti).unwrap().parse::<f64>().unwrap())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// A minimal fit toml for the anchored path: only [data.observations] (and
/// the schema-required blocks) are consulted by `--to`.
fn write_fit_toml(tmp: &Path, data: &Path) -> PathBuf {
    let toml = tmp.join("fit.toml");
    write(&toml, &format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ log_normal = {{ mu = -2.3, sigma = 0.5 }} }} }}
[fixed]
beta = 0.3
rho = 0.6
k = 5.0
[stages.dummy]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        out  = tmp.join("results").display(),
        ir   = golden("sir_two_patch_long_obs.ir.json").display(),
        data = data.display(),
    ));
    toml
}

/// Absolute `--to` extends the run past the model's baked horizon (t_end =
/// 100): trajectory rows appear out to the new end, and the run_id differs
/// from the un-overridden run (count-in-the-key).
#[test]
fn absolute_to_extends_trajectory_and_rekeys() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    let run = |to: Option<&str>, out: &str| {
        let mut args: Vec<String> = vec![
            "simulate".into(),
            golden("sir_two_patch_long_obs.ir.json").to_string_lossy().into_owned(),
            "--output".into(), tmp.path().join(out).to_string_lossy().into_owned(),
            "--seed".into(), "1".into(),
        ];
        args.extend(PARAMS.iter().map(|s| s.to_string()));
        if let Some(t) = to {
            args.push("--to".into());
            args.push(t.into());
        }
        let o = Command::new(&bin)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
            .args(&args)
            .output()
            .expect("spawn simulate");
        assert!(o.status.success(),
            "simulate must succeed (to={to:?}):\nstderr={}",
            String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stderr).into_owned()
    };

    let stderr_base = run(None, "base.tsv");
    let stderr_ext = run(Some("140"), "ext.tsv");
    assert!(stderr_ext.contains("--to \"140\" → t_end = 140"),
        "the resolved override must be announced:\n{stderr_ext}");

    let base = std::fs::read_to_string(tmp.path().join("base.tsv")).unwrap();
    let ext = std::fs::read_to_string(tmp.path().join("ext.tsv")).unwrap();
    assert_eq!(max_time(&base), 100.0, "baked horizon");
    assert_eq!(max_time(&ext), 140.0, "--to 140 must extend the output grid");

    // Distinct identities: the two runs committed different CAS leaves. The
    // stderr run lines carry the run ids; cheapest robust check is that the
    // leaves differ — assert the store has (at least) two sim leaves.
    let sims = tmp.path().join("cas").join("sims");
    let n_leaves = walk_count(&sims);
    assert!(n_leaves >= 2,
        "two horizons must commit two distinct CAS leaves, found {n_leaves}");
    let _ = stderr_base;
}

/// Count leaf run dirs (containing run.json) under the sims store.
fn walk_count(root: &Path) -> usize {
    let mut n = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").is_file() { n += 1; }
                stack.push(p);
            }
        }
    }
    n
}

/// Anchored `--to` without `--fit`: the named refusal.
#[test]
fn anchored_to_without_fit_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let mut args: Vec<String> = vec![
        "simulate".into(),
        golden("sir_two_patch_long_obs.ir.json").to_string_lossy().into_owned(),
        "--to".into(), "last_obs + 8 weeks".into(),
        "--stdout".into(),
    ];
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .args(&args)
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "anchored --to without --fit must refuse");
    assert!(stderr.contains("anchored to observed data") && stderr.contains("--fit"),
        "the refusal must name the fix:\n{stderr}");
}

/// Anchored `--to` with `--fit <toml>`: `last_obs` = 42 from the bound data,
/// so `last_obs + 2 weeks` runs to exactly t = 56, with synthetic obs rows
/// emitted past `last_obs` (the forecast-band use case).
#[test]
fn anchored_to_resolves_against_fit_data() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_obs_data(&data);
    let toml = write_fit_toml(tmp.path(), &data);

    let mut args: Vec<String> = vec![
        "simulate".into(),
        golden("sir_two_patch_long_obs.ir.json").to_string_lossy().into_owned(),
        "--to".into(), "last_obs + 2 weeks".into(),
        "--fit".into(), toml.to_string_lossy().into_owned(),
        "--output".into(), tmp.path().join("traj.tsv").to_string_lossy().into_owned(),
        "--obs".into(), tmp.path().join("bands.tsv").to_string_lossy().into_owned(),
        "--seed".into(), "1".into(),
    ];
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .args(&args)
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "anchored --to with --fit must run:\n{stderr}");
    assert!(stderr.contains("→ t_end = 56"),
        "last_obs = 42 (max obs time) + 14 days = 56:\n{stderr}");

    let traj = std::fs::read_to_string(tmp.path().join("traj.tsv")).unwrap();
    assert_eq!(max_time(&traj), 56.0, "trajectory runs to the anchored horizon");
    let bands = std::fs::read_to_string(tmp.path().join("bands.tsv")).unwrap();
    let bands_max = max_time(&bands);
    assert!(bands_max > 42.0,
        "synthetic obs must be emitted PAST last_obs (weekly cadence to 56), \
         got max emit time {bands_max}");
}

/// `--to` at or before t_start: refused up front with the resolved value.
#[test]
fn inverted_horizon_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let mut args: Vec<String> = vec![
        "simulate".into(),
        golden("sir_two_patch_long_obs.ir.json").to_string_lossy().into_owned(),
        "--to".into(), "0".into(),
        "--stdout".into(),
    ];
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .args(&args)
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "t_end <= t_start must refuse");
    assert!(stderr.contains("t_start"),
        "the refusal must name the ordering:\n{stderr}");
}

/// Scenario with a DIFFERENT declared horizon + `--to`: the named conflict.
/// (`sir_demography`'s `endemic` declares t_end = 3650 vs model 365.)
#[test]
fn scenario_horizon_conflict_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .args([
            "simulate",
            &golden("sir_demography.ir.json").to_string_lossy(),
            "--scenario", "endemic",
            "--to", "500",
            "--stdout",
        ])
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "conflicting horizons must refuse");
    assert!(stderr.contains("endemic") && stderr.contains("--to")
            && stderr.contains("label-only"),
        "the conflict must name both sources and the fix:\n{stderr}");

    // The no-op precedent: --to EQUAL to the scenario's declared horizon is
    // allowed (the same rule refuse_scenario_horizon applies to presets that
    // restate the run horizon). Params may fail later; the horizon check must
    // not be what refuses it.
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas2"))
        .args([
            "simulate",
            &golden("sir_demography.ir.json").to_string_lossy(),
            "--scenario", "endemic",
            "--to", "3650",
            "--stdout",
        ])
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!stderr.contains("refusing to pick one silently"),
        "--to equal to the scenario horizon is the allowed no-op:\n{stderr}");
}

/// An `at [...]` emit schedule cannot grow with the horizon: an extending
/// `--to` with `--obs` would run longer and emit nothing past the listed
/// times, exit 0 — refused instead (the silent-drop class, gh#626).
#[test]
fn extending_to_with_at_list_emit_schedule_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    // Rewrite the fixture's Regular emit schedules to a fixed `at` list.
    // (`validated_by` is an unverified provenance marker, so editing the
    // envelope JSON is safe for a test fixture.)
    let raw = std::fs::read_to_string(golden("sir_two_patch_long_obs.ir.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    for o in v["model"]["observations"].as_array_mut().unwrap() {
        o["emit_schedule"] = serde_json::json!({"at_times": [7.0, 14.0, 21.0, 28.0, 35.0, 42.0]});
    }
    let ir = tmp.path().join("at_list.ir.json");
    write(&ir, &serde_json::to_string(&v).unwrap());

    let mut args: Vec<String> = vec![
        "simulate".into(),
        ir.to_string_lossy().into_owned(),
        "--to".into(), "140".into(),
        "--obs".into(), tmp.path().join("bands.tsv").to_string_lossy().into_owned(),
        "--seed".into(), "1".into(),
    ];
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    let o = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .args(&args)
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(),
        "extending --to over a fixed at-list emit schedule must refuse:\n{stderr}");
    assert!(stderr.contains("fixed list") && stderr.contains("42"),
        "the refusal must name the stream's list end:\n{stderr}");
}
