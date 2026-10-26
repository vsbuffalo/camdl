//! `camdl profile --fit fit.toml` must apply the fit config's own scenario.
//!
//! The profile verb read `[estimate]`, `[fixed]` and `[data]` from the fit
//! config but never its top-level `scenario` / `enable` / `disable`, so a
//! config naming a scenario profiled the unmodified model. It now combines
//! the config's selection with `--scenario` / `--enable` / `--disable` by the
//! same rule as `survey --fit` (`util::combine_fit_scenario`): either alone
//! selects, both must name the same scenario, anything else is refused.
//!
//! Scenario `Z` sets `N0 = 5000` over the config's `[fixed] N0 = 1000`; the
//! assertions read the resolved `N0` each cell's `run.json` records, and the
//! cell log-likelihoods.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp {
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_profile_fit_scenario_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.0, 1.0]
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
simulate { from = 0 'days  to = 6 'days }
scenarios {
  Z { set = { N0 = 5000 } }
  W { set = { N0 = 2000 } }
}
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

/// Model, data and a fit config `<name>.toml` in `dir`; `top` is spliced in
/// above `[model]` (e.g. `scenario = "Z"`). `[fixed]` pins `N0 = 1000`.
fn setup(dir: &Path, name: &str, top: &str) -> (PathBuf, PathBuf) {
    let model_src = dir.join("m.camdl");
    std::fs::write(&model_src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).arg(&model_src).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let data = dir.join("cases.tsv");
    std::fs::write(&data, CASES_TSV).unwrap();
    let fit = dir.join(format!("{name}.toml"));
    std::fs::write(&fit, format!(r#"{top}
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
beta  = {{ bounds = [0.01, 1.0], start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.1 }}
[fixed]
N0 = 1000
"#, ir = ir.display(), data = data.display())).unwrap();
    (ir, fit)
}

/// Run `camdl profile` with `cwd` as the working directory, so its store
/// lands in `cwd/results` (profile has no output-root flag).
fn profile(cwd: &Path, ir: &Path, fit: &Path, extra: &[&str]) -> Output {
    std::fs::create_dir_all(cwd).unwrap();
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(cwd)
        .args(["profile", &ir.to_string_lossy(), "--fit", &fit.to_string_lossy()])
        .args(["--sweep", "gamma=0.1,0.5", "--particles", "50", "--iterations", "3",
               "--n-starts", "1", "--rw-sd", "auto", "--seed", "1"])
        .args(extra)
        .output()
        .expect("spawn camdl profile")
}

fn ok(out: Output) -> Output {
    assert!(out.status.success(), "camdl profile failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    out
}

/// Every cell's `run.json` under `cwd/results/profiles`, sorted by path.
fn cells(cwd: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![cwd.join("results").join("profiles")];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            found.push(dir.join("run.json"));
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() { stack.push(e.path()); }
            }
        }
    }
    found.sort();
    found
}

/// `(resolved N0, its source, best_loglik)` for each cell, in grid order.
fn n0_and_loglik(cwd: &Path) -> Vec<(f64, String, f64)> {
    let mut rows: Vec<(u64, f64, String, f64)> = cells(cwd).iter().map(|p| {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let inputs = &v["inputs"];
        let n0 = &inputs["provenance"]["parameters_provenance"]["N0"];
        (inputs["grid_point"].as_u64().unwrap(),
         n0["value"].as_f64().unwrap_or_else(|| panic!("no N0 value in {}", p.display())),
         n0["source"].as_str().unwrap().to_string(),
         inputs["best_loglik"].as_f64().unwrap())
    }).collect();
    rows.sort_by_key(|r| r.0);
    assert_eq!(rows.len(), 2, "expected two profile cells under {}", cwd.display());
    rows.into_iter().map(|(_, a, b, c)| (a, b, c)).collect()
}

#[test]
fn config_scenario_is_applied() {
    let tmp = tempdir("cfg");
    let (ir, plain) = setup(tmp.path(), "fit", "");
    let (_, with_z) = setup(tmp.path(), "fit_z", "scenario = \"Z\"");

    ok(profile(&tmp.path().join("none"), &ir, &plain, &[]));
    ok(profile(&tmp.path().join("cfg"), &ir, &with_z, &[]));
    ok(profile(&tmp.path().join("flag"), &ir, &plain, &["--scenario", "Z"]));

    let none = n0_and_loglik(&tmp.path().join("none"));
    let cfg = n0_and_loglik(&tmp.path().join("cfg"));
    let flag = n0_and_loglik(&tmp.path().join("flag"));
    for (n0, source, _) in &none {
        assert_eq!((*n0, source.as_str()), (1000.0, "fit_toml_fixed"));
    }
    for (n0, source, _) in &cfg {
        assert_eq!((*n0, source.as_str()), (5000.0, "scenario"),
            "the fit config's `scenario = \"Z\"` must set N0 = 5000");
    }
    assert_eq!(cfg, flag,
        "the config's scenario and the same name on --scenario must profile identically");
    assert_ne!(cfg.iter().map(|r| r.2).collect::<Vec<_>>(),
               none.iter().map(|r| r.2).collect::<Vec<_>>(),
        "the scenario must move the cell log-likelihoods");
}

#[test]
fn flag_agreeing_with_the_config_scenario_runs() {
    let tmp = tempdir("agree");
    let (ir, with_z) = setup(tmp.path(), "fit_z", "scenario = \"Z\"");
    ok(profile(&tmp.path().join("cfg"), &ir, &with_z, &[]));
    ok(profile(&tmp.path().join("both"), &ir, &with_z, &["--scenario", "Z"]));
    assert_eq!(n0_and_loglik(&tmp.path().join("cfg")),
               n0_and_loglik(&tmp.path().join("both")));
}

#[test]
fn flag_disagreeing_with_the_config_scenario_is_refused() {
    let tmp = tempdir("conflict");
    let (ir, with_z) = setup(tmp.path(), "fit_z", "scenario = \"Z\"");
    let cwd = tmp.path().join("run");
    let out = profile(&cwd, &ir, &with_z, &["--scenario", "W"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "config scenario Z with --scenario W must be refused; stderr:\n{stderr}");
    assert!(stderr.contains("`Z`") && stderr.contains("`W`"),
        "the refusal must name both scenarios; stderr:\n{stderr}");
    assert!(cells(&cwd).is_empty(), "a refused run must write no cell");
}
