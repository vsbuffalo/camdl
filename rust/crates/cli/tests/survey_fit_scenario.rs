//! `camdl survey --fit fit.toml --scenario NAME` must apply the scenario.
//!
//! The fit-aware branch of `survey` once read only the fit config's own
//! top-level `scenario` key, so a `--scenario` flag never reached parameter
//! resolution: `--scenario X` produced byte-identical landscape rows in the
//! same store leaf as a run with no scenario, and `--scenario typo` exited 0.
//!
//! Every assertion below is about the resolved landscape or the store leaf,
//! not about printed text, except the refusals, whose messages are the
//! user-facing contract (the declared names are listed).
//!
//! The model's scenario `X` sets `gamma = 0.9`. Runs under a scenario use a
//! fit config whose `[fixed]` does not pin `gamma`, so the scenario supplies it,
//! and `--scenario X` must score exactly what a config pinning `gamma = 0.9`
//! scores. A config that pins `gamma` to another value under a scenario that
//! sets it names two values for one parameter and is refused (run spec §1.3).

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
        "camdl_survey_fit_scenario_{}_{}_{}", tag, std::process::id(), ns));
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
  X { set = { gamma = 0.9 } }
  Y { set = { gamma = 0.5 } }
}
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

/// Lay out the model, data and a fit config `<name>.toml` in `dir`. `gamma`
/// is the value `[fixed]` pins (`None`: not pinned, so a scenario supplies it);
/// `top` is spliced in above `[model]` (e.g. `scenario = "X"`).
fn setup(dir: &Path, name: &str, gamma: Option<f64>, top: &str) -> (PathBuf, PathBuf) {
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
beta = {{ bounds = [0.01, 1.0], start = 0.3 }}
[fixed]
N0 = 1000
{gamma}
"#, ir = ir.display(), data = data.display(),
    gamma = gamma.map(|g| format!("gamma = {g}")).unwrap_or_default())).unwrap();
    (ir, fit)
}

fn survey(root: &Path, ir: &Path, fit: &Path, extra: &[&str]) -> Output {
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["survey", &ir.to_string_lossy(), "--fit", &fit.to_string_lossy()])
        .args(extra)
        .args(["--eval", "simulate", "--n-points", "10", "--seed", "1"])
        .args(["--output", &root.to_string_lossy()])
        .output()
        .expect("spawn camdl survey")
}

fn ok(out: Output) -> Output {
    assert!(out.status.success(), "camdl survey failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    out
}

/// Every survey leaf under `root`, sorted.
fn leaves(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.join("surveys")];
    while let Some(dir) = stack.pop() {
        if dir.join("landscape.tsv").is_file() {
            found.push(dir.clone());
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

/// The single leaf under `root`.
fn only_leaf(root: &Path) -> PathBuf {
    let l = leaves(root);
    assert_eq!(l.len(), 1, "expected exactly one survey leaf under {}: {l:?}", root.display());
    l.into_iter().next().unwrap()
}

/// The landscape's data rows (header comments carry the run hash, so they
/// are dropped: two leaves with different identities can still score the
/// same numbers, and that equality is what is being asserted).
fn rows(leaf: &Path) -> String {
    std::fs::read_to_string(leaf.join("landscape.tsv")).unwrap()
        .lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>().join("\n")
}

#[test]
fn scenario_flag_is_applied_and_keys_its_own_leaf() {
    let tmp = tempdir("apply");
    let (ir, fit) = setup(tmp.path(), "fit", Some(0.1), "");
    let (_, fit_free) = setup(tmp.path(), "fit_free", None, "");
    let (_, fit_09) = setup(tmp.path(), "fit_gamma_09", Some(0.9), "");

    let root_none = tmp.path().join("none");
    let root_x = tmp.path().join("x");
    let root_y = tmp.path().join("y");
    let root_ref = tmp.path().join("ref");
    ok(survey(&root_none, &ir, &fit, &[]));
    ok(survey(&root_x, &ir, &fit_free, &["--scenario", "X"]));
    ok(survey(&root_y, &ir, &fit_free, &["--scenario", "Y"]));
    // Reference: the same problem with `[fixed] gamma = 0.9` and no scenario.
    ok(survey(&root_ref, &ir, &fit_09, &[]));

    let (none, x, y, reference) =
        (only_leaf(&root_none), only_leaf(&root_x), only_leaf(&root_y), only_leaf(&root_ref));
    assert_ne!(rows(&none), rows(&x),
        "--scenario X scored the same landscape as [fixed] gamma = 0.1");
    assert_ne!(rows(&x), rows(&y),
        "--scenario X and --scenario Y scored the same landscape on one fit config");
    assert_eq!(rows(&x), rows(&reference),
        "--scenario X (gamma = 0.9, unpinned in [fixed]) must score exactly \
         what [fixed] gamma = 0.9 scores");

    // Same output root, same fit config: each scenario keys its own leaf.
    let shared = tmp.path().join("shared");
    ok(survey(&shared, &ir, &fit_free, &["--scenario", "X"]));
    ok(survey(&shared, &ir, &fit_free, &["--scenario", "Y"]));
    assert_eq!(leaves(&shared).len(), 2,
        "--scenario X and --scenario Y must occupy two leaves: {:?}", leaves(&shared));
}

#[test]
fn undeclared_scenario_flag_is_refused_with_the_declared_names() {
    let tmp = tempdir("typo");
    let (ir, fit) = setup(tmp.path(), "fit", Some(0.1), "");
    let root = tmp.path().join("results");
    let out = survey(&root, &ir, &fit, &["--scenario", "typo"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "--scenario typo must be refused; stderr:\n{stderr}");
    assert!(stderr.contains("typo") && stderr.contains('X') && stderr.contains('Y'),
        "the refusal must name the bad scenario and list the declared ones; stderr:\n{stderr}");
    assert!(leaves(&root).is_empty(), "a refused run must write no leaf");
}

#[test]
fn flag_agreeing_with_the_config_scenario_is_the_same_run() {
    let tmp = tempdir("agree");
    let (ir, cfg_path) = setup(tmp.path(), "fit_x", None, "scenario = \"X\"");
    let (_, fit_plain) = setup(tmp.path(), "fit", None, "");
    let (_, fit_09) = setup(tmp.path(), "fit_gamma_09", Some(0.9), "");

    let root_cfg = tmp.path().join("cfg");
    let root_both = tmp.path().join("both");
    let root_flag = tmp.path().join("flag");
    ok(survey(&root_cfg, &ir, &cfg_path, &[]));
    ok(survey(&root_both, &ir, &cfg_path, &["--scenario", "X"]));
    ok(survey(&root_flag, &ir, &fit_plain, &["--scenario", "X"]));

    let (cfg, both, flag) = (only_leaf(&root_cfg), only_leaf(&root_both), only_leaf(&root_flag));
    let rel = |leaf: &Path, root: &Path| leaf.strip_prefix(root).unwrap().to_path_buf();
    assert_eq!(rel(&cfg, &root_cfg), rel(&both, &root_both),
        "config `scenario = \"X\"` and the same name on --scenario are one run");
    assert_eq!(rel(&cfg, &root_cfg), rel(&flag, &root_flag),
        "the scenario's source (config or flag) is not identity");
    assert_eq!(rows(&cfg), rows(&flag));

    // The config's own `scenario` key is applied too, not only the flag's.
    let root_ref = tmp.path().join("ref");
    ok(survey(&root_ref, &ir, &fit_09, &[]));
    assert_eq!(rows(&cfg), rows(&only_leaf(&root_ref)),
        "config `scenario = \"X\"` (gamma = 0.9, unpinned in [fixed]) must \
         score exactly what [fixed] gamma = 0.9 scores");
}

#[test]
fn flag_disagreeing_with_the_config_scenario_is_refused() {
    let tmp = tempdir("conflict");
    let (ir, fit) = setup(tmp.path(), "fit_x", None, "scenario = \"X\"");
    let root = tmp.path().join("results");
    let out = survey(&root, &ir, &fit, &["--scenario", "Y"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "config scenario X with --scenario Y must be refused; stderr:\n{stderr}");
    assert!(stderr.contains("`X`") && stderr.contains("`Y`"),
        "the refusal must name both scenarios; stderr:\n{stderr}");
    assert!(leaves(&root).is_empty(), "a refused run must write no leaf");
}

/// A scenario (from the config or the flag) that sets a parameter `[fixed]`
/// pins to another value is refused, naming both values and both sources, the
/// rule `fit run` applies to the same file.
#[test]
fn scenario_against_a_pinned_fixed_value_is_refused() {
    let tmp = tempdir("collide");
    let (ir, cfg) = setup(tmp.path(), "fit_x", Some(0.1), "scenario = \"X\"");
    let (_, plain) = setup(tmp.path(), "fit", Some(0.1), "");
    for (what, fit, extra) in [("config scenario", &cfg, &[][..]),
                               ("--scenario", &plain, &["--scenario", "X"][..])] {
        let root = tmp.path().join(what.replace(' ', "_"));
        let out = survey(&root, &ir, fit, extra);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{what}: must be refused; stderr:\n{stderr}");
        for line in [
            "parameter `gamma` is given two values: by scenario `X` and by [fixed] in",
            "[fixed]:   gamma = 0.1",
            "scenario:  gamma = 0.9    (set gamma = 0.9)",
            "Fix: remove `gamma` from scenario `X`",
        ] {
            assert!(stderr.contains(line), "{what}: diagnostic lacks {line:?}; stderr:\n{stderr}");
        }
        assert!(leaves(&root).is_empty(), "{what}: a refused run must write no leaf");
    }
}
