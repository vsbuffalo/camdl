//! Which parameters `profile`, `survey` and `fit run` estimate, pin, sweep and
//! record, end to end through the release binary.
//!
//! Each test is a reproduced silent wrong answer: a run that finished, wrote
//! plausible output, and estimated, sampled or recorded a parameter other than
//! the one the invocation named.

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
        "camdl_parameter_role_defects_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// An SIR model with a reporting rate, a dispersion, two initial-state
/// parameters and one model constant (`omega`, a `let`, which compiles to a
/// parameter whose value the model fixes).
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]
  gamma : rate        in [0.01, 1.0]
  rho   : probability in [0.01, 1.0]
  N0    : count       in [100, 10000]
  I0    : count       in [1, 100]
  k     : positive    in [0.1, 100]
}
let omega : rate = 0.01
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
  wane      : R --> S @ omega * R
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ neg_binomial(mean = rho * projected, r = k)
  }
}
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days  to = 6 'days }
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

/// Compile `src` into `dir/<stem>.ir.json` and write the case data beside it.
fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let model_src = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model_src, src).unwrap();
    let out = Command::new(camdlc_bin()).arg(&model_src).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES_TSV).unwrap();
    ir
}

fn camdl(cwd: &Path, args: &[&str]) -> Output {
    std::fs::create_dir_all(cwd).unwrap();
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn ok(out: Output) -> Output {
    assert!(out.status.success(), "camdl failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    out
}

/// Every file named `name` under `root`, sorted.
fn find(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().is_some_and(|n| n == name) { found.push(p); }
            }
        }
    }
    found.sort();
    found
}

fn json(p: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

/// The profile base's recorded estimated set (`fit.meta.json` `estimated`).
fn profile_estimated(cwd: &Path) -> Vec<String> {
    let metas = find(&cwd.join("results").join("profiles"), "fit.meta.json");
    assert_eq!(metas.len(), 1, "expected one profile base under {}", cwd.display());
    json(&metas[0])["estimated"].as_array().unwrap().iter()
        .map(|v| v.as_str().unwrap().to_string()).collect()
}

const PROFILE_KNOBS: &[&str] = &[
    "--particles", "30", "--iterations", "2", "--n-starts", "1", "--seed", "1",
];

/// `--fixed-file` pins the names it carries: `--rw-sd auto` must not estimate
/// them. It did — the estimated set was every non-focal parameter less only
/// `--fixed` and `[fixed]`, so a file holding `gamma = 0.15` profiled gamma as
/// a nuisance parameter and moved it.
#[test]
fn profile_rw_sd_auto_does_not_estimate_fixed_file_names() {
    let tmp = tempdir("fixed_file");
    let ir = compile(tmp.path(), "m", MODEL);
    let fixed = tmp.path().join("fixed.toml");
    std::fs::write(&fixed,
        "beta = 0.3\ngamma = 0.15\nrho = 0.6\nk = 10\nN0 = 1000\nI0 = 10\n").unwrap();
    let cwd = tmp.path().join("run");
    let data = tmp.path().join("cases.tsv");
    let mut args = vec![
        "profile", ir.to_str().unwrap(),
        "--data", data.to_str().unwrap(),
        "--fixed-file", fixed.to_str().unwrap(),
        "--sweep", "beta=0.2,0.4", "--rw-sd", "auto",
    ];
    args.extend_from_slice(PROFILE_KNOBS);
    ok(camdl(&cwd, &args));
    let est = profile_estimated(&cwd);
    for name in ["gamma", "rho", "k", "N0", "I0"] {
        assert!(!est.contains(&name.to_string()),
            "{name} is pinned by --fixed-file but was estimated: {est:?}");
    }
}

/// A model constant (`let omega : rate = 0.01`, a parameter whose value the
/// model fixes) is not a nuisance parameter: `--rw-sd auto` must not estimate
/// it. It did, alongside every other non-focal name.
#[test]
fn profile_rw_sd_auto_does_not_estimate_a_model_constant() {
    let tmp = tempdir("constant");
    let ir = compile(tmp.path(), "m", MODEL);
    let fixed = tmp.path().join("fixed.toml");
    std::fs::write(&fixed, "beta = 0.3\nrho = 0.6\nk = 10\nN0 = 1000\nI0 = 10\n").unwrap();
    let cwd = tmp.path().join("run");
    let data = tmp.path().join("cases.tsv");
    let mut args = vec![
        "profile", ir.to_str().unwrap(),
        "--data", data.to_str().unwrap(),
        "--fixed-file", fixed.to_str().unwrap(),
        "--fixed", "gamma=0.15",
        "--sweep", "beta=0.2,0.4", "--rw-sd", "auto",
    ];
    args.extend_from_slice(PROFILE_KNOBS);
    // Every declared parameter is focal or pinned, so the model constant is
    // the only name `auto` could pick up.
    ok(camdl(&cwd, &args));
    let est = profile_estimated(&cwd);
    assert!(!est.contains(&"omega".to_string()),
        "omega is a model constant but was estimated: {est:?}");
}

/// A sweep value outside the parameter's bounds is refused, naming the
/// parameter, the offending values and the bounds — the model's declared
/// bounds, narrowed by the fit config's `[estimate]` bounds. A `--starts` value
/// outside them already was; the sweep ran.
#[test]
fn profile_refuses_sweep_values_outside_bounds() {
    let tmp = tempdir("sweep_bounds");
    let ir = compile(tmp.path(), "m", MODEL);
    let data = tmp.path().join("cases.tsv");
    let fixed = tmp.path().join("fixed.toml");
    std::fs::write(&fixed,
        "beta = 0.3\ngamma = 0.15\nrho = 0.6\nk = 10\nN0 = 1000\nI0 = 10\n").unwrap();
    let run = |cwd: &Path, sweep: &str, fit: Option<&Path>| {
        let mut args = vec![
            "profile", ir.to_str().unwrap(),
            "--data", data.to_str().unwrap(),
            "--fixed-file", fixed.to_str().unwrap(),
            "--sweep", sweep, "--rw-sd", "auto",
        ];
        if let Some(f) = fit { args.extend_from_slice(&["--fit", f.to_str().unwrap()]); }
        args.extend_from_slice(PROFILE_KNOBS);
        camdl(cwd, &args)
    };

    // Model bounds: beta in [0.05, 1].
    let out = run(&tmp.path().join("model"), "beta=0.5,2.0,3.0", None);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a sweep to beta=2.0 must be refused:\n{err}");
    assert!(err.contains("beta") && err.contains("2") && err.contains("3")
        && err.contains("[0.05, 1]"),
        "the refusal names the parameter, the values and the bounds:\n{err}");
    assert!(find(&tmp.path().join("model"), "run.json").is_empty(),
        "no cell may be written for a refused sweep");

    // `[estimate]` bounds narrow the model's: beta in [0.1, 0.5].
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, format!(r#"[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
beta = {{ bounds = [0.1, 0.5] }}
[fixed]
gamma = 0.15
"#, ir = ir.display(), data = data.display())).unwrap();
    let out = run(&tmp.path().join("est"), "beta=0.3,0.7", Some(&fit));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "beta=0.7 is outside [estimate] bounds:\n{err}");
    assert!(err.contains("0.7") && err.contains("[0.1, 0.5]"),
        "the refusal names the narrowed bounds:\n{err}");

    // Control: inside the bounds, the same profile runs.
    ok(run(&tmp.path().join("inside"), "beta=0.2,0.4", Some(&fit)));
}

/// `(role, source)` per parameter, from a `parameters_provenance` block.
fn roles(prov: &serde_json::Value) -> std::collections::BTreeMap<String, String> {
    prov.as_object().unwrap_or_else(|| panic!("no parameters_provenance: {prov}"))
        .iter().map(|(k, v)| (k.clone(), v["role"].as_str().unwrap().to_string())).collect()
}

/// A profile records each parameter's role as the profile gave it: the swept
/// parameter `focal`, the ones IF2 moves `estimated`, the rest `fixed`. It
/// recorded every parameter of a CLI-only profile as `fixed`, and its base
/// sidecar carried no provenance at all.
#[test]
fn profile_records_focal_and_estimated_roles() {
    let tmp = tempdir("roles");
    let ir = compile(tmp.path(), "m", MODEL);
    let data = tmp.path().join("cases.tsv");
    let fixed = tmp.path().join("fixed.toml");
    std::fs::write(&fixed, "beta = 0.3\nrho = 0.6\nk = 10\nN0 = 1000\nI0 = 10\n").unwrap();
    // `gamma` is the one estimated parameter; the fit config gives its start.
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, format!(r#"[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.15 }}
[fixed]
"#, ir = ir.display(), data = data.display())).unwrap();
    let cwd = tmp.path().join("run");
    let mut args = vec![
        "profile", ir.to_str().unwrap(),
        "--data", data.to_str().unwrap(),
        "--fit", fit.to_str().unwrap(),
        "--fixed-file", fixed.to_str().unwrap(),
        "--sweep", "beta=0.2,0.4", "--rw-sd", "gamma=0.02",
    ];
    args.extend_from_slice(PROFILE_KNOBS);
    ok(camdl(&cwd, &args));

    let want: std::collections::BTreeMap<String, String> = [
        ("beta", "focal"), ("gamma", "estimated"), ("rho", "fixed"), ("k", "fixed"),
        ("N0", "fixed"), ("I0", "fixed"), ("omega", "fixed"),
    ].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let cells = find(&cwd.join("results").join("profiles"), "run.json");
    assert_eq!(cells.len(), 2);
    for cell in &cells {
        let v = json(cell);
        assert_eq!(roles(&v["inputs"]["provenance"]["parameters_provenance"]), want,
            "{}", cell.display());
    }
    let metas = find(&cwd.join("results").join("profiles"), "fit.meta.json");
    assert_eq!(roles(&json(&metas[0])["parameters_provenance"]), want,
        "the profile base sidecar records the same roles");
}

/// A model whose reporting and recovery read three shares that must sum to
/// one (`p1 + p2 + p3 = 1`), declared as a `[[simplex_groups]]` in the fit
/// config.
const SIMPLEX_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]
  gamma : rate        in [0.01, 1.0]
  p1    : probability in [0, 1]
  p2    : probability in [0, 1]
  p3    : probability in [0, 1]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * (p2 + 2 * p3) * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = p1 * projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

/// `profile --fit` honours the fit config's `[[simplex_groups]]`: IF2 must
/// perturb the members jointly so every cell's optimum stays on the simplex.
/// The profile's IF2 config was built with no groups, so each member moved on
/// its own and the cells' optima summed to 1.85 and 1.44.
#[test]
fn profile_keeps_the_fit_configs_simplex_groups() {
    let tmp = tempdir("simplex");
    let ir = compile(tmp.path(), "s", SIMPLEX_MODEL);
    let data = tmp.path().join("cases.tsv");
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, format!(r#"[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[[simplex_groups]]
params = ["p1", "p2", "p3"]
[estimate]
beta = {{ bounds = [0.05, 1.0], start = 0.3 }}
p1 = {{ bounds = [0, 1], start = 0.5 }}
p2 = {{ bounds = [0, 1], start = 0.3 }}
p3 = {{ bounds = [0, 1], start = 0.2 }}
[fixed]
gamma = 0.2
"#, ir = ir.display(), data = data.display())).unwrap();
    let run = |cwd: &Path, sweep: &str| {
        camdl(cwd, &[
            "profile", ir.to_str().unwrap(), "--fit", fit.to_str().unwrap(),
            "--sweep", sweep, "--rw-sd", "auto", "--iterations", "5",
            "--particles", "50", "--n-starts", "1", "--seed", "1",
        ])
    };
    let cwd = tmp.path().join("run");
    ok(run(&cwd, "beta=0.2,0.4"));
    let mles = find(&cwd.join("results").join("profiles"), "mle.toml");
    assert_eq!(mles.len(), 2);
    for m in &mles {
        let text = std::fs::read_to_string(m).unwrap();
        let v: toml::Value = toml::from_str(&text).unwrap();
        let sum: f64 = ["p1", "p2", "p3"].iter()
            .map(|k| v["mle"][k].as_float().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-9,
            "simplex members must sum to one at each cell's optimum, got {sum}:\n{text}");
    }

    // Sweeping a member would leave the rest free to leave the simplex.
    let out = run(&tmp.path().join("member"), "p1=0.4,0.6");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && err.contains("simplex_groups[0] member 'p1'")
        && err.contains("swept"), "a swept member is refused by name:\n{err}");
}
