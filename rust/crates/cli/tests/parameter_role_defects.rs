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
