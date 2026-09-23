//! gh#595: `camdl fit diff` must report every change to the fit problem —
//! parameters added to or dropped from `[estimate]` / `[fixed]`, a changed
//! `[fixed]` value, a different model, different data — not only the
//! `[estimate]`↔`[fixed]` moves its hand-written comparison knew about.
//!
//! Before the fix, a config that added a new `[estimate]` parameter, dropped a
//! `[fixed]` one and pointed at a different model was reported as
//! "(no parameter changes)": a positive claim, and false.
//!
//! End-to-end via the release `camdl` binary; skipped when it or `camdlc.exe`
//! is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gh595_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// A SIR model; `waning` adds an R → S transition and its rate parameter,
/// so the two variants compile to different models.
fn model_src(waning: bool) -> String {
    let (param, trans) = if waning {
        ("  omega : rate  in [0.0, 1.0]\n", "  waning    : R --> S @ omega * R\n")
    } else {
        ("", "")
    };
    format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
{param}}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
{trans}}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 9990  I = 10 }}
simulate {{ from = 0 'days  to = 20 'days }}
"#)
}

fn run_diff(a: &Path, b: &Path) -> Option<String> {
    let bin = camdl_bin();
    let camdlc = camdlc()?;
    if !bin.exists() { return None; }
    // `fit diff` compiles each `.camdl` to compare the models; point it at the
    // compiler under test (this child process only).
    let out = Command::new(&bin)
        .args(["fit", "diff"]).arg(a).arg(b)
        .env("CAMDLC", camdlc)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(), "fit diff failed: {}", String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn fit_diff_names_parameter_model_and_data_changes() {
    let tmp = tempdir("changes");
    let d = tmp.path();
    std::fs::write(d.join("sir.camdl"), model_src(false)).unwrap();
    std::fs::write(d.join("sir_waning.camdl"), model_src(true)).unwrap();
    std::fs::write(d.join("cases_a.tsv"), "time\tcases\n2\t12\n4\t15\n").unwrap();
    std::fs::write(d.join("cases_b.tsv"), "time\tcases\n2\t12\n4\t16\n").unwrap();

    let a = d.join("a.toml");
    std::fs::write(&a, r#"
[model]
camdl = "sir.camdl"

[data.observations]
cases = "cases_a.tsv"

[estimate]
beta = { bounds = [0.05, 5.0], start = 1.5 }

[fixed]
gamma = 0.3
N0 = 10000

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#).unwrap();
    // B: a different model; a new parameter estimated; N0's value changed;
    // gamma dropped from [fixed] (left to the model default); other data.
    let b = d.join("b.toml");
    std::fs::write(&b, r#"
[model]
camdl = "sir_waning.camdl"

[data.observations]
cases = "cases_b.tsv"

[estimate]
beta = { bounds = [0.05, 5.0], start = 1.5 }
omega = { bounds = [0.0, 1.0], start = 0.1 }

[fixed]
N0 = 20000

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#).unwrap();

    let Some(out) = run_diff(&a, &b) else {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    };
    eprintln!("{out}");
    assert!(!out.contains("no parameter changes"),
        "the configs differ in their parameters; output:\n{out}");
    assert!(out.contains("omega"), "the new [estimate] parameter is not named:\n{out}");
    assert!(out.contains("gamma"), "the dropped [fixed] parameter is not named:\n{out}");
    assert!(out.contains("N0") && out.contains("10000") && out.contains("20000"),
        "the changed [fixed] value is not shown:\n{out}");
    assert!(out.contains("sir_waning.camdl") && out.contains("(compiled model differs)"),
        "the model change is not named:\n{out}");
    assert!(out.contains("cases"), "the data change is not named:\n{out}");

    // Negative control: a config against itself reports no change anywhere.
    let same = run_diff(&a, &a).unwrap();
    assert!(same.contains("no parameter changes"), "identity diff:\n{same}");
    assert!(!same.contains("sir_waning"), "identity diff:\n{same}");
}
