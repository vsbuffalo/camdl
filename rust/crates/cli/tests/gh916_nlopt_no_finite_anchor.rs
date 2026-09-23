//! gh#916: an NLopt stage in which no chain reached a finite log-likelihood
//! must refuse (non-zero exit, no `mle_params.toml`), the way the PGAS, PMMH
//! and IF2 drivers already do (gh#226).
//!
//! The fixture makes every θ unscorable, not just the start: the epidemic is
//! seeded with zero infecteds, so the projected prevalence is 0 at every
//! observation time, and a Poisson observation of a positive count at rate 0
//! has log-likelihood `-inf` whatever `beta` is. NLopt's objective floor
//! (`NONFINITE_FLOOR`) used to come back as the chain's log-likelihood, a
//! finite `-1e100`, which the whole-fit check would have passed as an anchor.
//!
//! End-to-end via the built `camdl` binary; skipped when the release binary or
//! `camdlc.exe` is missing.

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
        "camdl_gh916_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

fn files_named(dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                    found.push(p);
                }
            }
        }
    }
    found
}

/// Run a one-chain `nl-sbplx` fit on a SIR model seeded with `i0` infecteds
/// against a hand-written positive prevalence series. Returns
/// `(exit success, stderr, output dir)`.
fn run_fit(tmp: &TempDir, i0: u32) -> Option<(bool, String, PathBuf)> {
    let bin = camdl_bin();
    let camdlc = camdlc()?;
    if !bin.exists() {
        return None;
    }
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 9990  I = {i0} }}
simulate {{ from = 0 'days  to = 20 'days }}
"#);
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = tmp.path().join("sir.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();

    let mut tsv = String::from("time\tcases\n");
    for (k, c) in [12, 15, 19, 24, 30, 37, 45, 54, 63, 71].iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", 2 * (k + 1), c));
    }
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, tsv).unwrap();

    let outdir = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = 1.5 }}

[fixed]
gamma = 0.3
N0 = 10000

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#, out = outdir.display(), ir = ir.display(), data = data.display())).unwrap();

    let run = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit_toml).arg("--no-dt-check")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    Some((
        run.status.success(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        outdir,
    ))
}

#[test]
fn an_nlopt_stage_with_no_finite_chain_refuses() {
    let tmp = tempdir("refuse");
    let Some((ok, stderr, outdir)) = run_fit(&tmp, 0) else {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    };
    assert!(!ok, "a fit whose every θ scores -inf must exit non-zero; stderr:\n{stderr}");
    assert!(
        stderr.contains("no finite log-likelihood"),
        "the refusal must say why; stderr:\n{stderr}"
    );
    assert!(
        files_named(&outdir, "mle_params.toml").is_empty(),
        "a refused NLopt stage must not write mle_params.toml"
    );
}

/// Negative control: the same fixture seeded with ten infecteds is scorable,
/// so the stage succeeds and writes its MLE. Without this, the test above
/// would pass for a fixture that fails for any reason at all.
#[test]
fn the_same_fit_with_a_seeded_epidemic_succeeds() {
    let tmp = tempdir("control");
    let Some((ok, stderr, outdir)) = run_fit(&tmp, 10) else { return };
    assert!(ok, "the scorable control must succeed; stderr:\n{stderr}");
    assert_eq!(files_named(&outdir, "mle_params.toml").len(), 1);
}
