//! Warm-up observability golden invariant (PMMH).
//!
//! The streaming per-chain `trace.tsv` must carry warm-up rows (`step <
//! burn_in`) so a live tail sees the chain moving during burn-in, WHILE the
//! canonical posterior (`draws.tsv`) stays exactly the post-burn-in, thinned
//! tail. This pins the additive-observability contract:
//!
//!   1. `trace.tsv` DOES contain `step < burn_in` rows (the fix) AND
//!      post-burn-in rows.
//!   2. `draws.tsv` holds exactly the post-burn-in trace rows — one per
//!      `step >= burn_in` trace row (warm-up filtered back out of the
//!      posterior), and its estimated-parameter values match those rows
//!      verbatim (the draws reader copies the trace's own strings).
//!
//! Before the emission fix, PMMH wrote no warm-up rows, so assertion (1) fails
//! (red). Skipped when the release binary / camdlc is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

const BURN_IN: usize = 20;
const ITERATIONS: usize = 60;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_pmmh_warmup_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Compile a small SIR to IR and write a short synthetic case series.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
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
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 20 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    let mut data = String::from("time\tcases\n");
    let cases = [
        2, 3, 5, 7, 10, 12, 15, 18, 20, 22,
        21, 19, 17, 15, 13, 11, 9, 7, 6, 5,
    ];
    for (i, c) in cases.iter().enumerate() {
        data.push_str(&format!("{}\t{}\n", i + 1, c));
    }
    std::fs::write(&data_path, &data).unwrap();
    (ir_path, data_path)
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.001, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 1.0 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.post]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 1
particles  = 40
iterations = {iters}
burn_in    = {burn}
thin       = 1
init       = "single"
"#,
        out  = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
        iters = ITERATIONS,
        burn  = BURN_IN,
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Recursively collect every file named `name` under `root`.
fn find_named(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Header column index by exact name.
fn col_index(header: &str, name: &str) -> usize {
    header.split('\t').position(|h| h == name)
        .unwrap_or_else(|| panic!("no `{name}` column in header: {header}"))
}

#[test]
fn pmmh_trace_carries_warmup_but_draws_excludes_it() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc_bin().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("inv");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data);

    let out = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit_toml)
        .args(["--seed", "1"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(),
        "pmmh `fit run` must succeed (exit 0):\n{}", String::from_utf8_lossy(&out.stderr));

    let results = tmp.path().join("results");
    let traces = find_named(&results, "trace.tsv");
    assert_eq!(traces.len(), 1, "exactly one chain trace.tsv (chains=1); got {traces:?}");
    let draws = find_named(&results, "draws.tsv");
    assert_eq!(draws.len(), 1, "exactly one draws.tsv; got {draws:?}");

    // ── Parse the trace: split rows by phase, keep post-burn-in (beta, gamma) ──
    let trace_txt = std::fs::read_to_string(&traces[0]).unwrap();
    let mut tlines = trace_txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let theader = tlines.next().expect("trace header");
    let (t_step, t_beta, t_gamma) =
        (col_index(theader, "step"), col_index(theader, "beta"), col_index(theader, "gamma"));

    let mut warmup_rows = 0usize;
    let mut post_params: Vec<(String, String)> = Vec::new();
    for l in tlines {
        let f: Vec<&str> = l.split('\t').collect();
        let step: usize = f[t_step].parse().expect("step parses");
        if step < BURN_IN {
            warmup_rows += 1;
        } else {
            post_params.push((f[t_beta].to_string(), f[t_gamma].to_string()));
        }
    }

    // (1) The fix: warm-up rows ARE present (red before the emission change).
    assert!(warmup_rows > 0,
        "trace.tsv must contain warm-up rows (step < {BURN_IN}) for live burn-in observability");
    assert!(!post_params.is_empty(), "trace.tsv must also contain post-burn-in rows");

    // ── Parse draws.tsv: header is `chain draw <params...>` ──
    let draws_txt = std::fs::read_to_string(&draws[0]).unwrap();
    let mut dlines = draws_txt.lines().filter(|l| !l.trim().is_empty());
    let dheader = dlines.next().expect("draws header");
    let (d_beta, d_gamma) = (col_index(dheader, "beta"), col_index(dheader, "gamma"));
    let draw_params: Vec<(String, String)> = dlines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[d_beta].to_string(), f[d_gamma].to_string())
    }).collect();

    // (2) draws.tsv == exactly the post-burn-in trace rows, verbatim. Warm-up is
    //     filtered back out of the posterior; the reader copies the trace's own
    //     estimated-param strings, so the ordered lists must be equal.
    assert_eq!(draw_params.len(), post_params.len(),
        "draws.tsv must have exactly one row per post-burn-in trace row \
         (warm-up excluded): draws={} post-burn-in={}", draw_params.len(), post_params.len());
    assert_eq!(draw_params, post_params,
        "each draws.tsv row must match its post-burn-in trace row's (beta, gamma) verbatim");
}
