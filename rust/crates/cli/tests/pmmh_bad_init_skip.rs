//! gh#110 — PMMH skip-and-continue on PFDegenerate init.
//!
//! End-to-end test for the init-eval guard wired into
//! `pmmh::run_stage`. We construct a two-chain PMMH fit whose
//! `starts = { from_posterior = … }` draws file holds two rows:
//!
//!   - pathological β=4.8, γ=0.05 → R0 ≈ 96 against a flat-low data
//!     series. PF reweights kill all but ~1 particle within a handful
//!     of obs windows → ESS collapse → `Err(SimError::PFDegenerate)`.
//!   - sane β=0.30, γ=0.10 → R0 = 3, fits the data.
//!
//! Each chain draws one row; which chain gets which is a property of the
//! seeded draw, read back from `chain_starts.tsv` rather than assumed.
//!
//! Acceptance:
//!   1. `camdl fit run` exits 0 (the run does NOT fail when one
//!      chain's init triggers PFDegenerate — surviving chains
//!      continue).
//!   2. `diagnostics.json` contains a `bad_init` diagnostic for the
//!      pathological chain. The variant tag uses the snake-case
//!      rename declared on `DiagnosticKind`.
//!   3. `fit_state.toml` reports `n_good_chains = 1` (the good
//!      chain's MAP), distinct from `n_chains = 2`.
//!   4. The good chain wrote its `chain_<n>/trace.tsv` with post-burn-in
//!      rows.
//!
//! Skipped when the release binary or camdlc isn't present.

use std::path::{Path, PathBuf};
use std::process::Command;

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
        "camdl_pmmh_bad_init_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// SIR fixture with wide enough bounds that β=4.8 (pathological) is
/// inside the search space. The data is a small outbreak that levels
/// off — incompatible with R0 ≈ 96 dynamics.
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
simulate { from = 0 'days  to = 30 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    // 30 days of low daily case counts. Compatible with R0 ≈ 3
    // (good init); astronomically incompatible with R0 ≈ 96
    // (pathological init) → PF reweight kills all but one particle
    // within a handful of windows → ESS collapse trigger fires.
    let data_path = dir.join("cases.tsv");
    let mut data = String::from("time\tcases\n");
    let cases = [
        2, 3, 5, 7, 10, 12, 15, 18, 20, 22,
        21, 19, 17, 15, 13, 11, 9, 7, 6, 5,
        4, 4, 3, 3, 2, 2, 2, 1, 1, 1,
    ];
    for (i, c) in cases.iter().enumerate() {
        data.push_str(&format!("{}\t{}\n", i + 1, c));
    }
    std::fs::write(&data_path, &data).unwrap();

    (ir_path, data_path)
}

/// A two-row draws file: the pathological start and the sane one.
///
/// Row 1: β=4.8, γ=0.05. R0 = β/γ ≈ 96 with N=1000 → epidemic peaks within
/// ~3 days, incompatible with the flat case series → ESS collapse.
/// Row 2: β=0.30, γ=0.10 → R0 = 3.
fn write_draws(dir: &Path) -> PathBuf {
    let draws = "\
# gh#110 PMMH BadInit skip-and-continue test fixture\n\
beta\tgamma\n\
4.80\t0.05\n\
0.30\t0.10\n";
    let p = dir.join("starts.tsv");
    std::fs::write(&p, draws).unwrap();
    p
}

fn write_fit_toml(
    dir: &Path,
    ir: &Path,
    data: &Path,
    draws: &Path,
) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.001, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 1.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[method]
algorithm      = "pmmh"
backend        = "chain_binomial"
chains         = 2
particles      = 30
iterations     = 40
burn_in        = 5
thin           = 1
starts         = {{ from_posterior = "{draws}" }}
"#,
        out    = dir.join("results").display(),
        ir     = ir.display(),
        data   = data.display(),
        draws  = draws.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// The CAS method leaf for `stage_substr` under `fits_root` —
/// `<fit>/<method>-<h8>/seed_<N>-<h8>/` holding a `fit_stage` run.json.
fn cas_stage_leaf(fits_root: &Path, stage_substr: &str) -> PathBuf {
    let mut stack = vec![fits_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("run.json").is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(d.join("run.json")).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"].as_array().into_iter().flatten()
                        .find(|l| l["name"].as_str() == Some("method"))
                        .and_then(|l| l["label"].as_str()).unwrap_or("");
                    if stage.contains(stage_substr) { return d; }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no CAS '{}' method leaf under {}", stage_substr, fits_root.display());
}

/// Each chain's `beta` start, read from the leaf's `chain_starts.tsv` —
/// `(chain_id, beta)` in file order.
fn chain_betas(leaf: &Path) -> Vec<(usize, f64)> {
    let path = leaf.join("chain_starts.tsv");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut body = raw.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let id_idx = cols.iter().position(|c| *c == "chain_id").expect("chain_id column");
    let beta_idx = cols.iter().position(|c| *c == "beta").expect("beta column");
    body.map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        (cells[id_idx].parse().unwrap(), cells[beta_idx].parse().unwrap())
    }).collect()
}

/// gh#110 acceptance: a pathological drawn start must not hang the run — the
/// chain is skipped with a `BadInit` diagnostic and the sane chain completes.
#[test]
fn pmmh_skips_pathological_drawn_start_and_continues() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("skip");
    let (ir, data) = write_fixture(tmp.path());

    let draws = write_draws(tmp.path());
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &draws);
    let t0 = std::time::Instant::now();
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--seed", "1", "--progress", "none"])
        .output().expect("spawn camdl fit run");
    let elapsed = t0.elapsed();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Acceptance 1: exit success.
    assert!(out.status.success(),
        "pmmh fit must succeed when one chain hits PFDegenerate on init.\n\
         elapsed: {:?}\nstdout:\n{}\nstderr:\n{}",
        elapsed, stdout, stderr);

    // Sanity check: the run did not hang. Watchdog wall-clock is 120s
    // per PF call; this whole fit (incl. one watchdog-bailed PF +
    // ~40 iter on the good chain @ 30 particles) should be well
    // under that.
    assert!(elapsed.as_secs() < 240,
        "fit must complete well under the 120s-per-call watchdog \
         budget; took {:?}.\nstderr:\n{}", elapsed, stderr);

    // Acceptance 2: diagnostics.json contains a `bad_init` entry.
    let fits_dir = tmp.path().join("results/fits");
    let stage_dir = cas_stage_leaf(&fits_dir, "pmmh");
    assert!(stage_dir.join("run.json").is_file(),
        "pmmh method leaf missing run.json: {}\nstderr:\n{}", stage_dir.display(), stderr);

    // The premise: the seeded draw handed the two chains different rows.
    let starts = chain_betas(&stage_dir);
    let bad_chain = starts.iter().find(|(_, b)| (b - 4.8).abs() < 1e-9).map(|(c, _)| *c)
        .unwrap_or_else(|| panic!("no chain drew the pathological row: {starts:?}"));
    let good_chain = starts.iter().find(|(_, b)| (b - 0.3).abs() < 1e-9).map(|(c, _)| *c)
        .unwrap_or_else(|| panic!(
            "no chain drew the sane row: {starts:?}. If the from_posterior draw \
             changed, pick a seed under which the two chains draw different rows."));

    let diag_path = stage_dir.join("diagnostics.json");
    assert!(diag_path.exists(),
        "diagnostics.json must be written under {}\nstderr:\n{}",
        stage_dir.display(), stderr);
    let diag_raw = std::fs::read_to_string(&diag_path).unwrap();
    let diags: serde_json::Value = serde_json::from_str(&diag_raw)
        .expect("diagnostics.json must be valid JSON");
    let arr = diags.as_array().expect("diagnostics.json is an array");
    let n_bad = arr.iter()
        .filter(|d| d.get("kind").and_then(|k| k.get("type"))
            .and_then(|t| t.as_str()) == Some("bad_init"))
        .count();
    assert_eq!(n_bad, 1,
        "expected exactly 1 BadInit diagnostic; full diagnostics.json:\n{}\n\
         stderr:\n{}", diag_raw, stderr);

    // The BadInit entry must carry the pathological chain's index and its
    // β / γ pair.
    let bad = arr.iter().find(|d|
        d.get("kind").and_then(|k| k.get("type"))
            .and_then(|t| t.as_str()) == Some("bad_init"))
        .unwrap();
    let bad_kind = bad.get("kind").unwrap();
    let chain_id = bad_kind.get("chain_id").and_then(|c| c.as_u64())
        .expect("BadInit must carry a chain_id") as usize;
    assert_eq!(chain_id, bad_chain,
        "the refused chain must be the one that drew β=4.8; got chain_id={}.\n\
         BadInit:\n{}", chain_id, serde_json::to_string_pretty(bad).unwrap());

    let params = bad_kind.get("params").expect("BadInit must carry params");
    let beta = params.get("beta").and_then(|v| v.as_f64())
        .expect("BadInit.params must include beta");
    assert!((beta - 4.8).abs() < 1e-9,
        "BadInit.params.beta should = 4.8 (the pathological row); got {}", beta);

    // Acceptance 3: fit_state.toml reports n_good_chains = 1.
    let state_path = stage_dir.join("fit_state.toml");
    assert!(state_path.exists(),
        "fit_state.toml must be written\nstderr:\n{}", stderr);
    let state_raw = std::fs::read_to_string(&state_path).unwrap();
    let state: toml::Value = toml::from_str(&state_raw).unwrap();
    let n_good = state.get("n_good_chains").and_then(|v| v.as_integer())
        .expect("fit_state.toml must record n_good_chains when a chain \
                 was skipped (gh#110)");
    assert_eq!(n_good, 1,
        "n_good_chains should be 1 (the sane chain only). \
         fit_state.toml:\n{}", state_raw);
    let n_chains = state.get("n_chains").and_then(|v| v.as_integer())
        .expect("n_chains field");
    assert_eq!(n_chains, 2,
        "n_chains should remain 2 (the requested chain count). \
         fit_state.toml:\n{}", state_raw);

    // Acceptance 4: the good chain produced a trace with posterior draws.
    // The skipped chain may have a trace.tsv header but should not have
    // post-burn-in rows (its loop never ran).
    let good_trace = stage_dir.join(format!("chain_{}/trace.tsv", good_chain + 1));
    assert!(good_trace.exists(),
        "{} must exist for the surviving chain", good_trace.display());
    let good_lines = std::fs::read_to_string(&good_trace).unwrap()
        .lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .count();
    // Header + at least one post-burn-in draw. iterations=40, burn_in=5,
    // thin=1 → ~35 draws expected.
    assert!(good_lines >= 5,
        "{} should have header + post-burn-in draws; got {} non-comment lines",
        good_trace.display(), good_lines);

    // Stderr should surface the user-facing "ran 1 of 2 chains" line.
    assert!(stderr.contains("ran 1 of 2 chains"),
        "stderr must surface 'ran 1 of 2 chains'; got:\n{}", stderr);
}
