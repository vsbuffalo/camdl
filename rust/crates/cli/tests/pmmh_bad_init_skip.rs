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
//! `from_posterior` is a spread rule, so a start the init-eval refuses is
//! redrawn, up to `MAX_START_ATTEMPTS` (gh#887), with every attempt and the
//! ESS the filter reached on the record.
//!
//! Acceptance:
//!   1. `camdl fit run` exits 0 and no chain is skipped: the chain that
//!      drew the pathological row draws again and runs.
//!   2. `chain_starts.tsv` carries a `rejected` row for each pathological
//!      draw, with the `EssCollapsed` reason and the ESS at refusal, and an
//!      `accepted` row at the sane point.
//!   3. `fit_state.toml` carries no `n_good_chains` (every chain ran).
//!   4. Both chains wrote `chain_<n>/trace.tsv` with post-burn-in rows.
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

/// One row of the leaf's `chain_starts.tsv`: `(chain_id, attempt, status,
/// beta, ess, reason)` in file order.
fn start_rows(leaf: &Path) -> Vec<(usize, usize, String, f64, Option<f64>, String)> {
    let path = leaf.join("chain_starts.tsv");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut body = raw.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let col = |name: &str| cols.iter().position(|c| *c == name)
        .unwrap_or_else(|| panic!("{name} column in {cols:?}"));
    let (id, attempt, status, beta, ess, reason) =
        (col("chain_id"), col("attempt"), col("status"), col("beta"), col("ess"), col("reason"));
    body.map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        (
            cells[id].parse().unwrap(),
            cells[attempt].parse().unwrap(),
            cells[status].to_string(),
            cells[beta].parse().unwrap(),
            cells[ess].parse().ok(),
            cells[reason].to_string(),
        )
    }).collect()
}

/// gh#110 under gh#887: a pathological drawn start must not hang the run —
/// the init-eval refuses it, the chain draws again, and every attempt is on
/// the record with the ESS the filter reached.
#[test]
fn pmmh_redraws_a_pathological_drawn_start_and_runs_every_chain() {
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

    // Acceptance 1: exit success, and no chain skipped.
    assert!(out.status.success(),
        "pmmh fit must succeed when one chain's first draw hits PFDegenerate on init.\n\
         elapsed: {:?}\nstdout:\n{}\nstderr:\n{}",
        elapsed, stdout, stderr);

    // Sanity check: the run did not hang. Watchdog wall-clock is 120s
    // per PF call; this whole fit (incl. one watchdog-bailed PF +
    // ~40 iter on the good chain @ 30 particles) should be well
    // under that.
    assert!(elapsed.as_secs() < 240,
        "fit must complete well under the 120s-per-call watchdog \
         budget; took {:?}.\nstderr:\n{}", elapsed, stderr);

    // Acceptance 2: the record. The chain that drew β=4.8 has a rejected row
    // for each such draw — the EssCollapsed refusal and the ESS the filter
    // reached — and an accepted row at the sane point.
    let fits_dir = tmp.path().join("results/fits");
    let stage_dir = cas_stage_leaf(&fits_dir, "pmmh");
    assert!(stage_dir.join("run.json").is_file(),
        "pmmh method leaf missing run.json: {}\nstderr:\n{}", stage_dir.display(), stderr);
    let rows = start_rows(&stage_dir);
    let rejected: Vec<_> = rows.iter().filter(|r| r.2 == "rejected").collect();
    let accepted: Vec<_> = rows.iter().filter(|r| r.2 == "accepted").collect();
    assert!(!rejected.is_empty(),
        "the seeded draw must hand at least one chain the pathological row for this \
         fixture to exercise a redraw; rows: {rows:?}. If the draw changed, pick a \
         seed under which one chain draws it.");
    assert_eq!(accepted.len(), 2, "every chain ends on an accepted start: {rows:?}");
    assert!(rows.iter().all(|r| r.2 != "refused"), "no chain is refused: {rows:?}");
    for r in &rejected {
        assert!((r.3 - 4.8).abs() < 1e-9, "only the pathological row is rejected: {r:?}");
        assert!(r.5.contains("EssCollapsed"), "the refusal is the filter's: {r:?}");
        assert!(r.4.is_some(), "the ESS at refusal is recorded: {r:?}");
    }
    for a in &accepted {
        assert!((a.3 - 0.3).abs() < 1e-9, "the accepted start is the sane row: {a:?}");
    }
    let diag_path = stage_dir.join("diagnostics.json");
    if diag_path.exists() {
        let diags: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&diag_path).unwrap()).unwrap();
        let n_bad = diags.as_array().map(|arr| arr.iter()
            .filter(|d| d.get("kind").and_then(|k| k.get("type"))
                .and_then(|t| t.as_str()) == Some("bad_init"))
            .count()).unwrap_or(0);
        assert_eq!(n_bad, 0, "a redrawn chain is not a refused chain:\n{stderr}");
    }

    // Acceptance 3: every chain ran, so n_good_chains is unset.
    let state_raw = std::fs::read_to_string(stage_dir.join("fit_state.toml")).unwrap();
    let state: toml::Value = toml::from_str(&state_raw).unwrap();
    assert!(state.get("n_good_chains").is_none(),
        "every chain ran, so n_good_chains is unset:\n{state_raw}");
    assert_eq!(state.get("n_chains").and_then(|v| v.as_integer()), Some(2), "{state_raw}");

    // Acceptance 4: both chains produced a trace with posterior draws.
    for chain in 1..=2 {
        let trace = stage_dir.join(format!("chain_{chain}/trace.tsv"));
        assert!(trace.exists(), "{} must exist", trace.display());
        let lines = std::fs::read_to_string(&trace).unwrap()
            .lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count();
        assert!(lines >= 5, "{} should have header + post-burn-in draws; got {lines}", trace.display());
    }

    // The redraw was loud, and nothing was skipped.
    assert!(stderr.contains("refused by the filter") && stderr.contains("drawing another"),
        "the redraw must be announced; got:\n{}", stderr);
    assert!(!stderr.contains("ran 1 of 2 chains"),
        "nothing was skipped; got:\n{}", stderr);
}
