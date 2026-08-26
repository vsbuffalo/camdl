//! Integration test for `camdl fit run --resume` against PGAS.
//!
//! Verifies the wiring landed in 2026-04-30 (`Stage::identity_payload`
//! split + `--resume` flag plumbed through the dispatcher):
//!
//! 1. A first PGAS run writes `chain_<n>/resume_state.bin` containing
//!    completed_sweeps == n_sweeps and the stage's identity hash.
//! 2. A second invocation with `--resume --stage post --sweeps N>n_sweeps`
//!    succeeds and continues the chain (does not re-run burn-in).
//! 3. Changing an *identity* field (e.g. `chains`) between the two
//!    invocations causes resume to reject with a hash-mismatch error.
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
        "camdl_pgas_resume_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Build a tiny SIR model + Poisson obs IR and write trivial data so
/// PGAS can run end-to-end in seconds.
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
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    // Tiny dataset — 6 days, low counts.
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, sweeps: usize, chains: usize) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0],  prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = {chains}
particles = 30
sweeps = {sweeps}
# Tiny burn_in so the post-burn-in sample set is non-empty even with
# small `sweeps`. (PGAS panics if sweeps <= burn_in; default is 2000.)
burn_in = 2
"#,
        out = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join(format!("fit_{}_{}.toml", sweeps, chains));
    std::fs::write(&p, toml).unwrap();
    p
}

/// The `(run_id, dir, run_json)` of a `post` stage leaf under `<out>/fits/`,
/// skipping any run_id in `exclude` (to pick the resumed leaf after a resume).
fn post_leaf(out: &Path, exclude: &[String]) -> (String, PathBuf, serde_json::Value) {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"].as_array().into_iter().flatten()
                        .find(|l| l["name"].as_str() == Some("stage"))
                        .and_then(|l| l["label"].as_str()).unwrap_or("");
                    let rid = v["run_id"].as_str().unwrap_or("").to_string();
                    if stage.contains("post") && !exclude.contains(&rid) {
                        return (rid, d, v);
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no post stage leaf under {} (excluding {:?})", out.join("fits").display(), exclude);
}

/// Recursive `{relpath -> bytes}` snapshot of a leaf, for the base-untouched check.
fn snapshot(dir: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); }
            else { out.insert(p.strip_prefix(dir).unwrap().to_path_buf(), std::fs::read(&p).unwrap()); }
        }
    }
    out
}

/// gh#147 (M3.2): `--resume <base ref>` reads the base leaf read-only and
/// writes a distinct resumed leaf keyed on the new target_length with a dep on
/// the base. Covers: the base run is byte-identical before/after; the resumed
/// run gets a distinct run_id; the resumed leaf deps on the prior; and a
/// chained resume (8→16→24) deps on the *actual* immediate prior, not the
/// original base.
#[test]
fn pgas_resume_writes_distinct_leaf_with_base_untouched_and_dep() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("continues");
    let (ir, data) = write_fixture(tmp.path());
    let out = tmp.path().join("results");

    // Run 1: 8 sweeps → base leaf.
    let fit8 = write_fit_toml(tmp.path(), &ir, &data, 8, 1);
    let r1 = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit8).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r1.status.success(), "first PGAS run failed: {}", String::from_utf8_lossy(&r1.stderr));
    let (base_id, base_dir, _) = post_leaf(&out, &[]);
    assert!(base_dir.join("chain_1/resume_state.bin").exists(),
        "base must write chain_1/resume_state.bin");
    let base_before = snapshot(&base_dir);

    // Run 2: 16 sweeps, --resume <base run_id>. Distinct leaf; base read-only.
    let fit16 = write_fit_toml(tmp.path(), &ir, &data, 16, 1);
    let r2 = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit16)
        .arg("--seed").arg("1").arg("--stage").arg("post").arg("--resume").arg(&base_id)
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&r2.stderr);
    assert!(r2.status.success(), "resume run must succeed: {}", stderr);
    assert!(stderr.contains("resuming from sweep"), "must announce resumption: {}", stderr);

    let (resumed_id, _, resumed_json) = post_leaf(&out, std::slice::from_ref(&base_id));
    assert_ne!(resumed_id, base_id, "resumed run must have a distinct run_id");
    assert_eq!(snapshot(&base_dir), base_before, "the base leaf must be untouched by resume");
    let deps = serde_json::to_string(&resumed_json["deps"]).unwrap();
    assert!(deps.contains(&base_id), "resumed leaf must dep on the base {}; deps={}", base_id, deps);

    // Run 3 (chained): resume the *resumed* leaf, 16 → 24. The dep must point
    // at the actual prior (the 16-sweep leaf), not the original base.
    let fit24 = write_fit_toml(tmp.path(), &ir, &data, 24, 1);
    let r3 = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit24)
        .arg("--seed").arg("1").arg("--stage").arg("post").arg("--resume").arg(&resumed_id)
        .output().expect("spawn");
    assert!(r3.status.success(), "chained resume must succeed: {}", String::from_utf8_lossy(&r3.stderr));
    let (third_id, _, third_json) = post_leaf(&out, &[base_id.clone(), resumed_id.clone()]);
    assert_ne!(third_id, resumed_id, "chained resume must re-key");
    let deps3 = serde_json::to_string(&third_json["deps"]).unwrap();
    assert!(deps3.contains(&resumed_id),
        "chained resume must dep on the actual prior {}; deps={}", resumed_id, deps3);
}

/// gh#261: the PGAS per-sweep trace's log-likelihood column holds the
/// COMPLETE-DATA conditional value (IVP + transition + observation density
/// along the conditioned path), not a marginal/PF likelihood. The header must
/// name it as such (`log_complete_data_ll`), never a bare `log_likelihood` a
/// reader would mistake for a `camdl pfilter` marginal loglik.
///
/// gh#667: the same header must carry the `transition_ll` / `obs_ll`
/// decomposition. `fit summary` compares chains on `obs_ll` — because
/// `log_complete_data_ll` is evaluated at each chain's OWN latent path and is
/// not comparable across chains — so these two are not optional diagnostics
/// any more; a header without `obs_ll` costs the per-chain outlier table its
/// input.
#[test]
fn pgas_trace_loglik_column_names_complete_data() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("trace_header");
    let (ir, data) = write_fixture(tmp.path());
    let out = tmp.path().join("results");

    let fit8 = write_fit_toml(tmp.path(), &ir, &data, 8, 1);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit8).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r.status.success(), "PGAS run failed: {}", String::from_utf8_lossy(&r.stderr));

    let (_, base_dir, _) = post_leaf(&out, &[]);
    let trace = base_dir.join("chain_1/trace.tsv");
    let text = std::fs::read_to_string(&trace).expect("read trace.tsv");
    let header = text.lines().next().expect("trace has a header line");
    let cols: Vec<&str> = header.split('\t').collect();
    assert!(cols.contains(&"log_complete_data_ll"),
        "PGAS trace must name its complete-data loglik column; header was: {header}");
    assert!(!cols.contains(&"log_likelihood"),
        "PGAS trace must NOT use a bare `log_likelihood` (mistaken for the marginal); header was: {header}");
    assert!(cols.contains(&"obs_ll"),
        "gh#667: `fit summary` compares PGAS chains on `obs_ll` = log p(y | X, θ); \
         header was: {header}");
    assert!(cols.contains(&"transition_ll"),
        "gh#667: the latent-path term is what makes the complete-data spread \
         readable as an entropy effect; header was: {header}");

    // …and a data row carries finite values for both, so the columns are not
    // merely declared. The first sweep suffices — a PGAS sweep always scores
    // its conditioned path against the data.
    let hdr_idx = |name: &str| cols.iter().position(|c| *c == name).unwrap();
    let row: Vec<&str> = text.lines().nth(1).expect("at least one trace row").split('\t').collect();
    for name in ["obs_ll", "transition_ll"] {
        let v: f64 = row[hdr_idx(name)].parse()
            .unwrap_or_else(|e| panic!("{name} must parse as f64: {e}; row was {row:?}"));
        assert!(v.is_finite() && v < 0.0, "{name} must be a finite log-density, got {v}");
    }
}

/// gh#294: the PGAS per-sweep trace surfaces the standard cold-chain NUTS
/// diagnostics (`tree_depth`, `n_leapfrog`, `step_size`, `accept_stat`,
/// `n_divergent`, `energy`) so the dominant `θ|X` cost is observable. Asserts
/// both the header names and that a data row carries sane values (a NUTS step
/// actually ran: n_leapfrog ≥ 1, a positive finite step_size, accept_stat in
/// [0,1], a finite energy).
#[test]
fn pgas_trace_emits_nuts_diagnostics() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("trace_nuts");
    let (ir, data) = write_fixture(tmp.path());
    let out = tmp.path().join("results");

    let fit8 = write_fit_toml(tmp.path(), &ir, &data, 8, 1);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit8).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r.status.success(), "PGAS run failed: {}", String::from_utf8_lossy(&r.stderr));

    let (_, base_dir, _) = post_leaf(&out, &[]);
    let trace = base_dir.join("chain_1/trace.tsv");
    let text = std::fs::read_to_string(&trace).expect("read trace.tsv");
    let mut lines = text.lines();
    let header = lines.next().expect("trace has a header line");
    let cols: Vec<&str> = header.split('\t').collect();
    for c in ["tree_depth", "n_leapfrog", "step_size", "accept_stat", "n_divergent", "energy"] {
        assert!(cols.contains(&c),
            "PGAS trace must carry the `{c}` NUTS column (gh#294); header was: {header}");
    }
    let idx = |name: &str| cols.iter().position(|&c| c == name).unwrap();
    let (i_lf, i_ss, i_as, i_div, i_e) =
        (idx("n_leapfrog"), idx("step_size"), idx("accept_stat"), idx("n_divergent"), idx("energy"));

    let row = lines.find(|l| !l.trim().is_empty()).expect("trace has a data row");
    let f: Vec<&str> = row.split('\t').collect();
    let n_leapfrog: usize = f[i_lf].parse().expect("n_leapfrog parses");
    let step_size: f64 = f[i_ss].parse().expect("step_size parses");
    let accept_stat: f64 = f[i_as].parse().expect("accept_stat parses");
    let n_divergent: usize = f[i_div].parse().expect("n_divergent parses");
    let energy: f64 = f[i_e].parse().expect("energy parses");

    assert!(n_leapfrog >= 1, "a NUTS step must take ≥1 leapfrog; row was: {row}");
    assert!(step_size > 0.0 && step_size.is_finite(), "step_size must be positive finite; got {step_size}");
    assert!((0.0..=1.0).contains(&accept_stat), "accept_stat must be a probability; got {accept_stat}");
    assert!(n_divergent <= 1, "per-sweep n_divergent is 0 or 1 (one NUTS step); got {n_divergent}");
    // `energy = H0 = -log_p + KE` is legitimately `+inf` when a sweep starts
    // from a degenerate (-inf log-posterior) point — a faithful diagnostic, not
    // a bug. NaN would be the bug.
    assert!(!energy.is_nan(), "energy must not be NaN; got {energy}");
}

/// (3): changing an identity field (chains) between base and resume must reject
/// with a config-hash mismatch — the copied resume_state's identity hash
/// (chains=1) won't match the new run (chains=2).
#[test]
fn pgas_resume_rejects_when_identity_field_changes() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("rejects");
    let (ir, data) = write_fixture(tmp.path());
    let out = tmp.path().join("results");

    // Run 1: 1 chain → base.
    let fit1 = write_fit_toml(tmp.path(), &ir, &data, 8, 1);
    let r1 = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit1).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r1.status.success(), "first PGAS run failed: {}", String::from_utf8_lossy(&r1.stderr));
    let (base_id, _, _) = post_leaf(&out, &[]);

    // Run 2: 2 chains (an identity field) + --resume <base>. Reject.
    let fit2 = write_fit_toml(tmp.path(), &ir, &data, 8, 2);
    let r2 = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit2)
        .arg("--seed").arg("1").arg("--stage").arg("post").arg("--resume").arg(&base_id)
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&r2.stderr);
    assert!(!r2.status.success(), "resume with changed chains must reject");
    assert!(stderr.contains("config hash mismatch") || stderr.contains("no resume state"),
        "expected config-hash-mismatch error: got {}", stderr);
}

/// gh#280: PGAS reports a complete-data loglik, so its live progress feed must
/// read `ll(complete)=`, not the bare `ll=` that means a marginal for every
/// other method. `--progress plain` bumps verbosity to Info (main.rs) so the
/// throttled per-sweep line reaches stderr; the first sweep always emits.
#[test]
fn pgas_plain_progress_feed_marks_complete_data() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("plain_complete");
    let (ir, data) = write_fixture(tmp.path());
    let toml = write_fit_toml(tmp.path(), &ir, &data, 8, 1);

    let out = Command::new(&bin)
        .arg("fit").arg("run").arg(&toml)
        .arg("--seed").arg("1")
        .arg("--progress").arg("plain")
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "PGAS fit must run: {}", stderr);
    assert!(stderr.contains("ll(complete)="),
        "PGAS plain-progress feed must carry the complete-data prefix \
         `ll(complete)=`, not a bare `ll=`:\n{stderr}");
}
