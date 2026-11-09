//! End-to-end tests for the smooth-importation seed mechanism (mechanism B of
//! the 2026-05-20 seed-timing-inference proposal), exercised through the CLI.
//!
//! Model: `crates/sim/tests/fixtures/seed_timing.ir.json` — a SIR seeded by a
//! logistic importation pulse `lambda / (1 + exp(-(t - tau)/w))` into the
//! `#[lineage]`-tracked `I` compartment.
//!
//! Coverage:
//!   1. Dynamics respond to `tau`, AND Gillespie agrees with chain-binomial.
//!      This is the end-to-end regression for the frozen-propensity bug
//!      (incident 2026-05-20-gillespie-bare-time-frozen-propensity): before the
//!      fix, a late seed produced zero inflow and no epidemic on Gillespie.
//!   2. The seed inflow is Import-rooted in the realized line list (the §8
//!      lineage contract): a source-less inflow into a tracked compartment is
//!      minted with `parent = Import`, with no `#[lineage]` annotation.
//!   3. A particle-filter likelihood profile over `tau` is peaked at the true
//!      seed time — `tau` is identifiable (E1: seed size fixed).
//!
//! Silent-skip if the release `camdl` binary is not built (same convention as
//! `lineage_e2e.rs`). `CAMDL_SKIP_VERSION_CHECK=1` avoids a stale globally
//! installed `camdlc` making the test flaky.

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

fn model_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json")
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_seed_e2e_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl).env_remove("CLICOLOR_FORCE")
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// Common parameter overrides (everything except `tau`, which the caller sets).
const BASE_PARAMS: &[&str] = &[
    "--param", "beta=0.6",
    "--param", "gamma=0.2",
    "--param", "lambda=2.0",
    "--param", "w=3.0",
    "--param", "N0=5000",
    "--param", "rho=0.5",
    "--param", "k=20",
];

/// Parse a trajectory TSV (skipping `#` comment lines): returns (peak_I,
/// onset_t, total_seed_inflow). Columns: t S I R flow_infection flow_recovery
/// flow_seed.
fn traj_summary(path: &Path) -> (i64, Option<f64>, i64) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut header: Vec<&str> = vec![];
    let (mut peak_i, mut onset, mut seed_inflow) = (0i64, None, 0i64);
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if header.is_empty() {
            header = cols;
            continue;
        }
        let col = |name: &str| header.iter().position(|h| *h == name).map(|i| cols[i]);
        let t: f64 = col("t").unwrap().parse().unwrap();
        let i: i64 = col("I").unwrap().parse().unwrap();
        let fseed: i64 = col("flow_seed").map(|v| v.parse().unwrap()).unwrap_or(0);
        peak_i = peak_i.max(i);
        seed_inflow += fseed;
        if onset.is_none() && i > 50 {
            onset = Some(t);
        }
    }
    (peak_i, onset, seed_inflow)
}

/// Run a forward simulation and return its trajectory summary.
fn simulate_summary(camdl: &Path, backend: &str, tau: f64, out: &Path) -> (i64, Option<f64>, i64) {
    let ir = model_ir();
    let tau_arg = format!("tau={tau}");
    let mut args = vec![
        "simulate", ir.to_str().unwrap(),
        "--backend", backend, "--dt", "1", "--seed", "7",
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.push("--param");
    args.push(&tau_arg);
    args.extend(["--output", out.to_str().unwrap()]);
    let o = run(camdl, &args);
    assert!(o.status.success(), "simulate ({backend}) failed: {}", String::from_utf8_lossy(&o.stderr));
    traj_summary(out)
}

/// 1. Dynamics respond to `tau`, and Gillespie matches chain-binomial.
#[test]
fn seed_pulse_shifts_onset_and_backends_agree() {
    let camdl = camdl_bin();
    let tmp = tempdir("dyn");

    for backend in ["gillespie", "chain_binomial"] {
        let early = simulate_summary(&camdl, backend, 15.0, &tmp.join(format!("{backend}_15.tsv")));
        let late = simulate_summary(&camdl, backend, 45.0, &tmp.join(format!("{backend}_45.tsv")));

        // An epidemic must occur in BOTH cases. The late-seed Gillespie run is
        // the direct regression: pre-fix it produced 0 seed inflow / no epidemic.
        assert!(early.0 > 200, "{backend} tau=15: expected an epidemic, peak_I={}", early.0);
        assert!(late.0 > 200, "{backend} tau=45: expected an epidemic, peak_I={} (frozen-propensity regression?)", late.0);
        assert!(late.2 > 0, "{backend} tau=45: seed inflow must be > 0 (frozen-propensity regression?), got {}", late.2);

        // A later seed → a later epidemic onset.
        let (e_onset, l_onset) = (early.1.expect("early onset"), late.1.expect("late onset"));
        assert!(l_onset > e_onset, "{backend}: later seed (tau=45) should onset after tau=15: {l_onset} vs {e_onset}");
    }

    // Cross-backend agreement on total seed inflow at tau=30 (within 30% — the
    // residual is Gillespie's output-grid re-evaluation vs chain-binomial's
    // per-substep evaluation, not the freeze).
    let g = simulate_summary(&camdl, "gillespie", 30.0, &tmp.join("g30.tsv")).2 as f64;
    let c = simulate_summary(&camdl, "chain_binomial", 30.0, &tmp.join("c30.tsv")).2 as f64;
    assert!(g > 0.0 && c > 0.0, "both backends must seed at tau=30: g={g} c={c}");
    let rel = (g - c).abs() / c;
    assert!(rel < 0.30, "Gillespie/chain-binomial seed inflow disagree: g={g} c={c} (rel={rel:.2})");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// 2. The seed inflow is Import-rooted in the line list (§8 contract).
#[test]
fn seed_inflow_is_import_rooted() {
    let camdl = camdl_bin();
    let tmp = tempdir("import");
    let ir = model_ir();
    let ev = tmp.join("ev.tsv");
    let traj = tmp.join("traj.tsv");

    let mut args = vec![
        "simulate", ir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "7",
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.extend(["--param", "tau=20", "--event-log", ev.to_str().unwrap(), "--tsv", "--output", traj.to_str().unwrap()]);
    let o = run(&camdl, &args);
    assert!(o.status.success(), "simulate --event-log failed: {}", String::from_utf8_lossy(&o.stderr));

    let ll = tmp.join("ll.tsv");
    let r = run(&camdl, &["lineage", "realize", ev.to_str().unwrap(), "--identity-seed", "7", "-o", ll.to_str().unwrap()]);
    assert!(r.status.success(), "lineage realize failed: {}", String::from_utf8_lossy(&r.stderr));

    // Count parent_kind values. The seed pulse founders must be `import`-rooted;
    // transmissions must be `individual`-rooted.
    let text = std::fs::read_to_string(&ll).unwrap();
    let mut header: Vec<&str> = vec![];
    let (mut n_import, mut n_individual) = (0usize, 0usize);
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() { continue; }
        let cols: Vec<&str> = line.split('\t').collect();
        if header.is_empty() { header = cols; continue; }
        let pk = header.iter().position(|h| *h == "parent_kind").map(|i| cols[i]);
        match pk {
            Some("import") => n_import += 1,
            Some("individual") => n_individual += 1,
            _ => {}
        }
    }
    assert!(n_import > 0, "seed inflow must produce Import-rooted line-list entries, got {n_import}");
    assert!(n_individual > 0, "transmissions must produce individual-rooted entries, got {n_individual}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Parse the loglik (last numeric stdout line) from a `camdl pfilter` run.
fn pfilter_loglik(camdl: &Path, data: &Path, tau: f64, particles: &str, seed: &str) -> f64 {
    let ir = model_ir();
    let tau_arg = format!("tau={tau}");
    let mut args = vec![
        "pfilter", ir.to_str().unwrap(),
        "--particles", particles, "--dt", "1", "--seed", seed,
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.push("--param");
    args.push(&tau_arg);
    let o = run(camdl, &args);
    assert!(o.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik in pfilter output:\n{stdout}"))
}

/// 3. The particle-filter likelihood profile over `tau` is peaked at the true
///    seed time — `tau` is identifiable when the seed size is fixed (E1).
#[test]
fn likelihood_profile_identifies_seed_time() {
    let camdl = camdl_bin();
    let tmp = tempdir("ident");
    let ir = model_ir();

    // Generate synthetic daily cases at tau_true = 30.
    let data = tmp.join("cases.tsv");
    let mut args = vec![
        "simulate", ir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "11",
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.extend(["--param", "tau=30", "--obs-only", data.to_str().unwrap()]);
    let o = run(&camdl, &args);
    assert!(o.status.success(), "synthetic-data simulate failed: {}", String::from_utf8_lossy(&o.stderr));

    // Profile the likelihood. 2000 particles, fixed seed → the peak is stable
    // against Monte-Carlo noise; the asserted margins are far larger than it.
    let ll_15 = pfilter_loglik(&camdl, &data, 15.0, "2000", "5");
    let ll_30 = pfilter_loglik(&camdl, &data, 30.0, "2000", "5");
    let ll_45 = pfilter_loglik(&camdl, &data, 45.0, "2000", "5");

    // True tau beats an early guess by a wide margin (profile ~10 nats; assert 3).
    assert!(
        ll_30 > ll_15 + 3.0,
        "loglik should favour true tau=30 over tau=15: {ll_30:.2} vs {ll_15:.2}"
    );
    // A seed later than the observed onset cannot explain the early cases.
    assert!(
        ll_30 > ll_45 + 3.0,
        "loglik should favour true tau=30 over too-late tau=45: {ll_30:.2} vs {ll_45:.2}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
