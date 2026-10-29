//! Regression pins for IF2's default steps in z (support-keyed transforms
//! proposal D8). The defaults — 1/20 on the interval and log-interval maps,
//! 1/5 on the half-bounded ones — are starting heuristics chosen for
//! robustness to poor initialisation, not optimal values; these tests pin the
//! behaviour that chose them, on the closed SIR of
//! `tests/recovery/cases/sir` and its data in `tests/bench/support-transforms`
//! (where `make far` and `make grid` run the full benchmark).
//!
//! The basin is the closed SIR's maximum: the best clean log-likelihood any
//! benchmark run on these data reached is −58.26, and a chain is in the basin
//! within 3 nats of it. Each test also runs the step it guards against, so
//! that a pass is a contrast and not a property of the fixture.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Within 3 nats of −58.26, the best value any benchmark run reached.
const BASIN: f64 = -61.26;

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test"))
}

fn camdl_bin() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    p.exists().then_some(p)
}

fn bench_dir() -> PathBuf {
    manifest().join("../../../tests/bench/support-transforms")
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_if2step_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Fit `fixture`'s model (under `tests/bench/support-transforms/fixtures`) to
/// the closed SIR's weekly cases with IF2: 4 chains × 1000 particles,
/// `iterations`, cooling 0.7, seed 1. Returns the output and each chain's
/// clean log-likelihood (empty when the fit failed).
fn fit(camdlc: &Path, fixture: &str, estimate: &str, iterations: usize, starts: &str) -> (Output, Vec<f64>) {
    let tmp = tempdir(fixture);
    let dir = &tmp.0;
    let src = bench_dir().join("fixtures").join(fixture).join("model.camdl");
    let out = Command::new(camdlc).arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc {}: {}", src.display(), String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::copy(bench_dir().join("fixtures/sir/data/weekly_cases.tsv"), dir.join("cases.tsv")).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(
        "output_dir = \"out\"\n[model]\ncamdl = \"m.ir.json\"\n[data.observations]\n\
         weekly_cases = \"cases.tsv\"\n[estimate]\n{estimate}\n[fixed]\nN0 = 10000\nI0 = 10\n\
         rho = 0.6\nk = 10.0\n[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
         chains = 4\nparticles = 1000\niterations = {iterations}\ncooling = 0.7\nstarts = \"{starts}\"\n"
    )).unwrap();
    let out = Command::new(camdl_bin())
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"])
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .unwrap();
    let mut lls = Vec::new();
    let mut stack = vec![dir.join("out")];
    while let Some(d) = stack.pop() {
        let Ok(es) = std::fs::read_dir(&d) else { continue };
        for e in es.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("chain_evaluations.tsv") {
                let text = std::fs::read_to_string(&p).unwrap();
                let mut lines = text.lines().filter(|l| !l.starts_with('#'));
                let col = lines.next().unwrap().split('\t').position(|h| h == "loglik").unwrap();
                lls.extend(lines.map(|l| l.split('\t').nth(col).unwrap().parse::<f64>().unwrap()));
            }
        }
    }
    (out, lls)
}

fn best(lls: &[f64]) -> f64 {
    lls.iter().copied().filter(|x| x.is_finite()).fold(f64::NEG_INFINITY, f64::max)
}

/// Far start, half-bounded maps: β and γ with no range ([0, ∞)) started at 4
/// and 2, ten and thirteen times the truth (0.4, 0.15). The default step, 1/5
/// in z, reaches the basin in 60 iterations; a step of 1/50 leaves every chain
/// where the filter cannot score it, and the watchdog stops them all. The
/// benchmark's `far` rows show the same for every step at or below 1/10.
#[test]
fn the_one_sided_default_step_reaches_the_basin_from_a_far_start() {
    let Some(camdlc) = camdlc_bin() else { return };
    let (out, lls) = fit(&camdlc, "sir_lb", "beta = { start = 4.0 }\ngamma = { start = 2.0 }", 60, "single");
    assert!(out.status.success(), "the default step must fit from the far start:\n{}",
        String::from_utf8_lossy(&out.stderr));
    assert!(best(&lls) >= BASIN, "best clean log-likelihood {} is outside the basin (>= {BASIN}): {lls:?}",
        best(&lls));
    // A step of 1/50 in z: the natural-scale rw_sd is converted at the start,
    // where dθ/dz = θ.
    let (out, _) = fit(&camdlc, "sir_lb",
        "beta = { start = 4.0, rw_sd = 0.08 }\ngamma = { start = 2.0, rw_sd = 0.04 }", 60, "single");
    assert!(!out.status.success(),
        "contrast: a step of 1/50 should not survive this start, or the test pins nothing:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

/// Far coordinate, interval map: γ : rate in [0, 0.5] started at 0.45, three
/// times the truth, with β near it. The default step, 1/20 in z, finishes
/// within 0.45 nats of the maximum in 30 iterations; 1/50 has not
/// arrived (about −59.2) and the old midpoint-equivalent 2/3 overshoots
/// (about −80).
#[test]
fn the_interval_default_step_finishes_at_the_maximum_from_a_far_coordinate() {
    let Some(camdlc) = camdlc_bin() else { return };
    let est = |gamma_extra: &str| format!(
        "beta = {{ bounds = [0.05, 1.0], start = 0.4 }}\n\
         gamma = {{ bounds = [0.0, 0.5], start = 0.45{gamma_extra} }}");
    let finish = -58.7;
    let (out, lls) = fit(&camdlc, "sir_0hi", &est(""), 30, "single");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(best(&lls) >= finish, "default step: best {} < {finish}: {lls:?}", best(&lls));
    // rw_sd = c·(hi − lo)/4, converted at the midpoint, is a step of c in z.
    for (c, rw) in [("1/50", 0.5 / 50.0 / 4.0), ("2/3", 0.5 * 2.0 / 3.0 / 4.0)] {
        let (_, lls) = fit(&camdlc, "sir_0hi", &est(&format!(", rw_sd = {rw}")), 30, "single");
        assert!(best(&lls) < finish, "contrast: a step of {c} reached {} >= {finish}, so the test pins nothing",
            best(&lls));
    }
}

/// Spread starts, the default condition: `lhs` over the wide box lb : real in
/// [−8, 2], with the transmission rate exp(lb). The default step, 1/20 in z,
/// reaches the basin in 60 iterations; the old midpoint-equivalent 2/3 leaves
/// the swarm tens of nats short.
#[test]
fn the_boxed_default_steps_reach_the_basin_from_spread_starts() {
    let Some(camdlc) = camdlc_bin() else { return };
    let gamma = "gamma = { bounds = [0.01, 0.5], start = 0.10 }";
    let (out, lls) = fit(&camdlc, "sir_wide",
        &format!("lb = {{ bounds = [-8.0, 2.0], start = -1.386 }}\n{gamma}"), 60, "lhs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(best(&lls) >= BASIN, "default step: best {} < {BASIN}: {lls:?}", best(&lls));
    let (_, lls) = fit(&camdlc, "sir_wide",
        &format!("lb = {{ bounds = [-8.0, 2.0], start = -1.386, rw_sd = {} }}\n{gamma}", 10.0 * 2.0 / 3.0 / 4.0),
        60, "lhs");
    assert!(best(&lls) < BASIN, "contrast: a step of 2/3 reached {} >= {BASIN}, so the test pins nothing",
        best(&lls));
}
