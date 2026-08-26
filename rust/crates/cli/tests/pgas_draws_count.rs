//! Regression: `camdl fit run` (PGAS) must persist EVERY post-burn-in, thinned
//! posterior draw to `draws.tsv` — exactly `(n_sweeps − burn_in)` retained,
//! sub-sampled by `thin`. The sim-side recorder already applies burn-in + thin
//! when it builds the returned sweeps; the draws.tsv writer used to apply them a
//! SECOND time (indexing the already-retained list by position), silently
//! dropping the first `burn_in` retained draws — half the posterior at `thin=1`,
//! and ALL of it once `thin` ≥ the retained count. R̂/ESS, computed over the full
//! retained set, then disagreed with the truncated `draws.tsv`.
//!
//! Incident: docs/dev/incidents/2026-06-28-pgas-draws-double-thinning.md

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The number of draws the sim-side recorder retains — the SAME filter
/// (`sweep >= burn_in && (sweep − burn_in) % thin == 0` over one chain), so the
/// expectation is correct even when `thin` does not divide `n_sweeps − burn_in`.
fn expected_draws(n_sweeps: usize, burn_in: usize, thin: usize, n_chains: usize) -> usize {
    (burn_in..n_sweeps).filter(|s| (s - burn_in).is_multiple_of(thin)).count() * n_chains
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn draws_row_count(out_root: &Path) -> usize {
    let fits = out_root.join("fits");
    for e in std::fs::read_dir(&fits).unwrap().flatten() {
        // draws.tsv lives at <segment>/<NN-stage>/seed_*/draws.tsv
        for d in walk(&e.path()) {
            if d.file_name().map(|n| n == "draws.tsv").unwrap_or(false) {
                let txt = std::fs::read_to_string(&d).unwrap();
                // rows minus the header line
                return txt.lines().count().saturating_sub(1);
            }
        }
    }
    panic!("no draws.tsv under {}", fits.display());
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// Run one PGAS fit and assert `draws.tsv` carries exactly the retained count.
fn assert_draws_count(slug: &str, n_chains: usize, sweeps: usize, burn_in: usize, thin: usize) {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_pgas_count_{slug}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    let fit_toml = format!(
        r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = {n_chains}
particles = 100
sweeps = {sweeps}
burn_in = {burn_in}
thin = {thin}
"#
    );
    std::fs::write(tmp.join("fit.toml"), fit_toml).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let want = expected_draws(sweeps, burn_in, thin, n_chains);
    let got = draws_row_count(&tmp.join("results"));
    assert_eq!(
        got, want,
        "draws.tsv must carry every retained draw \
         (n_chains={n_chains}, sweeps={sweeps}, burn_in={burn_in}, thin={thin}); \
         got {got}, want {want} — a double-applied burn-in/thin drops the first \
         burn_in retained draws"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn pgas_draws_tsv_keeps_every_retained_draw_thin1() {
    // 1 chain, 30 sweeps, burn_in 10, thin 1 → 20 retained. The double-apply
    // dropped the first 10 retained draws → 10 (this is the red→green case).
    assert_draws_count("thin1", 1, 30, 10, 1);
}

#[test]
fn pgas_draws_tsv_keeps_every_retained_draw_thin2() {
    // 1 chain, 30 sweeps, burn_in 10, thin 2 → 10 retained (sweeps 10,12,…,28).
    // The double-apply skipped index < burn_in(10) → ALL 10 dropped → 0.
    assert_draws_count("thin2", 1, 30, 10, 2);
}
