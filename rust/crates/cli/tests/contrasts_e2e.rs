//! gh#322 stage C — end-to-end counterfactual `contrasts {}`: a PGAS chain_binomial
//! fit of a model with an SIA intervention + two scenarios + a contrast →
//! `fit predict` auto-emits `contrasts/<name>.tsv` (the two-arm CRN replay reducer).
//! There is no `over [..]` window: the fork is DERIVED (the last saved snapshot
//! before the toggled intervention fires).
//!
//! Asserts (1) a scalar "deaths averted" contrast has the band columns and a
//! positive median (the SIA mechanically averts deaths); (1b) the arms fork from
//! the smoothed X(fork) — the derived fork is reported and lands on an evolved
//! snapshot, NOT init at t=0; (2) a CRN sanity contrast of two scenarios toggling
//! a no-op (0%-transfer) intervention from the same X(fork) is identically zero;
//! (3) a point-estimate (IF2) fit emits NO contrast file (the LatentPath/posterior
//! gate); (4) a series−scalar contrast is a located shape-mismatch error.
//!
//! docs/dev/proposals/2026-06-25-counterfactual-contrasts.md (stage C).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    bin
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// `results/fits/<stem>-<hash>/<sub>/<stream>.tsv`, if present.
fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    let fits = root.join("fits");
    for e in std::fs::read_dir(&fits).ok()?.flatten() {
        let p = e.path().join(sub).join(format!("{stream}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// The first `chain_1/trajectories.tsv` under `results/fits/` — the fit's saved
/// smoothed paths. Recursive, since the stage/seed nesting under a fit dir varies.
fn find_chain_traj(root: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path) -> Option<PathBuf> {
        let hit = dir.join("chain_1").join("trajectories.tsv");
        if hit.is_file() {
            return Some(hit);
        }
        for e in std::fs::read_dir(dir).ok()?.flatten() {
            if e.path().is_dir() {
                if let Some(p) = walk(&e.path()) {
                    return Some(p);
                }
            }
        }
        None
    }
    walk(&root.join("fits"))
}

/// The derived fork time the reducer reported for `contrast` on stderr (the
/// `fork at t=<T>` note), proving the fork was DERIVED — not the old window, and
/// not init at t=0.
fn reported_fork_time(stderr: &str, contrast: &str) -> Option<f64> {
    let needle = format!("contrast '{contrast}' — fork at t=");
    let after = stderr.split(&needle).nth(1)?;
    let tok: String = after.chars().take_while(|c| !c.is_whitespace()).collect();
    tok.parse().ok()
}

/// Read the saved `S` compartment for the first row at time `t` (within a small
/// tolerance) in a tidy `trajectories.tsv` (columns: chain draw time S I R D V …).
fn saved_s_at(traj: &Path, t: f64) -> Option<f64> {
    let txt = std::fs::read_to_string(traj).ok()?;
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next()?.split('\t').collect();
    let ti = header.iter().position(|c| *c == "time")?;
    let si = header.iter().position(|c| *c == "S")?;
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let row_t: f64 = f.get(ti)?.parse().ok()?;
        if (row_t - t).abs() <= 1e-6 {
            return f.get(si)?.parse().ok();
        }
    }
    None
}

/// SIRD + SIA. The SIA (60% S→V) fires at week 4 (= 28 d); the fork is DERIVED as
/// the last saved snapshot strictly before 28 d, so both arms share X(fork) and
/// diverge only when the SIA fires. `no_sia`/`with_sia` toggle the real SIA. For
/// the CRN check, `noop` is a 0%-transfer intervention at the same time (firing it
/// is a no-op and RNG-free), and `null_a`/`null_b` enable/disable it — two distinct
/// arms with byte-identical dynamics, so their contrast is identically zero. This
/// also exercises the derived fork (a no-op intervention still drives the fork).
/// `total = final(D)`.
const MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia  : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
  noop : transfer(fraction = 0.0, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia]  }
  with_sia { enable  = [sia]  }
  null_a   { enable  = [noop] }
  null_b   { disable = [noop] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
contrasts {
  averted  = no_sia.quantities.total - with_sia.quantities.total
  crn_zero = null_a.quantities.total - null_b.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(algorithm_block: &str) -> String {
    format!(
        r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.5], start = 0.5 }}
gamma = {{ bounds = [0.05, 0.5], start = 0.15 }}
[fixed]
mu  = 0.05
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
{algorithm_block}
"#
    )
}

const PGAS: &str = r#"[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 120
sweeps = 40
burn_in = 15
thin = 1
"#;

const IF2: &str = r#"[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 120
iterations = 20
cooling = 0.5
"#;

/// Parse the single banded row of a scalar contrast TSV (header
/// `q05 q25 q50 q75 q95 mean n_used`) into its `q50`, `mean`, `n_used`.
fn scalar_band(path: &Path) -> (f64, f64, usize) {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let header = lines.next().expect("header");
    assert_eq!(
        header, "q05\tq25\tq50\tq75\tq95\tmean\tn_used",
        "scalar contrast band columns"
    );
    let row: Vec<&str> = lines.next().expect("one band row").split('\t').collect();
    assert_eq!(row.len(), 7, "row matches header");
    let q50: f64 = row[2].parse().unwrap();
    let mean: f64 = row[5].parse().unwrap();
    let n: usize = row[6].parse().unwrap();
    (q50, mean, n)
}

#[test]
fn fit_predict_emits_deaths_averted_contrast_with_positive_median() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_pgas_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // `--horizon free_forward`: the contrast is a forward counterfactual object;
    // this also sidesteps the one-step exact-filter's unrelated limitation with the
    // parametric `at [origin + 4 'weeks]` schedule. Contrasts are horizon-independent.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    let results = tmp.join("results");

    // (1) deaths averted: the SIA removes 60% of susceptibles at week 4, so the
    // counterfactual `with_sia` arm has FEWER deaths → `no_sia − with_sia > 0`.
    let averted = find_artifact(&results, "contrasts", "averted")
        .expect("contrasts/averted.tsv must be auto-emitted by fit predict");
    let (q50, mean, n_used) = scalar_band(&averted);
    assert!(n_used > 0, "the band is over a positive used-draw count, got {n_used}");
    assert!(
        q50 > 0.0,
        "the SIA averts deaths → median averted must be positive, got q50={q50} (mean={mean})"
    );
    assert!(mean > 0.0, "mean averted positive, got {mean}");

    // (1b) the arms fork from the smoothed X(fork), NOT init at t=0. The fork is
    // DERIVED: the reducer reports it, and it lands strictly inside (0, 28) — the
    // last saved snapshot before the SIA fires at day 28. (Were it forking from
    // init at t=0, there would be no derived-fork note / the fork would be 0.)
    let fork = reported_fork_time(&stderr, "averted")
        .unwrap_or_else(|| panic!("fit predict must report the derived fork; stderr:\n{stderr}"));
    assert!(
        fork > 0.0 && fork < 28.0,
        "fork must be derived strictly between t=0 and the SIA fire (t=28), got {fork}"
    );
    // The saved smoothed state at the fork is EVOLVED (S below init S0 = N0−I0 =
    // 9990): a growing epidemic by day {fork}. If the reducer forked from init, it
    // would inject S0 = 9990 at the fork instead of this evolved state.
    let traj = find_chain_traj(&results).expect("the fit must save a chain trajectory");
    let s_fork = saved_s_at(&traj, fork)
        .unwrap_or_else(|| panic!("no saved S at the derived fork t={fork} in {}", traj.display()));
    assert!(
        s_fork < 9990.0,
        "X(fork) must be the evolved smoothed state (S < init S0=9990), got S={s_fork} at t={fork}"
    );

    // (2) CRN sanity: two scenarios toggling a 0%-transfer no-op intervention,
    // forked from the SAME X(fork) with the SAME per-draw seed, are byte-identical
    // → the contrast is identically zero.
    let crn = find_artifact(&results, "contrasts", "crn_zero")
        .expect("contrasts/crn_zero.tsv must be emitted");
    let crn_txt = std::fs::read_to_string(&crn).unwrap();
    let row: Vec<&str> = crn_txt.lines().nth(1).expect("one row").split('\t').collect();
    for (i, cell) in row.iter().take(6).enumerate() {
        assert_eq!(
            *cell, "0",
            "CRN contrast must be identically zero (col {i} = {cell}); full row: {row:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn point_estimate_fit_emits_no_contrast_file() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_if2_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(IF2)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "if2 fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // An IF2 fit is a point estimate (no posterior cloud) → `fit predict` refuses
    // before any output. No contrast file is written, and the error names the gate.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml"]);
    assert!(!out.status.success(), "fit predict must refuse a point-estimate fit");
    let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(
        stderr.contains("optimizer") || stderr.contains("point") || stderr.contains("posterior"),
        "the refusal must name the posterior/point-estimate gate, got: {stderr}"
    );
    assert!(
        find_artifact(&tmp.join("results"), "contrasts", "averted").is_none(),
        "no contrast file may be written for a point-estimate fit"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Series − scalar: same dimension (both counts), different shape → the OCaml
/// frontend compiles it (shape is not a dimension), the Rust reducer rejects it.
const SHAPE_MISMATCH_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  infected = I              # series (no temporal reduce)
  total    = final(D)       # scalar
}
contrasts {
  bad = no_sia.quantities.infected - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

#[test]
fn series_minus_scalar_contrast_is_a_located_shape_error() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_shape_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), SHAPE_MISMATCH_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // The shape mismatch is a Rust reducer check (the OCaml dim check passes: both
    // operands are counts). `fit predict` fails with a located message.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(!out.status.success(), "a series−scalar contrast must fail fit predict");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("shape mismatch") && stderr.contains("'bad'"),
        "the error must name the shape mismatch and the contrast, got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A parameter-only counterfactual (`fitted` vs a `scale` scenario) toggles NO
/// intervention, so there is nothing to derive a fork from. With the window gone,
/// the reducer can no longer guess a fork — it must skip-with-note (gh#327) and
/// write no file, never silently mis-fork from init.
const PARAM_ONLY_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
scenarios {
  lower_trans { scale = { beta = 0.5 } }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
contrasts {
  param_only = fitted.quantities.total - lower_trans.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

#[test]
fn parameter_only_contrast_skips_with_a_located_note_and_no_file() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_paramonly_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), PARAM_ONLY_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // A param-only contrast has no toggled intervention → the reducer skips it with
    // a note (gh#327) and `fit predict` still succeeds. No contrast file is written.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict must succeed (the contrast is skipped, not a hard error):\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("skipping contrast 'param_only'")
            && stderr.contains("no toggled intervention")
            && stderr.contains("gh#327"),
        "the skip note must name the contrast, the missing toggle, and gh#327, got: {stderr}"
    );
    assert!(
        find_artifact(&tmp.join("results"), "contrasts", "param_only").is_none(),
        "no file may be written for a skipped parameter-only contrast"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── (gh#325) ODE deterministic forking is a named deferral ────────────────────

/// An ODE/MH-fit model with a `contrasts {}` block. The fit IS forkable (ODE is
/// `Deterministic`: X recomputes from θ), so the reducer passes the latent-state
/// gate — but ODE forking is not wired in this build, so the contrast is
/// skip-with-noted (gh#325) and NO file is written. Poisson observations; both
/// estimated params carry a `~` prior (MH requires priors for every estimate).
const ODE_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate  in [0.05, 1.5] ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate  in [0.05, 0.5] ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate  in [0.0, 0.3]
  N0    : count
  I0    : count
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  prevalence {
    columns       { time : time, prevalence : count }
    projected     = prevalence(I)
    emit_schedule = every 4 'days
    prevalence    ~ poisson(rate = projected)
  }
}
quantities {
  total = final(D)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 60 'days }
"#;

const PREVALENCE_DATA: &str = "time\tprevalence\n4\t30\n8\t90\n12\t260\n16\t640\n20\t980\n24\t1100\n28\t900\n32\t620\n36\t400\n40\t250\n44\t150\n48\t90\n52\t55\n56\t33\n60\t20\n";

const ODE_MH: &str = r#"[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 60
burn_in = 20
thin = 1
"#;

#[test]
fn ode_fit_with_contrasts_skips_with_gh325_note_and_no_file() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_ode_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), ODE_MODEL).unwrap();
    std::fs::write(tmp.join("prevalence.tsv"), PREVALENCE_DATA).unwrap();
    let fit = format!(
        r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
prevalence = "prevalence.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.5], start = 0.5 }}
gamma = {{ bounds = [0.05, 0.5], start = 0.15 }}
[fixed]
mu = 0.05
N0 = 10000
I0 = 10
{ODE_MH}
"#
    );
    std::fs::write(tmp.join("fit.toml"), fit).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "mh+ode fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // ODE is forkable (Deterministic), so the reducer passes the latent gate but
    // hits the ODE-backend deferral: skip-with-note (gh#325), predict succeeds,
    // no file is written.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict must succeed (the contrast is skipped, not a hard error):\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("skipping") && stderr.contains("ode") && stderr.contains("gh#325"),
        "the ODE skip note must name the deferral and gh#325, got: {stderr}"
    );
    assert!(
        find_artifact(&tmp.join("results"), "contrasts", "averted").is_none(),
        "no contrast file may be written for an ODE fit (gh#325 deferral)"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── (gh#326) observation-sourced operands are a named deferral ─────────────────

/// Two contrasts on one chain_binomial fit: a STATE-sourced `total = final(D)`
/// contrast that emits, and an OBS-sourced `<run>.observations.weekly_cases`
/// contrast that is skip-with-noted (gh#326). Predict still succeeds; only the
/// state-sourced file is written.
const OBS_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
contrasts {
  state_averted = no_sia.quantities.total - with_sia.quantities.total
  obs_diff      = no_sia.observations.weekly_cases - with_sia.observations.weekly_cases
}
simulate { from = 0 'days  to = 80 'days }
"#;

#[test]
fn obs_sourced_contrast_skips_with_gh326_state_sourced_emits() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_obs_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), OBS_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict must succeed (the obs-sourced contrast is skipped, not fatal):\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("skipping contrast 'obs_diff'") && stderr.contains("gh#326"),
        "the obs-sourced skip note must name the contrast and gh#326, got: {stderr}"
    );

    // The state-sourced contrast emits; the obs-sourced one writes no file.
    assert!(
        find_artifact(&tmp.join("results"), "contrasts", "state_averted").is_some(),
        "the state-sourced contrast must be emitted"
    );
    assert!(
        find_artifact(&tmp.join("results"), "contrasts", "obs_diff").is_none(),
        "no file may be written for the deferred obs-sourced contrast"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── Stratified-quantity contrast → per-stratum output ─────────────────────────

/// A patch-stratified SIRD+V. Scalar parameters (so the fit.toml is the flat
/// form), but the COMPARTMENTS are stratified, so the quantity
/// `final_D[p in patch]` expands to one leaf per patch. The SIA fires on patch
/// `a` only, so `no_sia − with_sia` is a per-patch contrast: positive for `a`
/// (deaths averted), near zero for `b`. The output carries a `patch` dim column
/// and one row per stratum.
const STRAT_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
dimensions { patch = [a, b] }
compartments { S, I, R, D, V }
stratify(by = patch)
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N[p in patch] = S[p] + I[p] + R[p] + D[p] + V[p]
transitions {
  infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
  death[p in patch]     : I[p] --> D[p]  @ mu * I[p]
}
init { S[p in patch] = N0 - I0   I[p in patch] = I0 }
interventions {
  sia_a : transfer(fraction = 0.6, from = S[a], to = V[a]) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia_a] }
  with_sia { enable  = [sia_a] }
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection_a)
    emit_schedule = every 7 'days
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  final_D[p in patch] = final(D[p])
}
contrasts {
  averted = no_sia.quantities.final_D - with_sia.quantities.final_D
}
simulate { from = 0 'days  to = 80 'days }
"#;

const STRAT_DATA: &str =
    "time\tcases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

#[test]
fn stratified_quantity_contrast_emits_per_stratum_output() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_strat_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), STRAT_MODEL).unwrap();
    std::fs::write(tmp.join("cases.tsv"), STRAT_DATA).unwrap();
    let fit = format!(
        r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
cases = "cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.5], start = 0.5 }}
gamma = {{ bounds = [0.05, 0.5], start = 0.15 }}
[fixed]
mu  = 0.05
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
{PGAS}
"#
    );
    std::fs::write(tmp.join("fit.toml"), fit).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let averted = find_artifact(&tmp.join("results"), "contrasts", "averted")
        .expect("contrasts/averted.tsv must be emitted for the stratified contrast");
    let txt = std::fs::read_to_string(&averted).unwrap();
    let lines: Vec<&str> = txt.lines().collect();
    let header: Vec<&str> = lines[0].split('\t').collect();

    // A per-stratum contrast carries the `patch` dim column and ends in n_used.
    assert_eq!(header[0], "patch", "first column is the stratum dim: {:?}", header);
    assert_eq!(*header.last().unwrap(), "n_used", "band columns end with n_used: {:?}", header);

    // One row per stratum (patch a, patch b) — no aggregation.
    let rows: Vec<Vec<&str>> = lines[1..].iter().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 2, "two strata → two rows, got {}: {:?}", rows.len(), rows);
    let patches: Vec<&str> = rows.iter().map(|r| r[0]).collect();
    assert!(patches.contains(&"a") && patches.contains(&"b"), "rows cover both patches: {patches:?}");

    // Patch `a` is where the SIA fires → deaths averted is positive there.
    let q50_i = header.iter().position(|c| *c == "q50").unwrap();
    let a_row = rows.iter().find(|r| r[0] == "a").unwrap();
    let a_q50: f64 = a_row[q50_i].parse().unwrap();
    assert!(a_q50 > 0.0, "the SIA averts deaths on patch a → median > 0, got {a_q50}");

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── gh#561: the arms reading refuses a per-scenario horizon ──────────────────
//
// A per-scenario `simulate { to }` is honoured on the `simulate` path — a MENU
// of runs, each answering its own question (see `tests/scenario_horizon.rs`).
// A contrast DIFFERENCES its operands, so unequal windows compare the windows
// rather than the counterfactual, and `fit predict` emits at the OBSERVED times
// so it cannot move its window at all. Both refuse rather than silently ignore.

/// `MODEL` reduced to one contrast, with a horizon declared on ONE of its two
/// arms — so the arms disagree (model horizon 80, `with_sia` asks for 200).
const MODEL_RAGGED_CONTRAST_ARMS: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia]  simulate { to = 200 'days } }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

/// The case a pairwise arm-vs-arm guard misses: BOTH arms declare the same
/// non-model horizon. This is the natural authoring pattern — the fit window in
/// `simulate {}`, the projection window on the arms — so it is the one that
/// matters most.
const MODEL_BOTH_ARMS_AGREE_OFF_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia]  simulate { to = 200 'days } }
  with_sia { enable  = [sia]  simulate { to = 200 'days } }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

/// gh#561: the guard must compare each arm against the horizon the replay
/// ACTUALLY uses (`model.simulation.t_end`), not the arms against each other.
///
/// With both arms at `to = 200` and a model horizon of 80, a pairwise check
/// sees 200 == 200 and passes — and `run_end = model.simulation.t_end` then
/// replays both to 80. The deaths-averted number is `final(D)` at day 80 for a
/// question the author posed at day 200, with nothing in the artifact saying
/// so: gh#561's own silent drop surviving inside its fix.
#[test]
fn contrast_arms_agreeing_on_a_non_model_horizon_are_refused() {
    let bin = skip_if_missing_binary();
    let tmp =
        std::env::temp_dir().join(format!("camdl_contrasts_bothoff_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL_BOTH_ARMS_AGREE_OFF_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let out =
        run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "both arms declaring t = 200 against a model horizon of 80 must be \
         refused — the replay uses 80 either way; it succeeded.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("declares a simulation horizon"),
        "the refusal must name the declared-vs-replayed mismatch; stderr:\n{stderr}"
    );
    let results = tmp.join("results");
    assert!(
        find_artifact(&results, "contrasts", "averted").is_none(),
        "a refused contrast must not leave a contrasts/averted.tsv behind"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn contrast_arms_with_different_horizons_are_refused() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_horizon_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL_RAGGED_CONTRAST_ARMS).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let out =
        run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "a contrast whose arms declare different horizons must be refused, not \
         differenced; it succeeded.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("declares a simulation horizon"),
        "the refusal must name the declared-vs-replayed mismatch; stderr:\n{stderr}"
    );
    // The offending arm and both numbers, so the reader can act without
    // re-reading the model.
    assert!(
        stderr.contains("with_sia") && stderr.contains("200") && stderr.contains("80"),
        "the refusal must name the offending arm and both horizons; \
         stderr:\n{stderr}"
    );

    // No contrast file is left behind — a refused contrast must not leave an
    // artifact a reader could mistake for a valid one.
    let results = tmp.join("results");
    assert!(
        find_artifact(&results, "contrasts", "averted").is_none(),
        "a refused contrast must not leave a contrasts/averted.tsv behind"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The same shape with NO `contrasts {}` block, so the `fit predict` guard is
/// exercised in isolation — with a contrast present the (correct) contrast-arm
/// refusal fires first and would mask it.
const MODEL_SCENARIO_HORIZON_NO_CONTRAST: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia]  simulate { to = 200 'days } }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total = final(D)
}
simulate { from = 0 'days  to = 80 'days }
"#;

#[test]
fn fit_predict_scenario_with_its_own_horizon_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_horizon_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL_SCENARIO_HORIZON_NO_CONTRAST).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    // PGAS, not IF2: `fit predict` gates on a posterior cloud BEFORE it reaches
    // the scenario overlay, so an optimizer fit would fail on that gate instead
    // and never exercise this guard.
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // `with_sia` declares `to = 200`; the model horizon is 80.
    let out = run(
        &bin,
        &tmp,
        &[
            "fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward", "--scenario",
            "with_sia",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "fit predict must refuse a scenario whose horizon it cannot honour; it \
         succeeded.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("replays every scenario at the model's own horizon"),
        "the refusal must say WHY the horizon cannot be honoured; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("camdl simulate --scenario"),
        "the refusal must point at the command that DOES run the scenario's own \
         window; stderr:\n{stderr}"
    );

    // Negative control: the sibling scenario declares no horizon, so it is
    // unaffected — the guard fires on a genuine difference, not on the mere
    // presence of a per-scenario `simulate {}` anywhere in the model.
    let out = run(
        &bin,
        &tmp,
        &[
            "fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward", "--scenario",
            "no_sia",
        ],
    );
    assert!(
        out.status.success(),
        "a scenario with no declared horizon must still predict:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── gh#694: an observation-anchored `value_at` quantity + a `contrasts {}` block

/// One banded row of a `fit predict` quantity sidecar, by column name. A
/// censorable scalar's header is
/// `scenario n_draws n_value n_censored p_censored q05 … q95`; `fit predict`
/// writes one row per scenario and these fixtures predict a single (`fitted`)
/// scenario, so the first data row is the row.
fn quantity_row(path: &Path) -> std::collections::HashMap<String, String> {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let row: Vec<&str> = lines.next().expect("one band row").split('\t').collect();
    assert_eq!(row.len(), header.len(), "row matches header in {}", path.display());
    header.iter().map(|h| h.to_string()).zip(row.iter().map(|v| v.to_string())).collect()
}

/// `MODEL`'s SIRD+SIA with three anchored `value_at` quantities added, none of
/// which any contrast references. `at_literal` pins the anchor's expected value:
/// the data ends at t = 56, and origin 2020-01-01 + 56 d = 2020-02-26, so a
/// correctly resolved `last_obs` reads exactly where the literal date does.
const ANCHORED_QUANTITY_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total       = final(D)
  at_last_obs = value_at(D, last_obs)
  at_literal  = value_at(D, date("2020-02-26"))
  at_first    = value_at(D, first_obs)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

/// gh#694: declaring `contrasts {}` must not cost the model its observation
/// anchors. `fit predict` holds the fit's data and resolves `last_obs` for the
/// ordinary `quantities/` sidecar; the contrast arms evaluate the same quantity
/// list, so an anchored quantity NO contrast references must not fail the
/// command — and the resolved value must be the one the data implies, not
/// merely a non-error.
#[test]
fn anchored_quantity_no_contrast_references_still_predicts() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_contrasts_anchor_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), ANCHORED_QUANTITY_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "an anchored quantity beside a contrasts block must not fail fit predict \
         (gh#694); stderr:\n{stderr}"
    );

    let results = tmp.join("results");
    let quantity = |name: &str| {
        quantity_row(
            &find_artifact(&results, "quantities", name)
                .unwrap_or_else(|| panic!("quantities/{name}.tsv must be written")),
        )
    };

    // The anchor resolved, and resolved to the END OF DATA (t = 56): the band is
    // cell-for-cell the literal-date quantity's. Anchoring anywhere else — the
    // horizon (t = 80), the fork, first_obs — moves at least one quantile.
    let last = quantity("at_last_obs");
    assert_eq!(last["n_censored"], "0", "last_obs = t 56 is inside [0, 80]: {last:?}");
    assert!(
        last["n_value"].parse::<usize>().unwrap() > 0,
        "the anchored quantity must band over real draws: {last:?}"
    );
    let literal = quantity("at_literal");
    assert_eq!(
        last, literal,
        "value_at(D, last_obs) must read exactly where value_at(D, date(\"2020-02-26\")) \
         does — the data ends at t = 56"
    );

    // first_obs (t = 7) and last_obs (t = 56) must resolve SEPARATELY: D is
    // non-decreasing, so the earlier read is strictly smaller. Equal medians
    // would mean both anchors collapsed onto one time.
    let first = quantity("at_first");
    let (f50, l50) =
        (first["q50"].parse::<f64>().unwrap(), last["q50"].parse::<f64>().unwrap());
    assert!(
        f50 < l50,
        "first_obs (t 7) and last_obs (t 56) must resolve to different times; \
         got q50 {f50} vs {l50}"
    );

    // The contrast itself is untouched by the anchored quantities beside it.
    let averted = find_artifact(&results, "contrasts", "averted")
        .expect("contrasts/averted.tsv must still be emitted");
    let (q50, _, n_used) = scalar_band(&averted);
    assert!(
        n_used > 0 && q50 > 0.0,
        "the deaths-averted contrast still bands: q50={q50} n_used={n_used}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The same fixture with the contrast taken OVER the anchored quantity, so the
/// arms themselves must resolve `last_obs` — the ordinary `quantities/` path
/// resolving it is not enough.
const ANCHORED_CONTRAST_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  total       = final(D)
  at_last_obs = value_at(D, last_obs)
  at_literal  = value_at(D, date("2020-02-26"))
}
contrasts {
  averted             = no_sia.quantities.total       - with_sia.quantities.total
  averted_at_last_obs = no_sia.quantities.at_last_obs - with_sia.quantities.at_last_obs
  averted_at_literal  = no_sia.quantities.at_literal  - with_sia.quantities.at_literal
}
simulate { from = 0 'days  to = 80 'days }
"#;

/// gh#694: a contrast whose operands ARE the anchored quantity. The SIA fires at
/// t = 28 and the data ends at t = 56, so the anchor is inside both arms'
/// `[fork, 80]` replay window — every draw must yield a value (n_used > 0), and
/// the SIA must still show as deaths averted by the end of the data.
///
/// The literal-time twin is the value oracle: the arms replay identically for
/// every contrast (same fork, same per-draw seeds), so a `last_obs` operand and
/// a `date("2020-02-26")` operand must produce the SAME file. Resolving the
/// anchor anywhere else — the horizon, the fork — moves it.
#[test]
fn contrast_over_an_anchored_quantity_resolves_in_both_arms() {
    let bin = skip_if_missing_binary();
    let tmp =
        std::env::temp_dir().join(format!("camdl_contrasts_anchor_op_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), ANCHORED_CONTRAST_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "a contrast over an anchored quantity must predict (gh#694); stderr:\n{stderr}"
    );

    let results = tmp.join("results");
    let anchored = find_artifact(&results, "contrasts", "averted_at_last_obs")
        .expect("contrasts/averted_at_last_obs.tsv must be emitted");
    let (q50, mean, n_used) = scalar_band(&anchored);
    assert!(
        n_used > 0,
        "last_obs (t 56) lies inside the arms' replay window, so no draw may \
         censor: n_used={n_used}"
    );
    assert!(
        q50 > 0.0 && mean > 0.0,
        "the SIA averts deaths by the end of the data too: q50={q50} mean={mean}"
    );

    // The arms resolved `last_obs` to t = 56 exactly: the literal-date twin
    // contrast, replayed from the same fork with the same per-draw seeds, is the
    // same file cell for cell.
    let literal = find_artifact(&results, "contrasts", "averted_at_literal")
        .expect("contrasts/averted_at_literal.tsv must be emitted");
    assert_eq!(
        std::fs::read_to_string(&anchored).unwrap(),
        std::fs::read_to_string(&literal).unwrap(),
        "a `last_obs` operand must read exactly where a date(\"2020-02-26\") \
         operand does — the data ends at t = 56"
    );

    // The sibling final-horizon contrast is unaffected.
    assert!(
        find_artifact(&results, "contrasts", "averted").is_some(),
        "the final(D) contrast must still be emitted alongside"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The one case an anchor genuinely cannot be read in an arm. The SIA fires at
/// t = 28, so the arms replay `[fork = 21, 80]`, while `first_obs` is t = 7 —
/// BEFORE the fork, outside the window the arms simulated. It right-censors,
/// exactly as an out-of-window literal time does. (A bare `last_obs` cannot
/// land there: the fork is the last SAVED snapshot before the toggle, and the
/// saved smoothed paths stop at the end of the data, so `fork ≤ last_obs`
/// always.)
const EARLY_ANCHOR_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  at_first_obs = value_at(D, first_obs)
}
contrasts {
  averted_at_first_obs = no_sia.quantities.at_first_obs - with_sia.quantities.at_first_obs
}
simulate { from = 0 'days  to = 80 'days }
"#;

/// gh#694: the narrow case that keeps a diagnostic. An anchor resolving OUTSIDE
/// the arms' replay window is right-censored (the existing censor-not-clamp
/// contract), not refused — but the reader is told, per contrast and by name,
/// why the band came back empty. Predict still succeeds and every other
/// artifact is written.
#[test]
fn an_anchor_before_the_fork_censors_with_a_located_note() {
    let bin = skip_if_missing_binary();
    let tmp =
        std::env::temp_dir().join(format!("camdl_contrasts_earlyanchor_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), EARLY_ANCHOR_MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "an out-of-window anchor censors, it does not fail the command; stderr:\n{stderr}"
    );
    // The note is located: the contrast, the quantity, the resolved anchor time,
    // and the window the arms actually ran.
    assert!(
        stderr.contains("averted_at_first_obs")
            && stderr.contains("at_first_obs")
            && stderr.contains("outside the arms' replay window"),
        "the out-of-window anchor must be reported per contrast and by name; \
         stderr:\n{stderr}"
    );

    // The band is honestly empty rather than fabricated: no draw contributed.
    let path = find_artifact(&tmp.join("results"), "contrasts", "averted_at_first_obs")
        .expect("contrasts/averted_at_first_obs.tsv must still be written");
    let txt = std::fs::read_to_string(&path).unwrap();
    let row: Vec<&str> = txt.lines().nth(1).expect("one band row").split('\t').collect();
    assert_eq!(
        *row.last().unwrap(),
        "0",
        "every draw censored → n_used = 0, no fabricated band: {row:?}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
