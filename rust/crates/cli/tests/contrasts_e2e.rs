//! gh#322 stage C — end-to-end counterfactual `contrasts {}`: a PGAS chain_binomial
//! fit of a model with an SIA intervention + two scenarios + a contrast →
//! `fit predict` auto-emits `contrasts/<name>.tsv` (the two-arm CRN replay reducer).
//!
//! Asserts (1) a scalar "deaths averted" contrast has the band columns and a
//! positive median (the SIA mechanically averts deaths); (2) a CRN sanity contrast
//! of two no-op scenarios from the same X(T*) is identically zero; (3) a
//! point-estimate (IF2) fit emits NO contrast file (the LatentPath/posterior gate);
//! (4) a series−scalar contrast is a located shape-mismatch error.
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

/// SIRD + SIA. The SIA (60% S→V) fires at week 4 (= 28 d), which is AFTER the fork
/// (week 2 = 14 d) and inside the accumulation window, so both arms share X(T*=14)
/// and diverge only when the SIA fires. `no_sia`/`with_sia` toggle it; `null_a`/
/// `null_b` are no-op (×1.0) scenarios for the CRN check. `total = final(D)`.
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
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
  null_a   { scale = { beta = 1.0 } }
  null_b   { scale = { beta = 1.0 } }
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
  averted  = no_sia.quantities.total - with_sia.quantities.total over [origin + 2 'weeks, origin + 10 'weeks]
  crn_zero = null_a.quantities.total - null_b.quantities.total   over [origin + 2 'weeks, origin + 10 'weeks]
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
/// `q05 q25 q50 q75 q95 mean n_forkable`) into its `q50`, `mean`, `n_forkable`.
fn scalar_band(path: &Path) -> (f64, f64, usize) {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let header = lines.next().expect("header");
    assert_eq!(
        header, "q05\tq25\tq50\tq75\tq95\tmean\tn_forkable",
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

    let results = tmp.join("results");

    // (1) deaths averted: the SIA removes 60% of susceptibles at week 4, so the
    // counterfactual `with_sia` arm has FEWER deaths → `no_sia − with_sia > 0`.
    let averted = find_artifact(&results, "contrasts", "averted")
        .expect("contrasts/averted.tsv must be auto-emitted by fit predict");
    let (q50, mean, n_forkable) = scalar_band(&averted);
    assert!(n_forkable > 0, "the band is over a positive forkable count, got {n_forkable}");
    assert!(
        q50 > 0.0,
        "the SIA averts deaths → median averted must be positive, got q50={q50} (mean={mean})"
    );
    assert!(mean > 0.0, "mean averted positive, got {mean}");

    // (2) CRN sanity: two no-op (×1.0) scenarios forked from the SAME X(T*) with the
    // SAME per-draw seed are byte-identical → the contrast is identically zero.
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
  bad = no_sia.quantities.infected - with_sia.quantities.total over [origin + 2 'weeks, origin + 10 'weeks]
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
