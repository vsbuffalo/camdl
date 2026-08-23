//! gh#722 — a `value_at` anchored inside the observed record is read off the
//! CONDITIONED smoothing path, not off the free-forward replay.
//!
//! The reported failure: `outbreak_size = value_at(N0 - S, last_obs)` came back
//! with a median of 4,058 cumulative infections against 5,157 observed confirmed
//! cases. Not a wide interval — an impossible one, since infections cannot be
//! fewer than the cases ascertained from them. The quantity was folded over a
//! fresh unconditioned replay from `init {}`, which discards all 85 observations
//! covering the instant it reads.
//!
//! The fixture reproduces the mechanism with `I0 = 1`: at `R0 = beta/gamma =
//! 1.75` the branching-process extinction probability is ≈ 1/R0 ≈ 0.57, so most
//! free-forward replays fizzle at one infection while every conditioned path
//! carries the epidemic the data record. The pre-fix and post-fix medians differ
//! by an order of magnitude.
//!
//! What is pinned here:
//!
//!   * `in_window_value_at_reads_the_conditioned_path` — the headline. The
//!     band equals, quantile for quantile, an INDEPENDENT reduction of the saved
//!     `trajectories.tsv` this test performs itself; the forkable subset is
//!     reported (`n_value` / `n_censored`, `n_conditioned_draws`) rather than
//!     substituted; and the manifest tags each quantity with the object its
//!     numbers came from.
//!   * The negative controls ride in the same run, so they cannot drift from
//!     it: a `value_at` PAST `last_obs`, a `time_of_max` and a `final` must all
//!     still take the replay. The saved path stops at `last_obs`, so a quantity
//!     wrongly routed to it would come back fully censored — `n_value = 0` is
//!     the sharp failure signal.

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

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// A closed SIR seeded with ONE infective, observed weekly to t = 70 with
/// perfect ascertainment (`poisson(rate = projected)`, no reporting fraction),
/// and a model horizon of t = 140.
///
/// `cum_infections` is the quantity under test; the other three are the
/// negative controls, chosen to cover each way a quantity can fail to be an
/// in-window `value_at`: anchored PAST the record, a time reduction, and a
/// whole-path reduction.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 1.5] ~ log_normal(mu = -0.7, sigma = 0.5)
  gamma : rate  in [0.05, 0.5]
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ poisson(rate = projected)
  }
}
quantities {
  cum_infections = value_at(N0 - S, last_obs)
  cum_at_horizon = value_at(N0 - S, last_obs + 4 'weeks)
  peak_time      = time_of_max(I)
  final_size     = final(N0 - S)
}
simulate { from = 0 'days  to = 140 'days }
"#;

/// Weekly counts simulated from this very model at `beta = 0.35, gamma = 0.2,
/// N0 = 10000, I0 = 1` (`camdl simulate --seed 11`), so the data are a draw from
/// the model rather than a hand-drawn curve the likelihood cannot explain.
/// 7,639 cases in total over the ten weeks.
const DATA: &str = "time\tweekly_cases\n\
     7\t22\n14\t58\n21\t192\n28\t433\n35\t1123\n42\t1911\n49\t1818\n56\t1129\n63\t660\n70\t293\n";

/// `n_trajectories = 20` per chain against 30 post-burn-in draws per chain, so
/// the saved subset is a PROPER subset (40 of 80): the reported forkable count
/// is exercised rather than trivially equal to the cloud.
const FIT_TOML: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.5], start = 0.35 }
[fixed]
gamma = 0.2
N0    = 10000
I0    = 1
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 300
sweeps = 60
burn_in = 20
thin = 1
n_trajectories = 20
"#;

/// The no-saved-paths fixture, fitted with PMMH — which stores no smoothed
/// latent path for any draw. Its own small model: a bootstrap filter cannot
/// track the headline fixture (perfect ascertainment over a sharp epidemic
/// seeded at one infective makes the observation near-deterministic given the
/// path, and the filter degenerates), so this one observes small counts through
/// an overdispersed `neg_binomial`.
///
/// `cum_at_35` is anchored one week PAST the last observation (t = 35 < t_end
/// = 40) — the in-window/out-of-window pair on one fit.
const MODEL_NO_PATHS: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
  k     : real in [0.5, 100.0]
}
let cum_inc = I + R
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10 }
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
}
quantities {
  cum_infections = value_at(cum_inc, last_obs)
  cum_at_35      = value_at(cum_inc, last_obs + 1 'weeks)
}
simulate { from = 0  to = 40 }
"#;

const DATA_NO_PATHS: &str = "time\tcases\n7\t6\n14\t14\n21\t9\n28\t4\n";

const FIT_TOML_NO_PATHS: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
cases = "cases.tsv"
[estimate]
beta  = { bounds = [0.05, 2.0], start = 0.4, prior = { uniform = {} } }
gamma = { bounds = [0.01, 1.0], start = 0.2, prior = { uniform = {} } }
[fixed]
rho = 0.6
k = 10.0
[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 80
burn_in = 20
thin = 5
"#;

const LAST_OBS: f64 = 70.0;
const N0: f64 = 10_000.0;

/// Linear-interpolated quantile, the type-7 rule `crate::quantile` uses. The
/// oracle has to reduce the saved paths the SAME way the bander reduces the
/// draws, or an exact comparison is meaningless.
fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() == 1 {
        return v[0];
    }
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

/// The fit segment directory under `results/fits/`.
fn segment(root: &Path) -> PathBuf {
    std::fs::read_dir(root.join("fits"))
        .expect("results/fits")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("one fit segment")
}

/// `N0 - S` at `LAST_OBS` for every saved smoothing path under the segment —
/// computed here, from the file on disk, with no help from the code under test.
fn cumulative_infections_on_saved_paths(seg: &Path) -> Vec<f64> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let traj = p.join("trajectories.tsv");
                if traj.is_file() {
                    out.push(traj);
                }
                walk(&p, out);
            }
        }
    }
    let mut files = Vec::new();
    walk(seg, &mut files);
    files.sort();
    assert!(!files.is_empty(), "the fit saved no trajectories.tsv under {}", seg.display());

    let mut vals = Vec::new();
    for f in &files {
        let txt = std::fs::read_to_string(f).unwrap();
        let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
        let ix = |n: &str| header.iter().position(|h| *h == n).unwrap();
        let (c_t, c_s) = (ix("time"), ix("S"));
        for l in lines {
            let c: Vec<&str> = l.split('\t').collect();
            if (c[c_t].parse::<f64>().unwrap() - LAST_OBS).abs() < 1e-9 {
                vals.push(N0 - c[c_s].parse::<f64>().unwrap());
            }
        }
    }
    vals
}

/// One quantity TSV's single banded row, as `column -> cell`.
fn banded_row(seg: &Path, name: &str) -> std::collections::HashMap<String, String> {
    let path = seg.join("quantities").join(format!("{name}.tsv"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let row: Vec<&str> = lines.next().expect("one banded row").split('\t').collect();
    header.iter().map(|h| h.to_string()).zip(row.iter().map(|c| c.to_string())).collect()
}

fn manifest_entry(seg: &Path, name: &str) -> serde_json::Value {
    let txt = std::fs::read_to_string(seg.join("quantities.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    v["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == name)
        .unwrap_or_else(|| panic!("no manifest entry for {name}"))
        .clone()
}

#[test]
fn in_window_value_at_reads_the_conditioned_path() {
    let bin = skip_if_missing_binary();
    let dir = std::env::temp_dir().join(format!("camdl_gh722_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();

    let fit = run(&bin, &dir, &["fit", "run", "fit.toml"]);
    assert!(
        fit.status.success(),
        "fit run failed:\n{}",
        String::from_utf8_lossy(&fit.stderr)
    );
    let pred = run(&bin, &dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        pred.status.success(),
        "fit predict failed:\n{}",
        String::from_utf8_lossy(&pred.stderr)
    );
    let stderr = String::from_utf8_lossy(&pred.stderr).to_string();
    let seg = segment(&dir.join("results"));

    // ── The independent oracle: reduce the saved paths ourselves ────────────
    let saved = cumulative_infections_on_saved_paths(&seg);
    assert_eq!(saved.len(), 40, "20 saved paths per chain x 2 chains");

    let row = banded_row(&seg, "cum_infections");
    // The forkable subset is REPORTED: the band is over the 40 draws that have
    // a conditioned path, and the other 40 are censored — not silently given
    // the replay's answer, which is the defect itself one draw at a time.
    assert_eq!(row["n_draws"], "80", "the cloud the cell replayed");
    assert_eq!(row["n_value"], "40", "banded over the forkable subset");
    assert_eq!(row["n_censored"], "40", "the rest are censored, not substituted");

    for (label, q) in [("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95)] {
        let got: f64 = row[label].parse().unwrap();
        let want = quantile(&saved, q);
        assert!(
            (got - want).abs() < 1e-6,
            "cum_infections {label} = {got}, but reducing the saved smoothing paths \
             gives {want} — the quantity was not folded over the conditioned path"
        );
    }

    // ── Non-vacuity, without needing the pre-fix binary ─────────────────────
    //
    // Cumulative infections are non-decreasing in t, so for any single replay
    // the value at last_obs is <= the value at last_obs + 4 weeks. The free-
    // forward `cum_at_horizon` band is therefore an UPPER bound, quantile by
    // quantile, on what `cum_infections` would read on the replay. Most replays
    // fizzle at one infection (I0 = 1, R0 = 1.75), so that upper bound is tiny —
    // and `cum_infections` is thousands. The two cannot be the same object.
    let horizon_row = banded_row(&seg, "cum_at_horizon");
    let replay_upper: f64 = horizon_row["q05"].parse().unwrap();
    let smoothed_q05: f64 = row["q05"].parse().unwrap();
    assert!(
        smoothed_q05 > 100.0 * replay_upper.max(1.0),
        "the conditioned q05 ({smoothed_q05}) must be far above the replay's own \
         upper bound at a LATER time ({replay_upper}); if they are comparable this \
         test is not distinguishing the two objects"
    );

    // ── The negative controls, in the same run ──────────────────────────────
    //
    // The saved path stops at last_obs, so a quantity wrongly routed onto it
    // would censor every draw. `n_value = 80` is the proof they stayed on the
    // replay, which spans the whole model horizon.
    for name in ["cum_at_horizon", "peak_time"] {
        let r = banded_row(&seg, name);
        assert_eq!(r["n_value"], "80", "{name} must stay on the free-forward replay");
        assert_eq!(r["n_censored"], "0", "{name} must not be censored");
    }
    // `final(N0 - S)` is a plain scalar (no censoring trio) — its whole cloud
    // is banded, so `n_draws` is the check.
    assert_eq!(banded_row(&seg, "final_size")["n_draws"], "80");

    // ── The artifact says which object each number came from ────────────────
    let cum = manifest_entry(&seg, "cum_infections");
    assert_eq!(cum["evaluated_on"], "smoothed");
    assert_eq!(
        cum["n_conditioned_draws"], 40,
        "the forkable count travels with the band it is the denominator of"
    );
    for name in ["cum_at_horizon", "peak_time", "final_size"] {
        let e = manifest_entry(&seg, name);
        assert_eq!(e["evaluated_on"], "replay", "{name} is a free-forward object");
        assert!(
            e.get("n_conditioned_draws").is_none(),
            "{name} has no conditioned denominator to report"
        );
    }

    // ── And says so on stderr, where a human reading a run log sees it ──────
    assert!(
        stderr.contains("cum_infections") && stderr.contains("smoothing path"),
        "the routing must be announced, not silent; stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("40/80"),
        "the forkable subset must be reported with its denominator; stderr was:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A fit that stored NO smoothed latent path — every PMMH/PF fit today — has
/// nothing conditioned to read an in-window `value_at` on. It reports the
/// quantity as fully censored and names it on stderr; it does NOT publish the
/// free-forward number, which is the reported defect verbatim.
///
/// The cost is real and deliberate: such a fit loses these quantities rather
/// than reporting an unconditioned value for them. The rest of the artifact —
/// the predictive bands, and every quantity that legitimately belongs to the
/// replay — is unaffected, which this test also pins.
#[test]
fn a_fit_with_no_saved_paths_censors_instead_of_falling_back() {
    let bin = skip_if_missing_binary();
    let dir = std::env::temp_dir().join(format!("camdl_gh722_nopaths_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL_NO_PATHS).unwrap();
    std::fs::write(dir.join("cases.tsv"), DATA_NO_PATHS).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML_NO_PATHS).unwrap();

    let fit = run(&bin, &dir, &["fit", "run", "fit.toml"]);
    assert!(fit.status.success(), "fit run failed:\n{}", String::from_utf8_lossy(&fit.stderr));
    let pred =
        run(&bin, &dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"]);
    assert!(
        pred.status.success(),
        "fit predict must still produce the rest of the artifact:\n{}",
        String::from_utf8_lossy(&pred.stderr)
    );
    let stderr = String::from_utf8_lossy(&pred.stderr).to_string();
    let seg = segment(&dir.join("results"));

    let row = banded_row(&seg, "cum_infections");
    assert_eq!(row["n_value"], "0", "no draw has a conditioned path to read");
    assert_eq!(row["q50"], "", "an empty band, not a free-forward number");
    let censored: u64 = row["n_censored"].parse().unwrap();
    assert!(censored > 0, "every draw is censored, and counted");

    let e = manifest_entry(&seg, "cum_infections");
    assert_eq!(
        e["evaluated_on"], "smoothed",
        "the quantity's object is still the smoothing path"
    );
    assert_eq!(e["n_conditioned_draws"], 0, "and zero draws supplied one");

    assert!(
        stderr.contains("cum_infections") && stderr.contains("saved no latent path"),
        "the empty band must be explained, by name; stderr was:\n{stderr}"
    );

    // The replay-side quantities and the predictive band are untouched.
    assert!(
        banded_row(&seg, "cum_at_35")["n_value"].parse::<u64>().unwrap() > 0,
        "an anchor PAST the record still bands off the replay, paths or no paths"
    );
    assert_eq!(manifest_entry(&seg, "cum_at_35")["evaluated_on"], "replay");
    assert!(seg.join("predictive").join("cases.tsv").is_file());

    let _ = std::fs::remove_dir_all(&dir);
}
