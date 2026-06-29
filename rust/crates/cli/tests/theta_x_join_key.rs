//! gh#322, (θ,X) step 1: a PGAS fit's `draws.tsv` carries leading `(chain, draw)`
//! key columns, and those keys are JOINABLE to the smoothed
//! `chain_N/trajectories.tsv` (which keys on the same `(chain, draw)`). This is
//! the foundation the keyed-joint `(θ, X)` join (step 2) is built on.
//!
//! Also pins the no-break property: the shared draws loader strips the key, so
//! `fit predict` (whose schema validator rejects a non-parameter draws column)
//! still runs — covered by the existing `fit_predict_e2e` suite, which now runs
//! against a keyed `draws.tsv`.
//!
//! docs/dev/proposals/2026-06-28-keyed-joint-param-trajectory-output.md §4

use std::collections::BTreeSet;
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

fn fit_toml() -> String {
    // n_trajectories defaults to 200, so for this small fit every retained draw
    // also gets a saved trajectory → the join is full (draws ⊇ traj, equal here).
    r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#
    .to_string()
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// Collect `(chain, draw)` from a TSV whose header has `chain` and `draw`
/// columns (the first two on draws.tsv; among the id columns on trajectories).
fn keys_from(path: &Path) -> BTreeSet<(i64, i64)> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    // trajectories.tsv leads with a `# camdl-trajectories …` comment line; skip
    // any comment lines to reach the column header.
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let ci = header.iter().position(|c| *c == "chain").expect("a `chain` column");
    let di = header.iter().position(|c| *c == "draw").expect("a `draw` column");
    let mut out = BTreeSet::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        // trajectories.tsv repeats (chain,draw) on every snapshot row — the set
        // dedups them to one key per draw.
        out.insert((f[ci].parse().unwrap(), f[di].parse().unwrap()));
    }
    out
}

#[test]
fn draws_tsv_key_joins_to_trajectories() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_thetax_key_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml()).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Locate the stage dir's draws.tsv + every chain_N/trajectories.tsv under it.
    let fits = tmp.join("results").join("fits");
    let mut draws_tsv: Option<PathBuf> = None;
    let mut traj_tsvs: Vec<PathBuf> = Vec::new();
    for f in walk(&fits) {
        match f.file_name().and_then(|n| n.to_str()) {
            Some("draws.tsv") => draws_tsv = Some(f.clone()),
            Some("trajectories.tsv") => traj_tsvs.push(f.clone()),
            _ => {}
        }
    }
    let draws_tsv = draws_tsv.expect("draws.tsv written");
    assert!(!traj_tsvs.is_empty(), "PGAS wrote at least one chain trajectories.tsv");

    // (1) draws.tsv leads with the key columns.
    let header = std::fs::read_to_string(&draws_tsv).unwrap().lines().next().unwrap().to_string();
    assert!(
        header.starts_with("chain\tdraw\t"),
        "draws.tsv must lead with `chain\\tdraw`; got header: {header}"
    );

    // (2) every saved trajectory's (chain, draw) key joins to a draws.tsv row —
    // the inner join (step 2) is non-empty and loses no saved path.
    let draws_keys = keys_from(&draws_tsv);
    let traj_keys: BTreeSet<(i64, i64)> =
        traj_tsvs.iter().flat_map(|p| keys_from(p)).collect();
    assert!(!draws_keys.is_empty() && !traj_keys.is_empty(), "both files carry keys");
    let unjoined: Vec<_> = traj_keys.difference(&draws_keys).collect();
    assert!(
        unjoined.is_empty(),
        "every saved trajectory key must join to a draw; {} unjoined: {unjoined:?}",
        unjoined.len()
    );

    // (3) the `draw` values are sweep NUMBERS, not 0..n positions: with burn_in=20
    // thin=1, the retained sweeps are 20..=59, so the per-chain draw set starts at
    // 20 (proves draw = the recorded sweep index, joinable to trajectories).
    let min_draw = draws_keys.iter().map(|(_, d)| *d).min().unwrap();
    assert_eq!(min_draw, 20, "draw is the sweep number (post-burn-in starts at 20), not a 0-based row index");

    let _ = std::fs::remove_dir_all(&tmp);
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
