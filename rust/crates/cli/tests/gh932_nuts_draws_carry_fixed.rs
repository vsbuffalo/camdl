//! gh#932: a NUTS leaf's `draws.tsv` carries every model parameter, so
//! `fit predict` can replay it.
//!
//! `draws.tsv` is the canonical posterior (`posterior_draws.rs`): one row per
//! post-warm-up draw, *every* model parameter a column, estimated first and the
//! `[fixed]` values after. PGAS, PMMH and MH write that shape. NUTS wrote only
//! the estimated columns, so `fit predict` — which refuses a draw that does not
//! cover every parameter rather than default one silently — refused every NUTS
//! fit with a `[fixed]` table.
//!
//! The claims pinned here, on a real ODE NUTS fit with two fixed parameters:
//!
//! 1. `draws.tsv` has a column for every model parameter.
//! 2. Each fixed column holds exactly the `[fixed]` value on every row — the
//!    value the fit conditioned on, not a model default.
//! 3. `fit predict` runs on the fit.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn require_binary() {
    assert!(
        binary().exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        binary().display()
    );
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir() -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base =
        std::env::temp_dir().join(format!("camdl_gh932_{}_{}", std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The issue's reproduction model: `gamma` and `N` carry no default and no
/// prior, so the only source of their values is the fit's `[fixed]` table.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  N     : count
  k     : positive
}
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = N - 10  I = 10 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ neg_binomial(mean = projected, r = k)
  }
}
simulate { from = 0 'days  to = 40 'days }
"#;

const FIT_TOML: &str = r#"
[model]
camdl = "sir.camdl"

[data.observations]
cases = "obs/cases.tsv"

[estimate]
beta = { bounds = [0.05, 2.0], prior = { log_normal = { mu = -1.0, sigma = 0.5 } } }
k    = { bounds = [1.0, 200.0], prior = { log_normal = { mu = 3.0, sigma = 1.0 } } }

[fixed]
gamma = 0.2
N     = 10000

[method]
algorithm = "nuts"
backend   = "ode"
chains    = 2
warmup    = 20
samples   = 20
starts    = "single"
"#;

/// The NUTS seed leaf: the directory holding `nuts_summary.json`.
fn seed_leaf(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("nuts_summary.json").is_file() {
            return d;
        }
        if let Ok(entries) = std::fs::read_dir(&d) {
            stack.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
        }
    }
    panic!("no nuts stage leaf under {}", root.display());
}

#[test]
fn a_nuts_fit_with_fixed_parameters_writes_complete_draws_and_predicts() {
    require_binary();
    let tmp = tempdir();
    let dir = &tmp.0;
    std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();

    let sim = run(
        dir,
        &[
            "simulate", "sir.camdl", "--backend", "ode",
            "--param", "beta=0.4", "--param", "gamma=0.2",
            "--param", "N=10000", "--param", "k=20",
            "--obs-only-dir", "obs",
        ],
    );
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));

    let fit = run(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(fit.status.success(), "nuts `fit run` failed: {}", String::from_utf8_lossy(&fit.stderr));

    // Claims 1 and 2: every parameter is a column; fixed columns hold [fixed].
    let leaf = seed_leaf(&dir.join("results"));
    let text = std::fs::read_to_string(leaf.join("draws.tsv")).unwrap();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("draws.tsv header").split('\t').collect();
    for p in ["beta", "gamma", "N", "k"] {
        assert!(header.contains(&p), "draws.tsv lacks parameter `{p}`: {header:?}");
    }
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split('\t').map(|f| f.parse::<f64>().unwrap()).collect())
        .collect();
    assert!(!rows.is_empty(), "draws.tsv has no draws");
    for (name, want) in [("gamma", 0.2), ("N", 10000.0)] {
        let c = header.iter().position(|h| *h == name).unwrap();
        for r in &rows {
            assert_eq!(r[c], want, "fixed `{name}` must be its [fixed] value on every draw");
        }
    }

    // Claim 3: predict replays the fit.
    let pred = run(dir, &["fit", "predict", "--fit", "fit.toml"]);
    assert!(
        pred.status.success(),
        "`fit predict` must accept a nuts fit with [fixed] parameters:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&pred.stdout),
        String::from_utf8_lossy(&pred.stderr)
    );
}
