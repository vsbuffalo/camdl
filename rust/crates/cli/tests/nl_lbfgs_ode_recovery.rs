//! End-to-end: `camdl fit run` with `algorithm = "nl-lbfgs"` on the `ode`
//! backend (gh#275) — the gradient deterministic MLE.
//!
//! Two claims, each about something only the whole pipeline can show.
//!
//! 1. **Recovery.** From synthetic incidence at a known `(beta, gamma)`,
//!    `nl-lbfgs` lands on the same optimum as the derivative-free `nl-sbplx`
//!    from the same starts, within the same tolerance, and leaves the same
//!    artifacts in the leaf. Run side by side rather than against a stored
//!    number, so the comparison is between the two search strategies on one
//!    likelihood rather than against a constant that could drift.
//!
//! 2. **Refusal.** A model whose ODE gradient cannot be taken — here a
//!    scheduled intervention, whose discrete state jump the smooth
//!    sensitivity flow cannot cross — is refused at config validation: exit
//!    non-zero, the gate's reason, the `nl-sbplx` line, and no leaf written.
//!
//! Both shell out to the release binary, skipped silently when it or
//! `camdlc.exe` is not built (mirrors `nuts_ode_recovery` / `ode_dt_check`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = PathBuf::from(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir {
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir()
        .join(format!("camdl_nllbfgs_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let model = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model, src).unwrap();
    let out = Command::new(camdlc().unwrap()).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// Every file named `name` anywhere under `root`.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                    found.push(p);
                }
            }
        }
    }
    found
}

const TRUE_BETA: f64 = 0.9;
const TRUE_GAMMA: f64 = 0.3;
/// The recovery tolerance, applied to BOTH algorithms: whatever `nl-sbplx`
/// must meet on this fixture, `nl-lbfgs` must meet too.
const TOL: f64 = 0.1;

/// SIR with a 3-daily incidence stream. Bounds are narrow enough that every
/// `starts = "uniform"` draw is a θ the deterministic likelihood can score —
/// wide bounds on an epidemic model routinely draw a start where the outbreak
/// never takes off and the Poisson rate is 0 against a positive count, which
/// is a property of the fixture rather than of either optimizer.
const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.5, 1.5]
  gamma : rate  in [0.1, 0.6]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 3 'days)
    projected     = incidence(infection)
    emit_schedule = every 3 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;

/// The same SIR with a scheduled intervention. `interventions { }` is
/// toggleable and defaults off, so the fit below enables it — the effect has
/// to be IN the fitted trajectory for the gradient gate to have anything to
/// refuse.
const SIR_WITH_SIA: &str = r#"
time_unit = 'days
compartments { S, I, R, V }
parameters {
  beta  : rate        in [0.5, 1.5]
  gamma : rate        in [0.1, 0.6]
  cover : probability in [0.0, 1.0]
  N0    : count       in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
interventions {
  sia : transfer(fraction = cover, from = S, to = V) at [20]
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 3 'days)
    projected     = incidence(infection)
    emit_schedule = every 3 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;

/// `[method]` for one NLopt algorithm, two chains from independent uniform
/// draws: a spread rule, so the chain-agreement leg of the gate is
/// informative and the two chains are a real multi-start.
fn method_body(algorithm: &str) -> String {
    format!(
        r#"[method]
algorithm = "{algorithm}"
backend = "ode"
chains = 2
starts = "uniform"
tolerance = 1e-8
max_evals = 5000
"#
    )
}

/// What one chain of a finished NLopt stage recorded.
struct Chain {
    loglik: f64,
    params: Vec<(String, f64)>,
}

fn read_chains(leaf: &Path) -> Vec<Chain> {
    let text = std::fs::read_to_string(leaf.join("chain_results.tsv"))
        .unwrap_or_else(|e| panic!("reading chain_results.tsv: {e}"));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    lines
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            Chain {
                loglik: cols[1].parse().unwrap(),
                params: header[4..]
                    .iter()
                    .zip(&cols[4..])
                    .map(|(n, v)| (n.to_string(), v.parse().unwrap()))
                    .collect(),
            }
        })
        .collect()
}

fn best(chains: &[Chain]) -> &Chain {
    chains
        .iter()
        .max_by(|a, b| a.loglik.total_cmp(&b.loglik))
        .expect("at least one chain")
}

fn value_of(c: &Chain, name: &str) -> f64 {
    c.params
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| *v)
        .unwrap_or_else(|| panic!("no `{name}` column in chain_results.tsv"))
}

#[test]
fn nl_lbfgs_recovers_the_truth_and_matches_the_derivative_free_optimum() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("recovery");
    let ir = compile(tmp.path(), "sir", SIR);

    // Synthetic incidence from the ODE skeleton at the true θ.
    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, format!("beta = {TRUE_BETA}\ngamma = {TRUE_GAMMA}\nN0 = 10000\n"))
        .unwrap();
    let data = tmp.path().join("cases.tsv");
    let sim = Command::new(&bin)
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"]).arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));

    let mut leaves: Vec<(String, PathBuf)> = Vec::new();
    for algorithm in ["nl-sbplx", "nl-lbfgs"] {
        let out_dir = tmp.path().join(format!("out-{algorithm}"));
        let fit_toml = tmp.path().join(format!("fit-{algorithm}.toml"));
        std::fs::write(&fit_toml, format!(
            r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.5, 1.5] }}
gamma = {{ bounds = [0.1, 0.6] }}

[fixed]
N0 = 10000

{body}"#,
            out = out_dir.display(),
            ir = ir.display(),
            data = data.display(),
            body = method_body(algorithm),
        )).unwrap();

        let r = Command::new(&bin)
            .args(["fit", "run"]).arg(&fit_toml)
            .args(["--seed", "1", "--progress", "none"])
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().unwrap();
        assert!(
            r.status.success(),
            "{algorithm}: `fit run` must exit 0\nstderr:\n{}",
            String::from_utf8_lossy(&r.stderr)
        );

        let states = find_all(&out_dir, "fit_state.toml");
        assert_eq!(
            states.len(), 1,
            "{algorithm}: expected exactly one stage leaf under {}",
            out_dir.display()
        );
        leaves.push((algorithm.to_string(), states[0].parent().unwrap().to_path_buf()));
    }

    let mut logliks: Vec<(String, f64)> = Vec::new();
    for (algorithm, leaf) in &leaves {
        let chains = read_chains(leaf);
        assert_eq!(chains.len(), 2, "{algorithm}: two chains were configured");
        let w = best(&chains);
        let (beta, gamma) = (value_of(w, "beta"), value_of(w, "gamma"));
        eprintln!(
            "{algorithm}: beta = {beta:.4} (true {TRUE_BETA}), \
             gamma = {gamma:.4} (true {TRUE_GAMMA}), loglik = {:.4}",
            w.loglik
        );
        assert!(
            (beta - TRUE_BETA).abs() < TOL,
            "{algorithm} did not recover beta: {beta} vs {TRUE_BETA} (tolerance {TOL})"
        );
        assert!(
            (gamma - TRUE_GAMMA).abs() < TOL,
            "{algorithm} did not recover gamma: {gamma} vs {TRUE_GAMMA} (tolerance {TOL})"
        );
        logliks.push((algorithm.clone(), w.loglik));
    }

    // Same likelihood, same data, same starts — so the two searches must end
    // at the same optimum. A gradient that is not the likelihood's gradient
    // would still converge to SOMETHING, and this is what tells the
    // difference: it would be a different optimum from the derivative-free
    // search on the same surface.
    let spread = (logliks[0].1 - logliks[1].1).abs();
    assert!(
        spread < 1e-3,
        "nl-sbplx and nl-lbfgs must find the same optimum on one likelihood: \
         {} = {}, {} = {} (Δ = {spread})",
        logliks[0].0, logliks[0].1, logliks[1].0, logliks[1].1
    );

    // The leaf a downstream reader walks.
    let lbfgs_leaf = &leaves.iter().find(|(a, _)| a == "nl-lbfgs").unwrap().1;
    for artifact in ["fit_state.toml", "mle_params.toml", "chain_results.tsv", "progress.json"] {
        assert!(
            lbfgs_leaf.join(artifact).is_file(),
            "the nl-lbfgs leaf must carry {artifact}; it holds {:?}",
            std::fs::read_dir(lbfgs_leaf).unwrap()
                .filter_map(|e| e.ok().map(|e| e.file_name())).collect::<Vec<_>>()
        );
    }
    let mle = std::fs::read_to_string(lbfgs_leaf.join("mle_params.toml")).unwrap();
    assert!(
        mle.contains("[provenance]") && mle.contains("method = \"nl-lbfgs\""),
        "mle_params.toml must record which method produced it:\n{mle}"
    );
    // gh#900: the heartbeat counts one objective evaluation, and a gradient
    // evaluation is one of those — a method that reports no terminal state is
    // indistinguishable from a dead one to a reader watching the artifact.
    let progress: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lbfgs_leaf.join("progress.json")).unwrap(),
    ).unwrap();
    assert_eq!(
        progress["state"], "done",
        "a completed nl-lbfgs stage's last self-report must be `done`:\n{progress:#}"
    );
}

#[test]
fn nl_lbfgs_on_a_scheduled_intervention_is_refused_before_any_leaf() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("refusal");
    let ir = compile(tmp.path(), "sia", SIR_WITH_SIA);
    // Any scorable data will do: the refusal happens before a likelihood is
    // ever evaluated.
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, "time\tcases\n3\t76\n6\t424\n9\t1885\n12\t3506\n").unwrap();

    let out_dir = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        r#"output_dir = "{out}"
enable = ["sia"]

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.5, 1.5] }}

[fixed]
gamma = 0.3
cover = 0.5
N0 = 10000

{body}"#,
        out = out_dir.display(),
        ir = ir.display(),
        data = data.display(),
        body = method_body("nl-lbfgs"),
    )).unwrap();

    let r = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit_toml)
        .args(["--seed", "1", "--progress", "none"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&r.stderr).into_owned();
    assert!(
        !r.status.success(),
        "a model the ODE gradient cannot be taken of must not fit silently:\n{stderr}"
    );
    assert!(
        stderr.contains("intervention") || stderr.contains("scheduled"),
        "the refusal must name the reason — the scheduled effect:\n{stderr}"
    );
    assert!(
        stderr.contains("nl-sbplx"),
        "the refusal must name the derivative-free alternative:\n{stderr}"
    );
    // Refused before the optimizer ran, so no stage result exists anywhere.
    for artifact in ["fit_state.toml", "mle_params.toml", "chain_results.tsv"] {
        let found = find_all(&out_dir, artifact);
        assert!(
            found.is_empty(),
            "a refused fit must write no {artifact}; found {found:?}"
        );
    }
}
