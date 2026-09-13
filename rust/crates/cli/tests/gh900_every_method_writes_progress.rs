//! gh#900: every fitting method leaves a terminal `progress.json`, and a
//! stage that fails says so instead of going stale.
//!
//! `progress.json` is the one artifact a watcher can read while a stage runs,
//! and the gh#278 contract is that a consumer decides liveness from its
//! freshness. Two things broke that. `Heartbeat::mcmc` was constructed in
//! exactly one place, `pgas.rs`, so PMMH, `mh`, NUTS, IF2 and the two NLopt
//! MLEs wrote no file at all and a five-hour fit was indistinguishable from a
//! dead one. And PGAS wrote its terminal state from the end of its happy path,
//! so every other exit left the last periodic `running` record on disk — which
//! `RunLiveness` reserves for an *uncaught* death (SIGKILL, panic), not for a
//! refusal camdl printed a sentence for.
//!
//! The two claims here are the two halves of that: a completed run of each
//! method ends `done`, and a failed PGAS run ends `failed` carrying the reason
//! the command printed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn require_tools() {
    assert!(
        binary().exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        binary().display()
    );
    assert!(
        camdlc().exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`",
        camdlc().display()
    );
}

/// Compile `src` and return the IR path.
fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let model = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model, src).unwrap();
    let out = Command::new(camdlc()).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// The seed leaf: the directory holding `progress.json`.
fn seed_leaf(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("progress.json").is_file() {
            return d;
        }
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no progress.json anywhere under {}", root.display());
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| {
        panic!("cannot read {}: {e}", p.display())
    }))
    .unwrap_or_else(|e| panic!("cannot parse {}: {e}", p.display()))
}

/// One completed `camdl fit run`, with the tree it wrote into.
struct Run {
    ok: bool,
    stderr: String,
    out: PathBuf,
}

fn fit_run(dir: &Path, tag: &str, fit_toml: String) -> Run {
    let out = dir.join(format!("out-{tag}"));
    let path = dir.join(format!("fit-{tag}.toml"));
    std::fs::write(&path, fit_toml).unwrap();
    let r = Command::new(binary())
        .args(["fit", "run"])
        .arg(&path)
        .args(["--seed", "1", "--progress", "none"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl fit run");
    Run {
        ok: r.status.success(),
        stderr: String::from_utf8_lossy(&r.stderr).into_owned(),
        out,
    }
}

// ── Claim 1: every method ends `done` ────────────────────────────────────────

/// SIR with a daily prevalence count. Runs on both backends the methods below
/// need: chain-binomial for PGAS / PMMH / IF2, the ODE skeleton for NUTS and the
/// three NLopt MLEs.
const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 20 'days }
"#;

/// The `[method]` body for each method, at the smallest budget that still
/// exercises its loop. The assertion is "the stage leaves a terminal record",
/// not convergence, so the counts are deliberately tiny.
fn method_body(method: &str) -> &'static str {
    match method {
        "pgas" => r#"algorithm = "pgas"
backend   = "chain_binomial"
chains    = 1
particles = 20
sweeps    = 4
burn_in   = 1
thin      = 1
starts    = "single"
"#,
        "pmmh" => r#"algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 8
burn_in    = 2
thin       = 1
starts     = "single"
"#,
        "if2" => r#"algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 3
cooling    = 0.9
starts     = "single"
[method.loglik_eval]
n_particles  = 50
n_replicates = 2
[method.dt_check]
enabled = false
"#,
        "nuts" => r#"algorithm = "nuts"
backend   = "ode"
chains    = 1
warmup    = 10
samples   = 10
starts    = "single"
"#,
        "nl-sbplx" => r#"algorithm = "nl-sbplx"
backend   = "ode"
chains    = 1
max_evals = 40
[method.dt_check]
enabled = false
"#,
        // The gradient NLopt method counts the SAME step — one objective
        // evaluation — so the heartbeat has to bump on the `det_grad` branch
        // too. A method absent from this list is exactly the silent gap gh#900
        // was about.
        "nl-lbfgs" => r#"algorithm = "nl-lbfgs"
backend   = "ode"
chains    = 1
max_evals = 40
[method.dt_check]
enabled = false
"#,
        other => panic!("unknown method {other}"),
    }
}

#[test]
fn every_method_leaves_a_terminal_progress_record() {
    require_tools();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ir = compile(dir, "sir", SIR);

    // Synthetic counts from the ODE skeleton at a known θ: deterministic, so
    // every method below starts from data it can score.
    let truth = dir.join("truth.toml");
    std::fs::write(&truth, "beta = 0.8\ngamma = 0.3\nN0 = 1000\n").unwrap();
    let data = dir.join("cases.tsv");
    let sim = Command::new(binary())
        .arg("simulate")
        .arg(&ir)
        .args(["--params"])
        .arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"])
        .arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .unwrap();
    assert!(sim.status.success(), "data gen: {}", String::from_utf8_lossy(&sim.stderr));

    for method in ["pgas", "pmmh", "if2", "nuts", "nl-sbplx", "nl-lbfgs"] {
        let run = fit_run(
            dir,
            method,
            format!(
                r#"output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.05, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
[fixed]
gamma = 0.3
N0 = 1000
[method]
{body}"#,
                out = dir.join(format!("out-{method}")).display(),
                ir = ir.display(),
                data = data.display(),
                body = method_body(method),
            ),
        );
        assert!(run.ok, "{method}: the fit must complete\nstderr:\n{}", run.stderr);

        let leaf = seed_leaf(&run.out);
        let progress = read_json(&leaf.join("progress.json"));
        assert_eq!(
            progress["state"], "done",
            "{method}: a completed stage's last self-report must be `done`. Before \
             gh#900 only PGAS wrote this file at all, and a reader deciding liveness \
             from its freshness saw every other method as a run that had not started \
             yet.\n{progress:#}"
        );
        // The envelope every reader already has.
        assert!(progress["updated_at"].is_u64(), "{method}: {progress:#}");
        assert!(progress["pid"].is_u64(), "{method}: {progress:#}");
    }
}

// ── Claim 2: a failed PGAS stage says why ────────────────────────────────────

/// The same SIR, observed as `binomial(n = prevalence(I), p = 0.5)`. `n` is
/// bounded by the reachable `I`, so an observed count far above the population
/// scores `-inf` at every particle and every θ (`obs_loglik`'s `k > n` branch),
/// and under `starts = "single"` — a point rule, which permits no redraw — the
/// one chain is refused on its only attempt. The refusal is a property of the
/// data, not of the seed, so this fails the same way every run.
const IMPOSSIBLE: &str = r#"
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
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ binomial(n = projected, p = 0.5)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 2 'days }
"#;

#[test]
fn a_pgas_stage_that_fails_records_why_it_stopped() {
    require_tools();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ir = compile(dir, "impossible", IMPOSSIBLE);
    let data = dir.join("cases.tsv");
    std::fs::write(&data, "time\tcases\n1\t1000000\n2\t1000000\n").unwrap();

    let run = fit_run(
        dir,
        "refused",
        format!(
            r#"output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[method]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 1
particles = 20
sweeps    = 5
burn_in   = 1
thin      = 1
starts    = "single"
"#,
            out = dir.join("out-refused").display(),
            ir = ir.display(),
            data = data.display(),
        ),
    );
    assert!(
        !run.ok,
        "fixture premise: the only chain must be refused and the run must exit \
         non-zero.\nstderr:\n{}",
        run.stderr
    );

    let leaf = seed_leaf(&run.out);
    let progress = read_json(&leaf.join("progress.json"));
    let failed = progress["state"]["failed"].as_object().unwrap_or_else(|| {
        panic!(
            "a stage that stopped because camdl refused it must say `failed`. A \
             `running` record here is the SIGKILL signature — a reader applying the \
             gh#278 staleness rule would call this run presumed dead and never learn \
             that camdl had a reason and printed it.\n{progress:#}"
        )
    });

    // The claim that matters: a reader holding only `progress.json` learns what
    // a reader of stderr learns. The recorded reason is the sentence the
    // command printed, verbatim.
    let why = failed["reason"].as_str().expect("a failure carries its reason");
    assert!(
        run.stderr.contains(why),
        "the recorded reason must be the sentence the command printed.\n\
         progress.json says:\n{why}\n\nstderr was:\n{}",
        run.stderr
    );
    assert!(
        why.contains("refused at their starting point"),
        "and it must say what happened, not carry a bare tag: {why}"
    );

    // The per-chain block (gh#751) still rides beside it, so the reader needs
    // nothing else to know which chain never ran.
    let chains = &progress["chains"];
    assert_eq!(chains["total"], 1, "{progress:#}");
    assert_eq!(chains["refused"], 1, "{progress:#}");
    assert_eq!(chains["running"], 0, "no chain may be left claiming to run:\n{progress:#}");
    assert_eq!(chains["chains"][0]["reason"], "non_finite_start", "{progress:#}");
}
