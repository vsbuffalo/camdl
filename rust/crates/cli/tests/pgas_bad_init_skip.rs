//! gh#607 — PGAS skip-and-continue on a chain start with zero posterior
//! density.
//!
//! `run_pgas` used to warn on a non-finite initial complete-data
//! log-likelihood and sample anyway, seeding every rung with `-inf`. On a
//! 40,000-sweep production fit one such chain held acceptance 0.000 and
//! `n_divergent` 1.000 for the whole run and contributed ONE distinct
//! parameter vector to the pooled posterior.
//!
//! The refusal is deferred by one Gibbs sweep, because a `-inf` at `(θ₀, X₀)`
//! is usually the observation term and the `X|θ,y` move can replace an unlucky
//! `X₀` at the same `θ₀`. This fixture is built so that it CANNOT: the
//! impossibility is a property of `θ₀` alone, so no trajectory rescues it and
//! the chain is refused after its probation sweep.
//!
//! ## The impossible-start mechanism
//!
//! The fixture seeds infection ONLY through an importation term `iota * S`
//! (the `I` compartment starts empty), and observes `cases ~ poisson(rate =
//! prevalence(I))`.
//!
//! - `iota = 0` ⇒ the infection rate is 0 at every substep FOR EVERY PARTICLE,
//!   so `I` stays 0 on every trajectory the conditional SMC can draw,
//!   `prevalence(I)` is 0 at every observation time, and `poisson_logpmf(k>0,
//!   λ=0)` is exactly `NEG_INFINITY` (`obs_loglik.rs`). No RNG dependence: the
//!   chain-binomial draw is `Binomial(S, p)` with `p` clamped to
//!   `BINOM_PROB_EPS`, so the trajectory is all-zero with probability
//!   1 − O(1e-9), and the *transition* term stays finite (measured: 0.0000) —
//!   the refusal is on the observation term, exactly the production case.
//! - `iota = 0.2` ⇒ ~180 importations on day 1 against `S = 1000`, so
//!   `prevalence(I) > 0` at every observation time and the Poisson term is
//!   finite whatever the counts.
//!
//! Per-chain starts come from a two-row draws file named by
//! `starts = { from_posterior = … }`: each chain draws one row, so which chain
//! gets the pathological row is a property of the seeded draw. The test reads
//! `chain_starts.tsv` — the artifact that records exactly this — rather than
//! assuming an assignment, and asserts the premise that the two chains drew
//! different rows. The same lever `pmmh_bad_init_skip.rs` uses.
//!
//! ## Acceptance
//!
//! 1. `one_bad_chain_is_skipped_and_survivors_finish` — exit 0; exactly one
//!    `bad_init` diagnostic, carrying the refused chain's index and the
//!    `iota = 0` start it actually ran from; `fit_state.toml` records
//!    `n_good_chains = 1` beside `n_chains = 2`; and `draws.tsv` holds draws
//!    for the surviving chain ONLY — the skipped chain enters no pooled number.
//! 2. `all_chains_refused_is_an_error` — both rows at `iota = 0` ⇒ non-zero
//!    exit and an `initial_loglik_infinite` diagnostic, rather than a
//!    degenerate posterior written at exit 0.
//! 3. `healthy_fit_keeps_every_chain` — the negative control. Both rows
//!    healthy ⇒ NO `bad_init`, both chains present in `draws.tsv`, and
//!    `fit_state.toml` carries no `n_good_chains` key, so a healthy fit's
//!    output is unchanged by the guard.
//! 4. `a_start_the_trajectory_move_can_rescue_is_not_refused` — the other half
//!    of the predicate: a start that is `-inf` only because its reference draw
//!    was unlucky must survive, because the `X|θ,y` move fixes it.
//!
//! Skipped when the release binary or camdlc isn't present, mirroring
//! `pmmh_bad_init_skip.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../target/release/camdl");
    if p.exists() { Some(p) } else { None }
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_pgas_bad_init_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// SIR whose only route into `I` is the importation term `iota * S`, so
/// `iota = 0` pins `prevalence(I)` at 0 for the whole window.
fn write_fixture(dir: &Path, camdlc: &Path) -> (PathBuf, PathBuf) {
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.001, 1.0]
  iota  : rate  in [0.0, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0 + iota * S
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 1000  I = 0 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    // Positive counts at every observation time: what makes `prevalence(I) = 0`
    // a probability-zero prediction rather than merely a poor one.
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t150\n2\t300\n3\t450\n4\t550\n5\t600\n6\t620\n").unwrap();

    (ir_path, data_path)
}

/// A two-row draws file: each chain draws one row as its start.
fn write_draws(dir: &Path, iotas: (f64, f64)) -> PathBuf {
    let draws = format!(
        "# gh#607 PGAS chain-start refusal test fixture\n\
         beta\tgamma\tiota\n\
         0.30\t0.10\t{}\n\
         0.30\t0.10\t{}\n",
        iotas.0, iotas.1);
    let p = dir.join("starts.tsv");
    std::fs::write(&p, draws).unwrap();
    p
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, draws: &Path) -> (PathBuf, PathBuf) {
    let out_root = dir.join("results");
    // `iota`'s prior is uniform over its whole bound range, so `iota = 0` is
    // INSIDE the prior's support: the refusal below must come from the
    // observation term, not from a start the prior already excludes.
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
iota  = {{ bounds = [0.0, 1.0],  prior = {{ uniform = {{ lower = 0.0, upper = 1.0 }} }}, start = 0.2 }}
[fixed]
N0 = 1000
[method]
algorithm      = "pgas"
backend        = "chain_binomial"
chains         = 2
particles      = 20
sweeps         = 10
burn_in        = 2
thin           = 1
starts         = {{ from_posterior = "{draws}" }}
"#,
        out    = out_root.display(),
        ir     = ir.display(),
        data   = data.display(),
        draws  = draws.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    (p, out_root)
}

/// The CAS stage leaf for `stage_substr` under `fits_root`.
fn cas_stage_leaf(fits_root: &Path, stage_substr: &str) -> Option<PathBuf> {
    let mut stack = vec![fits_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("run.json").is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(d.join("run.json")).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"].as_array().into_iter().flatten()
                        .find(|l| l["name"].as_str() == Some("method"))
                        .and_then(|l| l["label"].as_str()).unwrap_or("");
                    if stage.contains(stage_substr) { return Some(d); }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    None
}

/// Every `bad_init` diagnostic found in any `diagnostics.json` under `root`.
/// The all-refused run errors BEFORE CAS finalize, so its diagnostics land in
/// the streaming-claim leaf rather than at a predictable committed path —
/// walking the tree finds them wherever they are.
fn bad_init_entries(root: &Path) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for path in diagnostics_files(root) {
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let Some(arr) = v.as_array() else { continue };
        for d in arr {
            if d.get("kind").and_then(|k| k.get("type")).and_then(|t| t.as_str())
                == Some("bad_init")
            {
                out.push(d.get("kind").unwrap().clone());
            }
        }
    }
    out
}

fn has_diagnostic(root: &Path, tag: &str) -> bool {
    for path in diagnostics_files(root) {
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let Some(arr) = v.as_array() else { continue };
        if arr.iter().any(|d| d.get("kind").and_then(|k| k.get("type"))
            .and_then(|t| t.as_str()) == Some(tag))
        {
            return true;
        }
    }
    false
}

fn diagnostics_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().is_some_and(|n| n == "diagnostics.json") { out.push(p); }
            }
        }
    }
    out
}

/// Each chain's `iota` start, read from the leaf's `chain_starts.tsv` —
/// `(chain_id, iota)` in file order.
fn chain_iotas(leaf: &Path) -> Vec<(usize, f64)> {
    let path = leaf.join("chain_starts.tsv");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut body = raw.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let id_idx = cols.iter().position(|c| *c == "chain_id").expect("chain_id column");
    let iota_idx = cols.iter().position(|c| *c == "iota").expect("iota column");
    body.map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        (cells[id_idx].parse().unwrap(), cells[iota_idx].parse().unwrap())
    }).collect()
}

/// The chains that drew the `iota = 0` row and the ones that did not, from the
/// leaf's own record of its starts. Asserts the premise every skip test
/// stands on: the two chains drew different rows.
fn split_chains_by_start(leaf: &Path) -> (Vec<usize>, Vec<usize>) {
    let starts = chain_iotas(leaf);
    assert_eq!(starts.len(), 2, "two chains, two recorded starts: {starts:?}");
    let bad: Vec<usize> = starts.iter().filter(|(_, i)| *i == 0.0).map(|(c, _)| *c).collect();
    let good: Vec<usize> = starts.iter().filter(|(_, i)| *i != 0.0).map(|(c, _)| *c).collect();
    assert!(!bad.is_empty() && !good.is_empty(),
        "the seeded from_posterior draw must hand the two chains different rows for \
         this fixture to test a skip; chain_starts.tsv says {starts:?}. If the draw \
         changed, pick a seed under which the rows differ.");
    (bad, good)
}

/// The 0-based `chain` column of every row in `draws.tsv`.
fn draws_chain_ids(draws: &Path) -> Vec<usize> {
    let raw = std::fs::read_to_string(draws)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", draws.display()));
    let mut lines = raw.lines();
    let header = lines.next().expect("draws.tsv has a header");
    assert!(header.starts_with("chain\tdraw\t"),
        "draws.tsv must lead with the (chain, draw) key columns; got: {header}");
    lines.filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').next().unwrap().parse::<usize>().unwrap())
        .collect()
}

struct Run {
    ok: bool,
    stderr: String,
    stdout: String,
    out_root: PathBuf,
    _tmp: Tmp,
}

/// Drive `camdl fit run` on the fixture with the given (rank-1, rank-2) `iota`
/// starts.
fn run_fit(tag: &str, iotas: (f64, f64)) -> Option<Run> {
    let (Some(bin), Some(camdlc)) = (camdl_bin(), camdlc_bin()) else {
        eprintln!("skip: release camdl / camdlc not built");
        return None;
    };
    let tmp = tempdir(tag);
    let (ir, data) = write_fixture(tmp.path(), &camdlc);

    let draws = write_draws(tmp.path(), iotas);
    let (fit_toml, out_root) = write_fit_toml(tmp.path(), &ir, &data, &draws);
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--seed", "1", "--progress", "none"])
        .output().expect("spawn camdl fit run");

    Some(Run {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        out_root,
        _tmp: tmp,
    })
}

/// gh#607 acceptance 1. One refused start must not kill the fit, and must not
/// leak a single draw into the pooled posterior.
#[test]
fn one_bad_chain_is_skipped_and_survivors_finish() {
    // rank-1 (chain 1, 0-based id 0) impossible; rank-2 (chain 2) healthy.
    let Some(run) = run_fit("skip", (0.0, 0.2)) else { return };

    assert!(run.ok,
        "the fit must succeed when ONE chain's start is refused.\n\
         stdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);

    let stage_dir = cas_stage_leaf(&run.out_root.join("fits"), "pgas")
        .expect("committed `pgas` method leaf");
    let (bad_chains, good_chains) = split_chains_by_start(&stage_dir);

    // Exactly one BadInit, naming the refused chain and the start it actually
    // ran from.
    let bad = bad_init_entries(&run.out_root);
    assert_eq!(bad.len(), 1,
        "expected exactly 1 bad_init diagnostic, got {}: {:#?}\nstderr:\n{}",
        bad.len(), bad, run.stderr);
    let chain_id = bad[0].get("chain_id").and_then(|c| c.as_u64())
        .expect("bad_init must carry a chain_id") as usize;
    assert_eq!(vec![chain_id], bad_chains,
        "the refused chain must be the one that drew the iota = 0 row; \
         bad_init:\n{:#?}", bad[0]);

    // gh#513: the diagnostic quotes the start THIS chain ran from, which is
    // the drawn row — not the `[estimate].start` value of 0.2.
    let params = bad[0].get("params").expect("bad_init must carry params");
    let iota = params.get("iota").and_then(|v| v.as_f64())
        .expect("bad_init.params must include iota");
    assert_eq!(iota, 0.0,
        "bad_init must name the drawn start (iota=0), not the \
         configured `start` (0.2); got {iota}. bad_init:\n{:#?}", bad[0]);

    // The reason must identify WHICH term was non-finite — `observation` is a
    // bad start, `transition` would be a step_one/density bug (gh#80) — and
    // must say the chain was given its probation sweep before being refused.
    let reason = bad[0].get("reason").and_then(|r| r.as_str()).unwrap_or("");
    assert!(reason.contains("observation -inf"),
        "the reason must name the offending component; got: {reason}");
    assert!(reason.contains("still non-finite after the first trajectory update"),
        "the reason must record that the X|θ,y rescue was attempted and failed; \
         got: {reason}");

    // `fit_state.toml`: 1 of 2 chains usable.
    let state_raw = std::fs::read_to_string(stage_dir.join("fit_state.toml")).unwrap();
    let state: toml::Value = toml::from_str(&state_raw).unwrap();
    assert_eq!(state.get("n_good_chains").and_then(|v| v.as_integer()), Some(1),
        "fit_state.toml must record n_good_chains = 1:\n{state_raw}");
    assert_eq!(state.get("n_chains").and_then(|v| v.as_integer()), Some(2),
        "n_chains stays at the requested count:\n{state_raw}");

    // THE load-bearing assertion: the skipped chain contributes no draw.
    let chains = draws_chain_ids(&stage_dir.join("draws.tsv"));
    assert!(!chains.is_empty(), "the surviving chain must have written draws");
    assert!(chains.iter().all(|c| good_chains.contains(c)),
        "draws.tsv must hold ONLY the surviving chain {good_chains:?}; \
         saw chain ids {:?}", {
            let mut u = chains.clone(); u.sort_unstable(); u.dedup(); u
        });

    // And the skip was loud on stderr, not silent.
    assert!(run.stderr.contains("ran 1 of 2 chains"),
        "the run must report `ran 1 of 2 chains`.\nstderr:\n{}", run.stderr);
}

/// gh#607 acceptance 2. Nothing to pool ⇒ the run fails, rather than writing a
/// degenerate posterior and exiting 0.
#[test]
fn all_chains_refused_is_an_error() {
    let Some(run) = run_fit("allbad", (0.0, 0.0)) else { return };

    assert!(!run.ok,
        "a fit whose every chain start is refused must exit NON-ZERO.\n\
         stdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert_eq!(bad_init_entries(&run.out_root).len(), 2,
        "both refused chains must be named individually.\nstderr:\n{}", run.stderr);
    assert!(has_diagnostic(&run.out_root, "initial_loglik_infinite"),
        "the all-refused path must also carry `initial_loglik_infinite` — the \
         signal the gh#226 backstop taught consumers to look for.\nstderr:\n{}",
        run.stderr);
    assert!(run.stderr.contains("refused at their starting point"),
        "the error must say what happened.\nstderr:\n{}", run.stderr);
}

/// gh#607, the OTHER half of the predicate. A chain whose start is `-inf` only
/// because its reference trajectory was an unlucky draw must NOT be refused:
/// the `X|θ,y` move rescues it at the same `θ₀`, and refusing it would throw
/// away a working chain.
///
/// The fixture makes the rescue the ONLY thing that can help. `iota` is tiny
/// but non-zero (1e-4), so at `S = 1000` the expected importation count over
/// the 6-day window is 0.6: the *reference* draw is all-zero (hence `-inf`)
/// with probability ≈ e^(-0.6) ≈ 0.55, while at least one of the 40 conditional
/// SMC particles imports with probability ≈ 1 − e^(-24) — so a trajectory that
/// explains the data exists at this very `θ₀` and CSMC finds it. Both rows use
/// it, so both chains must survive.
#[test]
fn a_start_the_trajectory_move_can_rescue_is_not_refused() {
    let Some(run) = run_fit("rescued", (1e-4, 1e-4)) else { return };

    assert!(run.ok,
        "a chain whose start the X|θ,y move can rescue must not be refused.\n\
         stdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert!(bad_init_entries(&run.out_root).is_empty(),
        "no chain may be refused when the trajectory move rescues the start; \
         got {:#?}\nstderr:\n{}", bad_init_entries(&run.out_root), run.stderr);
    // The run must have exercised the probation path, not simply started
    // finite — otherwise this test would pass without testing anything.
    assert!(run.stderr.contains("chain recovered from a non-finite start"),
        "the fixture must actually START at -inf and recover, or this test is \
         vacuous.\nstderr:\n{}", run.stderr);

    let stage_dir = cas_stage_leaf(&run.out_root.join("fits"), "pgas")
        .expect("committed `pgas` method leaf");
    let mut chains = draws_chain_ids(&stage_dir.join("draws.tsv"));
    chains.sort_unstable();
    chains.dedup();
    assert_eq!(chains, vec![0, 1],
        "both rescued chains must contribute draws; got {chains:?}");
}

/// gh#607 negative control. The guard must be INERT when nothing is wrong: a
/// healthy multi-chain fit keeps every chain, writes no `bad_init`, and leaves
/// `n_good_chains` unset so its `fit_state.toml` is unchanged.
#[test]
fn healthy_fit_keeps_every_chain() {
    let Some(run) = run_fit("healthy", (0.2, 0.3)) else { return };

    assert!(run.ok,
        "a healthy fit must succeed.\nstdout:\n{}\nstderr:\n{}",
        run.stdout, run.stderr);
    assert!(bad_init_entries(&run.out_root).is_empty(),
        "a healthy fit must produce NO bad_init diagnostic; got {:#?}",
        bad_init_entries(&run.out_root));
    assert!(!run.stderr.contains("skipped via BadInit"),
        "a healthy fit must not report a skip.\nstderr:\n{}", run.stderr);

    let stage_dir = cas_stage_leaf(&run.out_root.join("fits"), "pgas")
        .expect("committed `pgas` method leaf");

    let state_raw = std::fs::read_to_string(stage_dir.join("fit_state.toml")).unwrap();
    let state: toml::Value = toml::from_str(&state_raw).unwrap();
    assert!(state.get("n_good_chains").is_none(),
        "a healthy fit must leave n_good_chains unset (an Option::None field is \
         omitted from the TOML), so its fit_state.toml is byte-unchanged by the \
         guard:\n{state_raw}");

    let mut chains = draws_chain_ids(&stage_dir.join("draws.tsv"));
    chains.sort_unstable();
    chains.dedup();
    assert_eq!(chains, vec![0, 1],
        "both chains must contribute draws; got {chains:?}");
}
