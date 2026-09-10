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
//! assuming an assignment. The same lever `pmmh_bad_init_skip.rs` uses.
//!
//! `from_posterior` is a spread rule, so a refused start is redrawn, up to
//! `MAX_START_ATTEMPTS` (gh#887): a chain that drew the `iota = 0` row draws
//! again, and is refused only when every attempt drew it. Each attempt is on
//! the record.
//!
//! ## Acceptance
//!
//! 1. `a_refused_start_is_redrawn_and_the_chain_runs` — exit 0; no
//!    `bad_init`; `chain_starts.tsv` carries a `rejected` row for every
//!    `iota = 0` draw the chain made, naming the refusal, and an `accepted`
//!    row at `iota = 0.2`; both chains contribute draws; `fit_state.toml`
//!    carries no `n_good_chains`.
//! 2. `all_chains_refused_is_an_error` — both rows at `iota = 0` ⇒ ten starts
//!    tried per chain, every one recorded, non-zero exit naming the count, and
//!    an `initial_loglik_infinite` diagnostic, rather than a degenerate
//!    posterior written at exit 0.
//! 3. `healthy_fit_keeps_every_chain` — the negative control. Both rows
//!    healthy ⇒ NO `bad_init`, no `rejected` row, both chains present in
//!    `draws.tsv`, and `fit_state.toml` carries no `n_good_chains` key, so a
//!    healthy fit's output is unchanged by the guard.
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

/// One row of the leaf's `chain_starts.tsv`.
#[derive(Debug, Clone)]
struct StartRow {
    chain_id: usize,
    attempt: usize,
    status: String,
    iota: f64,
    reason: String,
}

/// Every attempt recorded in the leaf's `chain_starts.tsv`, in file order.
fn start_rows(leaf: &Path) -> Vec<StartRow> {
    let path = leaf.join("chain_starts.tsv");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut body = raw.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let col = |name: &str| cols.iter().position(|c| *c == name)
        .unwrap_or_else(|| panic!("{name} column in {cols:?}"));
    let (id, attempt, status, iota, reason) =
        (col("chain_id"), col("attempt"), col("status"), col("iota"), col("reason"));
    body.map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        StartRow {
            chain_id: cells[id].parse().unwrap(),
            attempt: cells[attempt].parse().unwrap(),
            status: cells[status].to_string(),
            iota: cells[iota].parse().unwrap(),
            reason: cells[reason].to_string(),
        }
    }).collect()
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

/// gh#887 (and gh#780). A refused spread start is one unlucky draw, not a
/// verdict on the chain: it is redrawn, the rejection is on the record with
/// its reason, and the chain runs from the first draw the sampler accepts.
#[test]
fn a_refused_start_is_redrawn_and_the_chain_runs() {
    // One impossible row (iota = 0) beside one healthy row.
    let Some(run) = run_fit("redraw", (0.0, 0.2)) else { return };

    assert!(run.ok,
        "the fit must succeed: a refused draw is redrawn.\n\
         stdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);

    let stage_dir = cas_stage_leaf(&run.out_root.join("fits"), "pgas")
        .expect("committed `pgas` method leaf");
    let rows = start_rows(&stage_dir);
    let rejected: Vec<&StartRow> = rows.iter().filter(|r| r.status == "rejected").collect();
    let accepted: Vec<&StartRow> = rows.iter().filter(|r| r.status == "accepted").collect();
    assert!(!rejected.is_empty(),
        "the seeded draw must hand at least one chain the iota = 0 row for this \
         fixture to exercise a redraw; chain_starts.tsv rows: {rows:?}. If the draw \
         changed, pick a seed under which one chain draws it.");
    assert_eq!(accepted.len(), 2, "every chain ends on an accepted start: {rows:?}");
    assert!(rows.iter().all(|r| r.status != "refused"),
        "no chain may be refused while the other row is scoreable: {rows:?}");
    for r in &rejected {
        assert_eq!(r.iota, 0.0, "only the impossible row is rejected: {r:?}");
        // The reason names which term was non-finite — `observation` is a
        // bad start, `transition` would be a step_one/density bug (gh#80) —
        // and that the chain was given its probation sweep before the redraw.
        assert!(r.reason.contains("observation -inf")
                && r.reason.contains("still non-finite after the first trajectory update"),
            "a rejected row carries the refusal: {r:?}");
    }
    for a in &accepted {
        assert_eq!(a.iota, 0.2, "the accepted start is the scoreable row: {a:?}");
        let n_rejected = rejected.iter().filter(|r| r.chain_id == a.chain_id).count();
        assert_eq!(a.attempt, n_rejected, "the accepted attempt follows the rejections: {a:?}");
    }
    // The header counts the retries, so a skim of the file says it happened.
    let text = std::fs::read_to_string(stage_dir.join("chain_starts.tsv")).unwrap();
    assert!(text.starts_with("# camdl chain_starts; starts=from_posterior "), "{text}");
    assert!(text.lines().next().unwrap().ends_with(&format!("retried={}", rejected.len())), "{text}");

    // Nothing was skipped: no bad_init, every chain in the pool, no
    // n_good_chains, and the redraw was loud on stderr.
    assert!(bad_init_entries(&run.out_root).is_empty(),
        "a redrawn chain is not a refused chain; got {:#?}\nstderr:\n{}",
        bad_init_entries(&run.out_root), run.stderr);
    let mut chains = draws_chain_ids(&stage_dir.join("draws.tsv"));
    chains.sort_unstable();
    chains.dedup();
    assert_eq!(chains, vec![0, 1], "both chains contribute draws; got {chains:?}");
    let state_raw = std::fs::read_to_string(stage_dir.join("fit_state.toml")).unwrap();
    let state: toml::Value = toml::from_str(&state_raw).unwrap();
    assert!(state.get("n_good_chains").is_none(),
        "every chain ran, so n_good_chains is unset:\n{state_raw}");
    assert!(run.stderr.contains("refused") && run.stderr.contains("drawing another"),
        "the redraw must be announced.\nstderr:\n{}", run.stderr);
    assert!(!run.stderr.contains("ran 1 of 2 chains"),
        "nothing was skipped.\nstderr:\n{}", run.stderr);
}

/// gh#607 acceptance 2, under gh#887's retry. Every row is impossible, so
/// every redraw is too: each chain tries ten starts, every one is on the
/// record, and only then is the chain refused. Nothing to pool ⇒ the run
/// fails, naming the count, rather than writing a degenerate posterior and
/// exiting 0.
#[test]
fn all_chains_refused_is_an_error() {
    let Some(run) = run_fit("allbad", (0.0, 0.0)) else { return };

    assert!(!run.ok,
        "a fit whose every chain start is refused must exit NON-ZERO.\n\
         stdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    let bad = bad_init_entries(&run.out_root);
    assert_eq!(bad.len(), 2,
        "both refused chains must be named individually.\nstderr:\n{}", run.stderr);
    for b in &bad {
        let reason = b.get("reason").and_then(|r| r.as_str()).unwrap_or("");
        assert!(reason.starts_with("none of 10 starts drawn under `starts = from_posterior "),
            "the refusal must say how many starts were tried: {reason}");
    }
    // The record: ten attempts per chain, nine rejected and the last refused.
    let leaf = {
        let mut hit = None;
        let mut stack = vec![run.out_root.join("fits")];
        while let Some(d) = stack.pop() {
            if d.join("chain_starts.tsv").is_file() { hit = Some(d); break; }
            if let Ok(es) = std::fs::read_dir(&d) {
                for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
            }
        }
        hit.expect("a chain_starts.tsv under the run tree")
    };
    let rows = start_rows(&leaf);
    // `chain_id` is 1-based in the file (gh#781), like the `chain_N/`
    // directories and the stderr refusals.
    for chain in 1..=2 {
        let mine: Vec<&StartRow> = rows.iter().filter(|r| r.chain_id == chain).collect();
        assert_eq!(mine.len(), 10, "chain {chain}: ten attempts on the record: {mine:?}");
        assert_eq!(mine.iter().filter(|r| r.status == "rejected").count(), 9, "{mine:?}");
        assert_eq!(mine.last().unwrap().status, "refused", "{mine:?}");
        assert!(mine.iter().all(|r| r.iota == 0.0), "{mine:?}");
        let attempts: Vec<usize> = mine.iter().map(|r| r.attempt).collect();
        assert_eq!(attempts, (0..10).collect::<Vec<_>>(), "attempts are numbered in order");
    }
    assert!(run.stderr.contains("start 9 of 10 refused"),
        "every attempt is announced.\nstderr:\n{}", run.stderr);
    assert!(has_diagnostic(&run.out_root, "initial_loglik_infinite"),
        "the all-refused path must also carry `initial_loglik_infinite` — the \
         signal the gh#226 backstop taught consumers to look for.\nstderr:\n{}",
        run.stderr);
    // gh#885: and it says WHY an unscoreable start happens and what to do,
    // from the one string all three all-chains-refused errors carry — before
    // this, PGAS's message named neither, and pointed at the observation model
    // and the bounds with no reason attached.
    assert!(run.stderr.contains("standardised distance grows with the square root")
            && run.stderr.contains("`starts = \"from_prior\"`")
            && run.stderr.contains("widening the parameter bounds"),
        "the refusal must carry the shared reason and remedies.\nstderr:\n{}",
        run.stderr);
    assert!(run.stderr.contains("refused at their starting point"),
        "the error must say what happened.\nstderr:\n{}", run.stderr);

    // gh#891: and `progress.json` says it too. This is the worst case a fit
    // can have — nothing ran — and it used to be the case whose progress file
    // said the least: the stage returned without a terminal write, so the last
    // periodic `running` (up to five seconds stale, possibly from before any
    // chain reported) stayed on disk and an agent polling the file saw
    // `running` for ever.
    let progress: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(leaf.join("progress.json"))
            .expect("a stage that ran at all writes progress.json"),
    ).expect("progress.json parses");
    let failed = progress["state"]["failed"].as_object()
        .unwrap_or_else(|| panic!(
            "the run's last self-report must be `failed`, not a stale \
             `running`:\n{progress:#}"));
    let why = failed["reason"].as_str().expect("a failure carries its reason");
    assert!(why.contains("refused at their starting point"),
        "the reason is the one the command printed, not a bare tag: {why}");
    // And the per-chain block beside it names every chain, all refused, so a
    // reader needs nothing else to know what happened.
    let chains = &progress["chains"];
    assert_eq!(chains["total"], 2, "{progress:#}");
    assert_eq!(chains["refused"], 2, "every chain refused:\n{progress:#}");
    assert_eq!(chains["running"], 0, "and none left claiming to run:\n{progress:#}");
    let rows = chains["chains"].as_array().expect("per-chain rows");
    assert_eq!(rows.len(), 2, "{progress:#}");
    for r in rows {
        assert_eq!(r["status"], "refused", "{r:#}");
        assert_eq!(r["reason"], "non_finite_start", "{r:#}");
    }
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
    let rows = start_rows(&stage_dir);
    assert!(rows.iter().all(|r| r.status == "accepted" && r.attempt == 0),
        "a healthy fit redraws nothing: {rows:?}");

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
