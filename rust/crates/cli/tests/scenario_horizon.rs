//! End-to-end tests for a per-scenario simulation horizon
//! (`scenarios { x { simulate { to = ... } } }`), gh#561.
//!
//! The compiler has always resolved the field into `Preset::t_end`, but the
//! runtime dropped it on the floor: every scenario ran to the model-level
//! `t_end` with no diagnostic. These tests pin the two halves of the fix —
//! the cell's *window* and the cell's *identity* — and the two properties
//! that must NOT move (the paired-seed prefix, and the hash of every model
//! that does not use the feature).
//!
//! Proposal: `docs/dev/proposals/2026-08-13-per-scenario-simulation-horizon.md`

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
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// A deterministic SIR whose scenarios differ ONLY in their declared horizon.
/// No `set`/`scale`/`enable`/`disable`, so the scenario-level identity digest
/// is byte-identical across all three — which is exactly what makes the
/// collision test (below) meaningful.
fn write_horizon_menu_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

init { S = 990  I = 10  R = 0 }

transitions {
  infection : S --> I  @ beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 100 'days }

scenarios {
  shorter { simulate { to = 40 'days } }
  same    { simulate { to = 100 'days } }
  longer  { simulate { to = 160 'days } }
}
"#;
    std::fs::write(path, src).unwrap();
}

/// The same model with no per-scenario horizon at all — the negative control
/// for the identity test. `plain` must hash exactly as it does today.
fn write_no_horizon_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

init { S = 990  I = 10  R = 0 }

transitions {
  infection : S --> I  @ beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 100 'days }

scenarios {
  plain { }
}
"#;
    std::fs::write(path, src).unwrap();
}

fn write_params(path: &Path) {
    std::fs::write(path, "beta = 0.35\ngamma = 0.1\n").unwrap();
}

struct Run {
    out: PathBuf,
    store: PathBuf,
    _tmp: tempfile::TempDir,
}

/// Simulate `scenarios` from `model` into a private CAS store on `backend`,
/// returning the wide-format TSV path plus the store root.
fn simulate_on(
    bin: &Path,
    model: &Path,
    params: &Path,
    scenarios: &[&str],
    backend: &str,
) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("traj.tsv");
    let store = tmp.path().join("store");
    let mut cmd = Command::new(bin);
    cmd.args([
        "simulate",
        &model.to_string_lossy(),
        "--params",
        &params.to_string_lossy(),
        "--backend",
        backend,
        "--seed",
        "7",
        "--output-dir",
        &store.to_string_lossy(),
        "-o",
        &out.to_string_lossy(),
    ]);
    for s in scenarios {
        cmd.args(["--scenario", s]);
    }
    let o = cmd.output().expect("spawn");
    assert!(
        o.status.success(),
        "simulate failed; stderr: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    Run { out, store, _tmp: tmp }
}

/// The deterministic default: window and identity claims do not need RNG.
fn simulate(bin: &Path, model: &Path, params: &Path, scenarios: &[&str]) -> Run {
    simulate_on(bin, model, params, scenarios, "ode")
}

/// Rows of the wide TSV as `(scenario, t, S)`, skipping the `#` version
/// comment and the header. A single-scenario run has no `scenario` column,
/// so the caller passes the name to attribute rows to.
fn rows(path: &Path) -> Vec<(String, f64, f64)> {
    let content = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();
    let header: Vec<&str> = lines[0].split('\t').collect();
    let scen_i = header.iter().position(|c| *c == "scenario");
    let t_i = header.iter().position(|c| *c == "t").expect("t column");
    let s_i = header.iter().position(|c| *c == "S").expect("S column");
    lines[1..]
        .iter()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let scen = scen_i.map(|i| f[i].to_string()).unwrap_or_default();
            (scen, f[t_i].parse().unwrap(), f[s_i].parse().unwrap())
        })
        .collect()
}

fn last_t(rows: &[(String, f64, f64)], scenario: &str) -> f64 {
    rows.iter()
        .filter(|(s, _, _)| s == scenario)
        .map(|(_, t, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Every `run.json` under `<store>/sims` (the trajectory leaves; the ensemble
/// record lives under `ensembles/` and is not a cell).
fn sim_records(store: &Path) -> Vec<serde_json::Value> {
    fn walk(dir: &Path, out: &mut Vec<serde_json::Value>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let s = std::fs::read_to_string(&p).unwrap();
                out.push(serde_json::from_str(&s).unwrap());
            }
        }
    }
    let mut out = Vec::new();
    walk(&store.join("sims"), &mut out);
    out
}

/// The distinct `run_id`s among the trajectory leaves. This — NOT the leaf
/// count — is the collision check: the store PATH carries a readable label
/// segment per level, so cells with distinct labels land in distinct
/// directories even when their identity is byte-identical. The `run_id` is
/// derived from the level HASHES alone, so it is the thing that actually
/// collides.
fn distinct_run_ids(store: &Path) -> Vec<String> {
    let mut ids: Vec<String> = sim_records(store)
        .iter()
        .map(|r| r["run_id"].as_str().expect("run.json has a run_id").to_string())
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// The identity-bearing hash suffix of each level, in path order — the
/// `run.json` `levels` array with the readable labels dropped.
fn level_hashes(store: &Path) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = sim_records(store)
        .iter()
        .map(|r| {
            r["levels"]
                .as_array()
                .expect("levels array")
                .iter()
                .map(|l| l["hash"].as_str().expect("level hash").to_string())
                .collect()
        })
        .collect();
    out.sort();
    out
}

// ── The window ──────────────────────────────────────────────────────────────

/// gh#561, the reported bug: a scenario declaring a LONGER horizon than the
/// model must run to its own horizon, not the model's. Before the fix every
/// scenario stopped at the model-level `t_end` (100) with no diagnostic.
#[test]
fn longer_scenario_horizon_extends_that_cell() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_horizon_menu_model(&model);
    write_params(&params);

    let run = simulate(&bin, &model, &params, &["longer"]);
    let r = rows(&run.out);
    let last = r.iter().map(|(_, t, _)| *t).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        last, 160.0,
        "scenario `longer` declares `simulate {{ to = 160 }}` against a model \
         horizon of 100, so its last output time must be 160, got {last}"
    );
}

/// The other direction, and the isolation property: a SHORTER horizon confines
/// that scenario only — the sibling scenarios in the same ensemble keep their
/// own windows. Ragged `t` across scenarios is the intended shape.
#[test]
fn shorter_scenario_horizon_truncates_only_that_cell() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_horizon_menu_model(&model);
    write_params(&params);

    let run = simulate(&bin, &model, &params, &["shorter", "same", "longer"]);
    let r = rows(&run.out);
    assert_eq!(last_t(&r, "shorter"), 40.0, "`shorter` must stop at its own 40");
    assert_eq!(last_t(&r, "same"), 100.0, "`same` must stop at the model's 100");
    assert_eq!(last_t(&r, "longer"), 160.0, "`longer` must run to its own 160");
}

/// The prefix-safety property the whole design rests on (proposal §2): `to` is
/// legal as a per-scenario overlay precisely because extending or truncating
/// the horizon never re-tiles `[t_start, old_end]`. Two scenarios differing
/// ONLY in horizon must therefore agree exactly over their shared span — which
/// is what keeps paired-seed CRN meaningful across a horizon menu.
///
/// Run on the STOCHASTIC backends. The argument in §2 is about the RNG stream
/// being consumed per substep in state order; under a fixed-step ODE the prefix
/// is identical by construction, so an ODE-only version of this test would
/// remove the very mechanism the claim is about and could not detect a
/// paired-seed break (`.claude/rules/sim-and-inference.md`: tests follow the
/// matrix).
#[test]
fn differing_horizons_share_an_identical_prefix_chain_binomial() {
    assert_prefix_identical("chain_binomial");
}

#[test]
fn differing_horizons_share_an_identical_prefix_gillespie() {
    assert_prefix_identical("gillespie");
}

fn assert_prefix_identical(backend: &str) {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_horizon_menu_model(&model);
    write_params(&params);

    let run = simulate_on(&bin, &model, &params, &["shorter", "longer"], backend);
    let r = rows(&run.out);
    let short: Vec<(f64, f64)> =
        r.iter().filter(|(s, _, _)| s == "shorter").map(|(_, t, v)| (*t, *v)).collect();
    let long: Vec<(f64, f64)> =
        r.iter().filter(|(s, _, _)| s == "longer").map(|(_, t, v)| (*t, *v)).collect();

    assert!(!short.is_empty() && short.len() < long.len(), "expected a shorter prefix");
    for (i, (t, v)) in short.iter().enumerate() {
        assert_eq!(
            (*t, *v),
            long[i],
            "prefix diverged at row {i}: a horizon override must not re-tile \
             the substep grid before the old end"
        );
    }
}

// ── The identity ────────────────────────────────────────────────────────────

/// The horizon menu is the case that FORCES the identity half. These three
/// cells share a model, params, seed, and an identical scenario-level digest
/// (`enable`/`disable`/`params` are all empty), and the scenario label is
/// provenance, not identity. Without the per-cell horizon in the config level
/// they resolve to ONE `run_id` and the store serves one trajectory for three
/// different questions.
#[test]
fn scenarios_differing_only_in_horizon_get_distinct_run_ids() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_horizon_menu_model(&model);
    write_params(&params);

    let run = simulate(&bin, &model, &params, &["shorter", "same", "longer"]);
    let ids = distinct_run_ids(&run.store);
    assert_eq!(
        ids.len(),
        3,
        "three scenarios differing only in horizon must resolve to three \
         distinct run_ids, got {}: {:#?}",
        ids.len(),
        ids
    );
}

/// The negative control for the re-key: routing the horizon through
/// `SimConfig::t_end` must be inert for a model that does not use the feature.
/// A scenario with no `simulate {}` block, and one whose `to` EQUALS the model
/// horizon, must both land on the leaf they land on today — same store path,
/// same hash segments.
#[test]
fn absent_or_equal_horizon_does_not_rekey() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let params = tmp.path().join("p.toml");
    write_params(&params);

    // `plain` declares no horizon; `same` declares one equal to the model's.
    // Both models are otherwise identical, so if the horizon is folded in
    // correctly (as a resolved VALUE, not as "was it declared?") the two cells
    // land on the same leaf path relative to their store roots.
    let no_horizon = tmp.path().join("plain.camdl");
    let menu = tmp.path().join("menu.camdl");
    write_no_horizon_model(&no_horizon);
    write_horizon_menu_model(&menu);

    let a = simulate(&bin, &no_horizon, &params, &["plain"]);
    let b = simulate(&bin, &menu, &params, &["same"]);

    // Levels are (model, config, params, scenario, seed) in path order. The
    // horizon lives in `config`, so that is the one to compare: the two models
    // differ in source text (three scenarios vs one), so their MODEL level
    // legitimately differs and is not part of this claim.
    let config_of = |run: &Run| -> Vec<String> {
        level_hashes(&run.store).iter().map(|ls| ls[1].clone()).collect()
    };
    assert_eq!(
        config_of(&a),
        config_of(&b),
        "a scenario whose `to` equals the model horizon must produce the same \
         config-level digest as one that declares no horizon at all — the \
         identity folds the resolved VALUE, not whether it was declared"
    );

    // And the scenario level stays empty-delta-identical for both, so the
    // horizon has not leaked into the scenario digest (which would re-key
    // every existing scenario'd run in every store).
    let scenario_of = |run: &Run| -> Vec<String> {
        level_hashes(&run.store).iter().map(|ls| ls[3].clone()).collect()
    };
    assert_eq!(
        scenario_of(&a),
        scenario_of(&b),
        "the horizon must not enter the SCENARIO level: both scenarios carry \
         an empty enable/disable/set delta and must hash identically"
    );
}

// ── Composition ─────────────────────────────────────────────────────────────

/// `extends` has always inherited a parent's horizon (the expander merges it);
/// `compose` must too, walking the chain the way `set` and `scale` already do.
/// Otherwise `combined { compose = [endemic] }` silently runs to the model
/// horizon while `endemic` runs to its own — the same silent drop gh#561 is
/// about, one level up from the parser, and the identity would key on the
/// window the cell does not run.
fn write_compose_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

init { S = 990  I = 10  R = 0 }

transitions {
  infection : S --> I  @ beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 80 'days }

scenarios {
  endemic  { simulate { to = 160 'days } }
  combined { compose = [endemic] }
  inherit  { extends = endemic }
  own      { compose = [endemic]  simulate { to = 200 'days } }
}
"#;
    std::fs::write(path, src).unwrap();
}

#[test]
fn a_composed_scenario_inherits_the_horizon_and_its_own_wins() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_compose_model(&model);
    write_params(&params);

    let run = simulate(&bin, &model, &params, &["endemic", "combined", "inherit", "own"]);
    let r = rows(&run.out);

    assert_eq!(last_t(&r, "endemic"), 160.0, "the declaring scenario runs its own window");
    assert_eq!(
        last_t(&r, "combined"),
        160.0,
        "`compose = [endemic]` must inherit endemic's horizon, not fall back to \
         the model's 80 — `set`/`scale` compose, and the horizon is a preset \
         field like any other"
    );
    assert_eq!(last_t(&r, "inherit"), 160.0, "`extends` inherits it (unchanged)");
    assert_eq!(
        last_t(&r, "own"),
        200.0,
        "a composing scenario's OWN `to` applies last and wins, mirroring how \
         its own `set` wins over a composed member's"
    );

    // Identity follows the same composed value: endemic/combined/inherit all
    // resolve to 160 and carry an identical (empty) scenario delta, so they are
    // legitimately ONE cell; `own` runs 200 and must be a second. Two windows,
    // two run_ids — and in particular `combined` must not be keyed on the model
    // horizon it does not run.
    let ids = distinct_run_ids(&run.store);
    assert_eq!(
        ids.len(),
        2,
        "expected two distinct windows (160 and 200) → two run_ids, got {ids:#?}"
    );
}

// ── The observation axis ────────────────────────────────────────────────────

/// A model with an observation stream, so the horizon's SECOND time axis is
/// exercised. `prevalence(I)` is the dangerous projection: past the end of the
/// trajectory every reader clamps to the last snapshot, so a fabricated tail is
/// a frozen compartment wearing fresh observation noise — it reads as a
/// perfectly plausible plateau rather than as obviously missing data.
fn write_observed_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
  rho   : probability in [0.0, 1.0]
}

init { S = 990  I = 10  R = 0 }

transitions {
  infection : S --> I  @ beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}

observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    cases         ~ poisson(rate = rho * projected)
    emit_schedule = every 10 'days
  }
}

quantities { last_seen = final(observations.cases) }

simulate { from = 0 'days  to = 100 'days }

scenarios {
  shorter { simulate { to = 40 'days } }
}
"#;
    std::fs::write(path, src).unwrap();
}

/// gh#561: the observation emission schedule must follow the CELL's horizon.
///
/// The expander bakes `ObsRegular.end` from the MODEL-level `simulate { to }`
/// at compile time, so it is a copy of the model horizon, not an author-declared
/// observation end. Trusting it once a scenario can shorten the window emits
/// observations past the end of that scenario's own trajectory — and every
/// reader clamps, so those rows are FABRICATED. Exit code 0, no warning, and for
/// a prevalence stream the tail is not obviously fake.
///
/// This is the same silent-wrong class gh#561 exists to remove, on the second
/// time axis. It reaches `simulate --obs`, the CAS `obs/` subtree, `[synthetic]`
/// dataset generation, and obs-sourced `quantities {}`.
#[test]
fn a_shortened_horizon_emits_no_observations_past_its_own_trajectory() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_observed_model(&model);
    std::fs::write(&params, "beta = 0.35\ngamma = 0.1\nrho = 0.5\n").unwrap();

    let store = tmp.path().join("store");
    let obs = tmp.path().join("obs.tsv");
    let traj = tmp.path().join("traj.tsv");
    let out = Command::new(&bin)
        .args([
            "simulate",
            &model.to_string_lossy(),
            "--params",
            &params.to_string_lossy(),
            "--backend",
            "ode",
            "--seed",
            "7",
            "--scenario",
            "shorter",
            "--output-dir",
            &store.to_string_lossy(),
            "--obs",
            &obs.to_string_lossy(),
            "-o",
            &traj.to_string_lossy(),
        ])
        .output()
        .expect("spawn");
    assert!(
        out.status.success(),
        "simulate failed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The trajectory stops at the scenario's own 40.
    let traj_last = rows(&traj).iter().map(|(_, t, _)| *t).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(traj_last, 40.0, "the scenario's trajectory must end at its declared 40");

    // So must the observations. Before the fix these ran to the model's 100,
    // with t = 50..100 read off the frozen final snapshot.
    let txt = std::fs::read_to_string(&obs).unwrap();
    let obs_times: Vec<f64> = txt
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .skip(1)
        .map(|l| l.split('\t').next().unwrap().parse().unwrap())
        .collect();
    assert!(!obs_times.is_empty(), "the scenario must still emit its own observations");
    let obs_last = obs_times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        obs_last, 40.0,
        "observations must stop at the cell's horizon (40), not the model's \
         (100) — rows past the trajectory are read off the clamped final \
         snapshot and are fabricated. Emitted times: {obs_times:?}"
    );
}
