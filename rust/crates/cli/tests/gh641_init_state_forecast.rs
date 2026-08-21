//! gh#641 — `camdl simulate --init-state`: forecast forward from a filtered
//! state instead of the model's `init {}` block.
//!
//! The write side already existed (`pfilter --save-final-state`, an unweighted
//! sample from p(x_T | y_{1:T}) at the filter's θ) and the engine's
//! start-from-state seam already existed (gh#322, splice-verified in
//! `sim/tests/splice_invariant.rs`). What did not exist was a user surface that
//! joins them, and — the part that makes a stale answer possible — a run
//! identity that notices the state changed.
//!
//! These tests shell out to the release binary, so they exercise the real
//! pfilter → file → simulate → CAS path.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Particle count for the fixture filter, and therefore the replicate count of
/// every forecast here (replicate i restores particle row i). Small enough to
/// keep the CAS leaf count down, large enough that the bootstrap filter does
/// not collapse on a Poisson-likelihood SIR.
const N_SMALL: usize = 24;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing() -> PathBuf {
    let b = binary();
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        b.display()
    );
    b
}

const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  daily_cases {
    columns       { time : time, daily_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    daily_cases   ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 30 'days }
"#;

/// The forecast model: identical dynamics, a longer declared horizon so the
/// run has somewhere to go after the forecast origin.
const MODEL_LONG: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  daily_cases {
    columns       { time : time, daily_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    daily_cases   ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 90 'days }
"#;

/// The forecast model with a `#[lineage]` annotation, so `simulate --event-log`
/// is reachable on it. Same compartments and transition names as `MODEL_LONG`,
/// so a state file written for one describes the other.
const MODEL_LINEAGE: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  #[lineage]
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

simulate { from = 0 'days  to = 90 'days }
"#;

/// A model whose `emit_schedule` lists a time OUTSIDE its own simulation window
/// (`at [0, 20, 40]` against `from = 10`). gh#589 rejects it: the run has no
/// snapshot at t=0, so the observation cannot be projected. Used as the control
/// that the forecast-origin confinement does not silently drop it instead.
const MODEL_AT_TIMES: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  daily_cases {
    columns       { time : time, daily_cases : count }
    projected     = incidence(infection)
    emit_schedule = at [0 'days, 20 'days, 40 'days]
    daily_cases   ~ poisson(rate = projected)
  }
}

simulate { from = 10 'days  to = 40 'days }
"#;

struct Fixture {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    model: PathBuf,
    model_long: PathBuf,
    params: PathBuf,
    data: PathBuf,
    /// The `pfilter --save-final-state` output.
    state: PathBuf,
}

fn run(bin: &Path, args: &[String]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Simulate observations, filter them, and save the final particle states.
fn fixture(bin: &Path, particles: usize) -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let model = dir.join("sir.camdl");
    let model_long = dir.join("sir_long.camdl");
    std::fs::write(&model, MODEL).unwrap();
    std::fs::write(&model_long, MODEL_LONG).unwrap();
    let params = dir.join("p.toml");
    std::fs::write(&params, "beta = 0.4\ngamma = 0.15\n").unwrap();

    let data = dir.join("cases.tsv");
    let out = run(bin, &[
        "simulate".into(), s(&model), "--params".into(), s(&params),
        "--obs".into(), s(&data), "--seed".into(), "7".into(),
        "--output-dir".into(), s(&dir.join("cas_gen")),
    ]);
    assert!(out.status.success(), "data gen failed: {}", String::from_utf8_lossy(&out.stderr));

    let state = dir.join("final.tsv");
    let out = run(bin, &[
        "pfilter".into(), s(&model), "--params".into(), s(&params),
        format!("--data={}", s(&data)),
        "--particles".into(), particles.to_string(),
        "--seed".into(), "1".into(),
        "--save-final-state".into(), s(&state),
    ]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));

    Fixture { _tmp: tmp, dir, model, model_long, params, data, state }
}

/// Parse the state file: (origin_t, header columns, rows of fields).
fn read_state(path: &Path) -> (f64, Vec<String>, Vec<Vec<String>>) {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let hdr = lines.next().unwrap();
    assert!(hdr.starts_with("# camdl-final-state v1"), "header line: {hdr}");
    let t: f64 = hdr
        .split('\t')
        .find_map(|f| f.trim().strip_prefix("t="))
        .expect("header carries t=")
        .parse()
        .unwrap();
    let cols: Vec<String> = lines.next().unwrap().split('\t').map(str::to_string).collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    (t, cols, rows)
}

/// Read a trajectory TSV into (header, rows).
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let hdr: Vec<String> = lines.next().unwrap().split('\t').map(str::to_string).collect();
    let rows = lines.map(|l| l.split('\t').map(str::to_string).collect()).collect();
    (hdr, rows)
}

fn col<'a>(hdr: &[String], rows: &'a [Vec<String>], name: &str) -> Vec<&'a str> {
    let i = hdr.iter().position(|h| h == name)
        .unwrap_or_else(|| panic!("column `{name}` in {hdr:?}"));
    rows.iter().map(|r| r[i].as_str()).collect()
}

/// Every SIM leaf under a CAS root, as (run_id, config-level hash). A multi-cell
/// `simulate` also commits one `sim_ensemble` leaf referencing the cells; that
/// is a different artifact kind, not a per-cell trajectory.
fn leaf_ids(root: &Path) -> Vec<(String, String)> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.file_name().is_some_and(|n| n == "run.json") {
                    out.push(path);
                }
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    let mut out: Vec<(String, String)> = files
        .iter()
        .filter_map(|f| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
            if v["kind"] != "sim" {
                return None;
            }
            let cfg = v["levels"].as_array().unwrap().iter()
                .find(|l| l["name"] == "config").expect("config level")
                ["hash"].as_str().unwrap().to_string();
            Some((v["run_id"].as_str().unwrap().to_string(), cfg))
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

// ── The workflow ─────────────────────────────────────────────────────────────

/// The headline: the file the filter wrote seeds a forward run at the origin
/// the file declares, and the trajectory's first row IS the restored state —
/// not the model's `init {}` (S=990, I=10, R=0).
#[test]
fn forecast_starts_from_the_restored_state_at_the_declared_origin() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);

    let (origin, scols, srows) = read_state(&f.state);
    assert_eq!(origin, 30.0, "the filter's last observation time is t=30");
    assert_eq!(srows.len(), N_SMALL, "one row per particle");

    let traj = f.dir.join("fc.tsv");
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--replicates".into(), N_SMALL.to_string(),
        "--seed".into(), "3".into(),
        "-o".into(), s(&traj),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    assert!(out.status.success(), "forecast failed: {}", String::from_utf8_lossy(&out.stderr));

    let (hdr, rows) = read_tsv(&traj);
    let times: Vec<f64> = col(&hdr, &rows, "t").iter().map(|v| v.parse().unwrap()).collect();
    let tmin = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(tmin, 30.0, "the forecast starts at the file's origin, not the model's t_start");
    assert_eq!(tmax, 90.0, "and runs to the model's declared horizon");

    // Replicate 1's first row must be particle row 0's counts, exactly.
    let reps = col(&hdr, &rows, "replicate");
    let first_idx = (0..rows.len())
        .find(|&i| reps[i] == "1" && times[i] == 30.0)
        .expect("replicate 1 has a row at the origin");
    for comp in ["S", "I", "R"] {
        let si = scols.iter().position(|c| c == comp).unwrap();
        assert_eq!(
            col(&hdr, &rows, comp)[first_idx],
            srows[0][si],
            "compartment {comp} at the origin must be particle row 0's count, \
             not the model's init {{}}"
        );
    }

    // Negative control: the model's own init is NOT what we started from.
    let s0 = col(&hdr, &rows, "S")[first_idx];
    assert_ne!(s0, "990", "starting from init {{}} would put S at 990");

    // Every replicate restores its OWN row.
    for rep in 1..=N_SMALL {
        let i = (0..rows.len())
            .find(|&i| reps[i] == rep.to_string() && times[i] == 30.0)
            .unwrap_or_else(|| panic!("replicate {rep} has an origin row"));
        let si = scols.iter().position(|c| c == "I").unwrap();
        assert_eq!(
            col(&hdr, &rows, "I")[i], srows[rep - 1][si],
            "replicate {rep} must restore particle row {}", rep - 1
        );
    }
}

/// Compose with gh#626: `--init-state` supplies the origin, `--to "last_obs +
/// N"` supplies the horizon. This is the actual forecast invocation.
#[test]
fn composes_with_an_observation_anchored_horizon() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);

    // A bare fit.toml is enough for an anchored `--to` — only its
    // [data.observations] is consulted.
    let fit = f.dir.join("fit.toml");
    std::fs::write(
        &fit,
        format!(
            "[model]\ncamdl = \"{}\"\n\n[data.observations]\ndaily_cases = \"{}\"\n\n[estimate]\nbeta = {{ bounds = [0.05, 1.0], start = 0.4 }}\n\n[fixed]\ngamma = 0.15\n\n[stages.posterior]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\niterations = 2\ncooling = 0.7\n",
            s(&f.model_long), s(&f.data)
        ),
    )
    .unwrap();

    let traj = f.dir.join("fc.tsv");
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--to".into(), "last_obs + 8 weeks".into(),
        "--fit".into(), s(&fit),
        "--replicates".into(), N_SMALL.to_string(),
        "--seed".into(), "3".into(),
        "-o".into(), s(&traj),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    assert!(
        out.status.success(),
        "--init-state + --to failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let (hdr, rows) = read_tsv(&traj);
    let times: Vec<f64> = col(&hdr, &rows, "t").iter().map(|v| v.parse().unwrap()).collect();
    let tmin = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // last_obs = 30, + 8 weeks = 30 + 56 = 86.
    assert_eq!(tmin, 30.0, "origin from --init-state");
    assert_eq!(tmax, 86.0, "horizon from --to \"last_obs + 8 weeks\"");
}

/// Synthetic observations from a forecast run cover the forecast window and
/// nothing before it. The emit schedule starts at t=0, but the run starts at
/// the origin — an observation time before the run's first snapshot cannot be
/// projected at all, so it is not this run's to emit.
///
/// The row AT the origin is emitted, carrying 0 for an incidence stream: its
/// interval has zero length. That is the same convention `t_start` already has
/// on an ordinary run (see `incidence_t0.rs`), not something the forecast
/// origin introduces.
#[test]
fn synthetic_observations_cover_the_forecast_window_only() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);

    let obs = f.dir.join("fc_obs.tsv");
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--replicates".into(), N_SMALL.to_string(),
        "--seed".into(), "3".into(),
        "--obs".into(), s(&obs),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    assert!(
        out.status.success(),
        "--init-state + --obs failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let (hdr, rows) = read_tsv(&obs);
    let times: Vec<f64> = col(&hdr, &rows, "time").iter().map(|v| v.parse().unwrap()).collect();
    assert!(!times.is_empty(), "the forecast must emit observations");
    assert_eq!(
        times.iter().cloned().fold(f64::INFINITY, f64::min), 30.0,
        "no observation may precede the forecast origin"
    );
    assert_eq!(
        times.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 90.0,
        "and the series runs to the horizon"
    );
}

/// A run that does NOT restart must be untouched by the origin confinement.
///
/// An `emit_schedule = at [...]` list carrying a time outside the model's own
/// window is a declaration the run cannot honour, and gh#589 rejects it. The
/// forecast-origin confinement must not turn that fail-closed guard into silent
/// truncation — emitting 2 of 3 declared observations and exiting 0 is exactly
/// the plausible-looking wrong answer the guard exists to prevent. This is why
/// the origin is an `Option` and not the run's `t_start`.
#[test]
fn a_non_restarted_run_still_fails_closed_on_an_out_of_window_at_list() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let model = dir.join("attimes.camdl");
    // Window [10, 40]; the list declares t=0, which the run has no snapshot for.
    std::fs::write(&model, MODEL_AT_TIMES).unwrap();
    let params = dir.join("p.toml");
    std::fs::write(&params, "beta = 0.4\ngamma = 0.15\n").unwrap();

    let obs = dir.join("obs.tsv");
    let out = run(&bin, &[
        "simulate".into(), s(&model), "--params".into(), s(&params),
        "--obs".into(), s(&obs), "--seed".into(), "1".into(),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "an `at` time outside the run's window must stay a hard error; \
         stderr={err}"
    );
    assert!(
        !obs.exists() || std::fs::read_to_string(&obs).unwrap().lines().count() <= 1,
        "a refused run must not leave a truncated observation file"
    );
}

// ── Identity ─────────────────────────────────────────────────────────────────

/// The mandatory property: the same model and the same seed, seeded from two
/// DIFFERENT state files, must resolve to two different `run_id`s. Otherwise
/// the store serves the first forecast for the second state.
#[test]
fn two_different_state_files_produce_distinct_run_ids() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);

    // A second state file differing in exactly one count. Same shape, same
    // origin — only the state changed, which is precisely the case a
    // model-keyed-only identity would miss.
    let (_t, _cols, rows) = read_state(&f.state);
    let txt = std::fs::read_to_string(&f.state).unwrap();
    let bumped: i64 = rows[0][1].parse::<i64>().unwrap() + 1;
    let mut lines: Vec<String> = txt.lines().map(str::to_string).collect();
    let mut fields: Vec<String> = lines[2].split('\t').map(str::to_string).collect();
    fields[1] = bumped.to_string();
    lines[2] = fields.join("\t");
    let state_b = f.dir.join("final_b.tsv");
    std::fs::write(&state_b, format!("{}\n", lines.join("\n"))).unwrap();

    let store = f.dir.join("store");
    for st in [&f.state, &state_b] {
        let out = run(&bin, &[
            "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
            "--init-state".into(), s(st),
            "--replicates".into(), N_SMALL.to_string(),
            "--seed".into(), "5".into(),
            "--output-dir".into(), s(&store),
        ]);
        assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let ids = leaf_ids(&store);
    // 2 state files × N replicates = 2N distinct leaves. If the file's bytes did
    // not key the run, the second invocation would be served entirely from the
    // first's cache and only N leaves would exist.
    assert_eq!(
        ids.len(), 2 * N_SMALL,
        "two state files × {N_SMALL} replicates must be {} distinct leaves",
        2 * N_SMALL
    );
    let configs: std::collections::BTreeSet<&str> =
        ids.iter().map(|(_, c)| c.as_str()).collect();
    assert_eq!(
        configs.len(), 2 * N_SMALL,
        "each (file, row) pair must have its own config-level hash"
    );
}

/// The same state file re-run at the same seed IS a cache hit — the re-key is
/// scoped to a real change, not blanket cache defeat.
#[test]
fn the_same_state_file_still_hits_the_cache() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);
    let store = f.dir.join("store");
    let args: Vec<String> = vec![
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--replicates".into(), N_SMALL.to_string(),
        "--seed".into(), "5".into(),
        "--output-dir".into(), s(&store),
    ];
    for _ in 0..2 {
        let out = run(&bin, &args);
        assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    assert_eq!(leaf_ids(&store).len(), N_SMALL, "the second run must reuse every leaf");
}

/// The restored ROW must be in the key on top of the seed level. `--seeds 7,7`
/// gives two replicates the SAME `process_seed`, so without the row they
/// resolve to one leaf and the store serves particle row 0's forecast for row 1.
/// The control pins that this is really about the row: with no `--init-state`,
/// those same two cells legitimately DO collapse to one leaf.
#[test]
fn duplicate_seeds_restoring_different_rows_do_not_collide() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);

    let store = f.dir.join("store");
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--seeds".into(), vec!["7"; N_SMALL].join(","),
        "--output-dir".into(), s(&store),
    ]);
    assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        leaf_ids(&store).len(), N_SMALL,
        "replicates restoring different particle rows must not share a leaf, \
         even at an identical process_seed"
    );

    // Control: the same duplicated seeds WITHOUT --init-state genuinely are one
    // run, so the assertion above is about the restored row rather than about
    // `--seeds` defeating the cache.
    let store_ctl = f.dir.join("store_ctl");
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--seeds".into(), vec!["7"; N_SMALL].join(","),
        "--output-dir".into(), s(&store_ctl),
    ]);
    assert!(out.status.success(), "control failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        leaf_ids(&store_ctl).len(), 1,
        "without --init-state, identical seeds ARE one run"
    );
}

// ── Refusals ─────────────────────────────────────────────────────────────────

/// `run_simulate` has a SECOND consumer of the cell spec: the `--event-log`
/// branch routes through `util::run_simulation_event_log`, which shares
/// `resolve_run_model` (so `t_start` moves to the origin) but then builds state
/// from `initial_state` and passes `Resume::default()`.
///
/// Left unguarded that is the worst available outcome: the run silently starts
/// from `init {}`, the output cursor is never re-seated so the time axis is
/// wrong too, AND the leaf commits under the run_id that says the state WAS
/// restored — two different trajectories at one content address. `--event-log`
/// conflicts with `--replicates`, but `--replicates` defaults to 1, so a
/// one-row state file reaches this with no unusual flags.
///
/// The engine seam itself already refuses an observer resume
/// (`chain_binomial.rs`: an attached observer carries mid-run state an injected
/// `(state)` cannot reconstruct). This surfaces that refusal at the CLI instead
/// of letting the event-log path walk around it.
#[test]
fn event_log_with_init_state_is_refused_not_silently_restarted() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let model = dir.join("lineage.camdl");
    std::fs::write(&model, MODEL_LINEAGE).unwrap();
    let params = dir.join("p.toml");
    std::fs::write(&params, "beta = 0.4\ngamma = 0.15\n").unwrap();

    // A hand-written one-row state file: counts deliberately far from the
    // model's `init { S = 990  I = 10  R = 0 }`, so a run that ignores it is
    // unmistakable in the output.
    let state = dir.join("final.tsv");
    write_state(&state, &[("S", 400), ("I", 50), ("R", 550)], 30.0);

    let ev = dir.join("events.tsv");
    let traj = dir.join("traj.tsv");
    let out = run(&bin, &[
        "simulate".into(), s(&model), "--params".into(), s(&params),
        "--init-state".into(), s(&state),
        "--event-log".into(), s(&ev),
        "--seed".into(), "3".into(),
        "-o".into(), s(&traj),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "--init-state + --event-log must be refused, not silently run from \
         init {{}}. stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout), err
    );
    assert!(err.contains("--event-log"), "the refusal must name the flag: {err}");
    assert!(err.contains("--init-state"), "and the other flag: {err}");

    // And nothing may be committed: a leaf here would carry the state-keyed
    // run_id over a trajectory that never restored the state.
    assert_eq!(
        leaf_ids(&dir.join("cas")).len(), 0,
        "a refused run must commit no leaf"
    );
}

/// Write a hand-made state file for `comps` at time `t`, one row.
fn write_state(path: &Path, comps: &[(&str, i64)], t: f64) {
    let names: Vec<&str> = comps.iter().map(|(n, _)| *n).collect();
    let vals: Vec<String> = comps.iter().map(|(_, v)| v.to_string()).collect();
    std::fs::write(
        path,
        format!(
            "# camdl-final-state v1\tt={t}\nparticle\t{}\n0\t{}\n",
            names.join("\t"),
            vals.join("\t")
        ),
    )
    .unwrap();
}

/// A reactive policy carries mid-run state the file does not hold — the
/// observation history its trigger reads, its once/cooldown gating, the queue of
/// effects already scheduled, and its own surveillance RNG stream. Restarting
/// with an empty agenda would produce a forecast in which the policy has never
/// seen anything, which is a different model, not a different starting point.
#[test]
fn a_reactive_model_is_refused_by_policy_name() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json");
    let state = dir.join("final.tsv");
    write_state(&state, &[("S", 800), ("I", 40), ("R", 150), ("V", 10)], 30.0);

    let out = run(&bin, &[
        "simulate".into(), s(&ir),
        "--param".into(), "beta=0.3".into(), "--param".into(), "gamma=0.1".into(),
        "--param".into(), "rho=0.2".into(),
        "--param".into(), "trigger_threshold=2".into(),
        "--param".into(), "sia_cov=0.7".into(),
        "--param".into(), "N0=1000".into(), "--param".into(), "I0=10".into(),
        "--enable".into(), "sia".into(),
        "--init-state".into(), s(&state),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a reactive model must refuse --init-state: {err}");
    assert!(err.contains("reactive intervention"), "{err}");
    assert!(
        err.contains("[sia]"),
        "the refusal must name the policy, not just the class: {err}"
    );
}

/// The particle filter's state is integer counts only, so a real-valued
/// reservoir's value at the origin is recorded nowhere. Refuse by name rather
/// than restart the reservoir from `init {}` while the counts come from the
/// filter — a half-restored state is the plausible-looking wrong answer.
#[test]
fn a_real_compartment_model_is_refused_by_compartment_name() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let ir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_reservoir_mixed.ir.json");
    let state = dir.join("final.tsv");
    write_state(&state, &[("S", 800), ("I", 40), ("R", 150)], 30.0);

    let out = run(&bin, &[
        "simulate".into(), s(&ir),
        "--param".into(), "beta=0.3".into(), "--param".into(), "gamma=0.1".into(),
        "--param".into(), "xi=0.05".into(), "--param".into(), "delta=0.1".into(),
        "--param".into(), "N0=1000".into(), "--param".into(), "I0=10".into(),
        "--init-state".into(), s(&state),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a real-compartment model must refuse: {err}");
    assert!(err.contains("real-valued compartment"), "{err}");
    assert!(err.contains("W1"), "the refusal must name the compartment: {err}");
}

/// The forecast origin must coincide with an output-emit time. Flow
/// accumulators reset only at emits, so an origin between snapshots would seed
/// a window whose first interval is neither the model's nor the file's — the
/// seam rejects it with a located error rather than snapping to a neighbour.
/// This is the constraint the design calls load-bearing, so it is pinned.
#[test]
fn an_off_grid_origin_is_refused_never_snapped() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let model = dir.join("sir.camdl");
    std::fs::write(&model, MODEL_LINEAGE.replace("  #[lineage]\n", "")).unwrap();
    let params = dir.join("p.toml");
    std::fs::write(&params, "beta = 0.4\ngamma = 0.15\n").unwrap();

    // Output cadence is daily, so t = 30.5 lies strictly between two snapshots.
    let state = dir.join("final.tsv");
    write_state(&state, &[("S", 400), ("I", 50), ("R", 550)], 30.5);

    let out = run(&bin, &[
        "simulate".into(), s(&model), "--params".into(), s(&params),
        "--init-state".into(), s(&state),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an off-grid origin must be refused: {err}");
    assert!(
        err.contains("30.5"),
        "the refusal must name the offending time, not just complain: {err}"
    );
    assert_eq!(
        leaf_ids(&dir.join("cas")).len(), 0,
        "a refused run must commit no leaf"
    );
}

/// A backend with no start-from-state seam must refuse, not silently start
/// from `init {}` and call the result a forecast.
#[test]
fn a_backend_without_the_seam_is_refused() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);
    for backend in ["gillespie", "ode"] {
        let out = run(&bin, &[
            "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
            "--init-state".into(), s(&f.state),
            "--backend".into(), backend.into(),
            "--replicates".into(), N_SMALL.to_string(),
            "--output-dir".into(), s(&f.dir.join("cas")),
        ]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{backend} must refuse --init-state");
        assert!(err.contains("start-from-state seam"), "{backend}: {err}");
        assert!(err.contains(backend), "the refusal must name the backend: {err}");
    }
}

/// The saved states are p(x_T | y) at ONE θ. Pairing them with unrelated
/// posterior draws would be an incoherent (θ, x_T) product.
///
/// The refusal now points at `--init-state fit` (gh#697), which IS the paired
/// (θ_i, X_i) source; before that landed it pointed at the blocker instead.
#[test]
fn pairing_with_draws_is_refused() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);
    let draws = f.dir.join("draws.tsv");
    std::fs::write(&draws, "beta\tgamma\n0.4\t0.15\n0.5\t0.2\n").unwrap();
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--draws".into(), s(&draws),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "--init-state + --draws must be refused");
    assert!(err.contains("cannot be combined with --draws"), "{err}");
    assert!(
        err.contains("--init-state fit"),
        "the refusal must point at the paired source: {err}"
    );
}

/// Replicate i restores row i, so a mismatch is refused with both counts —
/// never a silent prefix of an ancestor-ordered swarm.
#[test]
fn a_replicate_row_count_mismatch_is_refused() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL + 1);
    let out = run(&bin, &[
        "simulate".into(), s(&f.model_long), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--replicates".into(), "3".into(),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a replicate/row mismatch must be refused");
    assert!(err.contains(&format!("{} particle rows", N_SMALL + 1)), "{err}");
    assert!(err.contains("3 replicate"), "{err}");
    assert!(err.contains(&format!("--replicates {}", N_SMALL + 1)),
        "the fix must be spelled out: {err}");
}

/// A state file whose origin is at or past the horizon leaves nothing to
/// forecast — refused with both times rather than emitting an empty tail.
#[test]
fn an_origin_at_or_past_the_horizon_is_refused() {
    let bin = skip_if_missing();
    let f = fixture(&bin, N_SMALL);
    // MODEL's own horizon is t=30, which is exactly the forecast origin.
    let out = run(&bin, &[
        "simulate".into(), s(&f.model), "--params".into(), s(&f.params),
        "--init-state".into(), s(&f.state),
        "--replicates".into(), N_SMALL.to_string(),
        "--output-dir".into(), s(&f.dir.join("cas")),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "origin == horizon must be refused");
    assert!(err.contains("nothing to forecast"), "{err}");
}
