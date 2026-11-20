//! A restart from a saved filtered state must not write a reporting period
//! that was under way at the restart time (D100; spec
//! `docs/dev/temporal-architecture.md` §4.4 "Restarts" and §5.2).
//!
//! `pfilter --save-final-state` saves the particles at T, the last stop of the
//! filter's union axis; `simulate --init-state` restarts there. The saved
//! state holds compartment counts only, not the part of a period's total that
//! accrued before T, so a period that began before T and ends after it cannot
//! be written correctly from the file. Before this rule, `simulate` dropped
//! such a row without a message (gh#1081) or, for window columns, wrote a
//! short window as the first row (gh#1085).
//!
//! Rule 1: `simulate --init-state` refuses, before writing anything, when a
//! period of a stream it would write (`--obs`, `--obs-dir`, `--obs-only`,
//! `--obs-only-dir` and the store's `obs/` child) or a quantity would read is
//! under way at T. Rule 2: `pfilter --save-final-state` refuses a state whose
//! stream's data and `emit_schedule` disagree (the D82 check), and notes a
//! period under way at T.
//!
//! The restart tests build their state files by hand (one particle at a
//! chosen T), so they test `simulate` alone; the pfilter tests run the
//! filter. Both shell out to the release binary.
//!
//! The byte-comparison tests compare against output recorded from main at
//! c92e7c9e3, before the rule, under `fixtures/restart_guard/`. To re-record
//! (only ever from a build of main), run this file's `baseline_` tests with
//! `CAMDL_RECORD_RESTART_BASELINE=1`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The two-cadence SIR: daily and weekly incidence on one flow, so a restart
/// at day 25 falls inside the week [21, 28).
const BASE: &str = include_str!("fixtures/forecast/two_cadence_sir.camdl");
const WEEKLY_BLOCK_COVERS: &str = "    covers        = closing_at(time, 7 'days)";

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn bin() -> PathBuf {
    let b = manifest().join("../../target/release/camdl");
    assert!(b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`", b.display());
    b
}

fn params() -> PathBuf {
    manifest().join("tests/fixtures/forecast/truth.params.toml")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn run(dir: &Path, args: &[String]) -> Output {
    Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("pf_store"))
        .output().expect("spawn camdl")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// `BASE` with each `(from, to)` replacement applied; each `from` must occur.
fn variant(edits: &[(&str, &str)]) -> String {
    let mut m = BASE.to_string();
    for (from, to) in edits {
        assert!(m.contains(from), "the fixture no longer contains {from:?}");
        m = m.replace(from, to);
    }
    m
}

const DAILY_BLOCK: &str = "  daily_cases {
    columns       { time : time, daily_cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    daily_cases ~ poisson(rate = rho * projected)
  }
";

const WEEKLY_BLOCK: &str = "  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases ~ neg_binomial(mean = rho * projected, r = k)
  }
";

fn starting_on() -> String {
    variant(&[(WEEKLY_BLOCK_COVERS, "    covers        = starting_on(time, 7 'days)")])
}

/// The weekly stream with `window_start` / `window_stop` columns: its windows
/// are the gaps between consecutive emit times, the first opening at the
/// model's start.
fn window_columns(schedule: &str) -> String {
    variant(&[(WEEKLY_BLOCK, &format!("  weekly_cases {{
    columns       {{ ws : window_start, we : window_stop, weekly_cases : count }}
    projected     = incidence(infection)
    emit_schedule = {schedule}
    weekly_cases ~ neg_binomial(mean = rho * projected, r = k)
  }}
"))])
}

/// The weekly stream as a ratio of flows: the share of the week's events that
/// were recoveries, observed with normal noise (a likelihood that reads no
/// data column, so `--obs-dir` can draw it).
fn ratio_of_flows() -> String {
    variant(&[(WEEKLY_BLOCK, "  weekly_share {
    columns       { time : time, weekly_share : real }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(recovery) / (incidence(infection) + incidence(recovery))
    emit_schedule = every 7 'days
    weekly_share ~ normal(mean = projected, sd = 0.05)
  }
")])
}

fn weekly_only() -> String {
    variant(&[(DAILY_BLOCK, "")])
}

/// A model in weeks with a four-week reporting period: a restart at week 6
/// falls inside the period [4, 8).
const WEEKS_MODEL: &str = "
time_unit = 'weeks
compartments { S, I, R }
parameters {
  beta : rate
  gamma : rate
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
init { S = 995  I = 5  R = 0 }
simulate { from = 0 'weeks  to = 20 'weeks }
observations {
  monthly_cases {
    columns       { time : time, monthly_cases : count }
    covers        = closing_at(time, 4 'weeks)
    projected     = incidence(infection)
    emit_schedule = every 4 'weeks
    monthly_cases ~ poisson(rate = 0.5 * projected)
  }
}
";

/// A hand-made one-particle saved state at `t` (the `--save-final-state`
/// format), in a population of 1000.
fn write_state(path: &Path, t: f64) {
    std::fs::write(path, format!(
        "# camdl-final-state v1\tt={t}\nparticle\tS\tI\tR\n0\t600\t150\t250\n")).unwrap();
}

struct Restart {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    model: PathBuf,
    state: PathBuf,
}

/// A directory holding `model` and a one-particle state at `t`.
fn restart(model: &str, t: f64) -> Restart {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("model.camdl");
    std::fs::write(&path, model).unwrap();
    let state = dir.join("state.tsv");
    write_state(&state, t);
    Restart { _tmp: tmp, dir, model: path, state }
}

impl Restart {
    /// `simulate --init-state` from the state, seed 3, store under `cas/`,
    /// with `extra` flags.
    fn simulate(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate".into(), s(&self.model), "--params".into(), s(&params()),
            "--init-state".into(), s(&self.state), "--seed".into(), "3".into(),
            "--output-dir".into(), s(&self.dir.join("cas")),
        ];
        args.extend(extra.iter().map(|a| a.to_string()));
        run(&self.dir, &args)
    }

    fn path(&self, rel: &str) -> String {
        s(&self.dir.join(rel))
    }
}

/// Every `run.json` under the store: a committed leaf.
fn leaves(root: &Path) -> usize {
    fn walk(p: &Path, n: &mut usize) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, n);
                } else if path.file_name().is_some_and(|f| f == "run.json") {
                    *n += 1;
                }
            }
        }
    }
    let mut n = 0;
    walk(root, &mut n);
    n
}

/// The refusal names the stream, the period under way and the restart time,
/// and says what to use instead.
fn assert_refused(out: &Output, stream: &str, period: &str, t: &str) {
    let err = stderr(out);
    assert!(!out.status.success(), "the restart must be refused; stderr:\n{err}");
    assert!(err.contains(&format!("stream '{stream}' cannot be written from this restart: \
        the restart time t = {t} falls inside its period {period}")),
        "the refusal must name the stream, the period and t; stderr:\n{err}");
    assert!(err.contains("camdl fit predict"), "and name the alternative; stderr:\n{err}");
}

// ── Rule 1: one test per covers form ─────────────────────────────────────────

/// gh#1081: `closing_at` dropped label 28 (the week [21, 28)) without a
/// message and wrote label 35 first.
#[test]
fn closing_at_period_under_way_is_refused() {
    let r = restart(BASE, 25.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
}

/// `starting_on` dropped label 21 (the week [21, 28)) without a message.
#[test]
fn starting_on_period_under_way_is_refused() {
    let r = restart(&starting_on(), 25.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
}

/// gh#1085: window columns wrote the row ws = 25, we = 28, a three-day total
/// for a period no data row uses.
#[test]
fn window_columns_period_under_way_is_refused() {
    let r = restart(&window_columns("every 7 'days"), 25.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
}

/// A ratio of flows reads both flows over the period, so it is refused the
/// same way.
#[test]
fn ratio_of_flows_period_under_way_is_refused() {
    let r = restart(&ratio_of_flows(), 25.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_share", "[21, 28)", "25");
}

/// Fourteen-day periods on a seven-day schedule overlap, so every restart
/// time falls inside one: the refusal does not offer a clean restart time.
#[test]
fn overlapping_periods_are_refused_without_a_clean_restart_time() {
    let r = restart(&variant(&[(WEEKLY_BLOCK_COVERS, "    covers        = closing_at(time, 14 'days)")]), 21.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_cases", "[14, 28)", "21");
    let err = stderr(&out);
    assert!(err.contains("every time falls inside one of them"), "{err}");
    assert!(!err.contains("writes it cleanly"), "{err}");
}

/// A four-week period in a model whose time unit is weeks: week 6 is inside
/// [4, 8).
#[test]
fn a_four_week_period_in_a_weeks_model_is_refused() {
    let r = restart(WEEKS_MODEL, 6.0);
    let out = Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &s(&r.model), "--param", "beta=2.0", "--param", "gamma=1.0",
            "--init-state", &s(&r.state), "--seed", "3", "--output-dir", &r.path("cas"),
            "--obs-dir", &r.path("od")])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert_refused(&out, "monthly_cases", "[4, 8)", "6");
    // At a boundary of the period it runs.
    write_state(&r.state, 8.0);
    let out = Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &s(&r.model), "--param", "beta=2.0", "--param", "gamma=1.0",
            "--init-state", &s(&r.state), "--seed", "3", "--output-dir", &r.path("cas"),
            "--obs-dir", &r.path("od")])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(), "a restart on a period boundary runs: {}", stderr(&out));
}

/// Window columns with `emit_schedule = at [30, 45]`: the first window opens
/// at the model's start, so at T = 25 the window [0, 30) is under way, and
/// main wrote [25, 30). Restarted at the model's start, nothing is under way.
#[test]
fn an_at_list_window_opening_at_the_model_start_is_refused_inside_and_runs_at_the_start() {
    let model = window_columns("at [30 'days, 45 'days]");
    let r = restart(&model, 25.0);
    let out = r.simulate(&["--obs-dir", &r.path("od")]);
    assert_refused(&out, "weekly_cases", "[0, 30)", "25");
    let r0 = restart(&model, 0.0);
    let out = r0.simulate(&["--obs-dir", &r0.path("od")]);
    assert!(out.status.success(), "a restart at the model's start runs: {}", stderr(&out));
    let weekly = std::fs::read_to_string(r0.dir.join("od/weekly_cases.tsv")).unwrap();
    let rows: Vec<&str> = weekly.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "both windows are written: {weekly}");
    assert!(rows[0].starts_with("0\t30\t") && rows[1].starts_with("30\t45\t"), "{weekly}");
}

// ── Rule 1: quantities, --emit-every, every output route ─────────────────────

fn with_quantity(q: &str) -> String {
    variant(&[("simulate { from", &format!("quantities {{ {q} }}\n\nsimulate {{ from"))])
}

/// A quantity over a stream whose period is under way at T, with no `--obs`
/// flag, is refused; one over a stream on a boundary at T runs.
#[test]
fn a_quantity_over_a_period_under_way_is_refused_without_any_obs_flag() {
    let r = restart(&with_quantity("peak_weekly = max(observations.weekly_cases)"), 25.0);
    let out = r.simulate(&["--to", "49", "--quantities-out", &r.path("q")]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
    assert!(!r.dir.join("q").exists(), "a refusal writes no quantities");

    let r = restart(&with_quantity("peak_daily = max(observations.daily_cases)"), 25.0);
    let out = r.simulate(&["--to", "49", "--quantities-out", &r.path("q")]);
    assert!(out.status.success(),
        "a quantity over the daily stream, on a boundary at 25, runs: {}", stderr(&out));
}

/// `--emit-every 7` re-widens the daily stream to weeks [0, 7), ... so the
/// week [21, 28) is under way at 25 and the daily stream is named first;
/// `--emit-every 5` puts a boundary of both streams at 25.
#[test]
fn emit_every_moves_the_periods_the_guard_judges() {
    let r = restart(BASE, 25.0);
    let out = r.simulate(&["--to", "49", "--obs-dir", &r.path("od"), "--emit-every", "7"]);
    assert_refused(&out, "daily_cases", "[21, 28)", "25");
    let out = r.simulate(&["--to", "50", "--obs-dir", &r.path("od5"), "--emit-every", "5"]);
    assert!(out.status.success(), "a cadence with a boundary at 25 runs: {}", stderr(&out));
}

/// Every route that writes a stream is refused, and a refusal writes
/// nothing: no trajectory file, no observation file or directory, no store
/// leaf.
#[test]
fn every_output_route_is_refused_and_a_refusal_writes_nothing() {
    let routes: [(&str, &str); 4] = [
        ("--obs", "obs.tsv"),
        ("--obs-dir", "od"),
        ("--obs-only", "only.tsv"),
        ("--obs-only-dir", "only_dir"),
    ];
    for (flag, target) in routes {
        let r = restart(&weekly_only(), 25.0);
        // `-o` is refused beside the trajectory-suppressing routes by clap.
        let traj = r.path("traj.tsv");
        let dest = r.path(target);
        let mut args = vec!["--to", "49", flag, dest.as_str()];
        if !flag.starts_with("--obs-only") {
            args.extend(["-o", traj.as_str()]);
        }
        let out = r.simulate(&args);
        assert_refused(&out, "weekly_cases", "[21, 28)", "25");
        assert!(!r.dir.join("traj.tsv").exists(), "{flag}: no -o file");
        assert!(!r.dir.join(target).exists(), "{flag}: no {target}");
        assert_eq!(leaves(&r.dir.join("cas")), 0, "{flag}: no store leaf");
    }
}

/// A stream the filter was never given data for is judged by its declared
/// schedule like any other: the filter binds the daily data only, its state
/// sits at 25, and the weekly stream's week [21, 28) is under way.
#[test]
fn a_stream_with_no_bound_data_is_refused_when_its_schedule_puts_a_period_across_t() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), BASE).unwrap();
    std::fs::write(dir.join("daily.tsv"), daily_through(25)).unwrap();
    let pf = run(dir, &[
        "pfilter".into(), "model.camdl".into(), "--params".into(), s(&params()),
        "--data".into(), "daily_cases=daily.tsv".into(),
        "--particles".into(), "20".into(), "--seed".into(), "1".into(),
        "--save-final-state".into(), "state.tsv".into(),
    ]);
    assert!(pf.status.success(), "pfilter: {}", stderr(&pf));
    assert!(stderr(&pf).contains("'weekly_cases'") && stderr(&pf).contains("[21, 28)"),
        "the save notes the week under way: {}", stderr(&pf));
    let out = run(dir, &[
        "simulate".into(), "model.camdl".into(), "--params".into(), s(&params()),
        "--init-state".into(), "state.tsv".into(), "--replicates".into(), "20".into(),
        "--to".into(), "49".into(), "--obs-dir".into(), "od".into(),
        "--output-dir".into(), "cas".into(),
    ]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
}

/// `--init-state fit` takes its origin from a PGAS fit's saved paths, which
/// end at the last observation, day 25: the same guard applies to that
/// origin, and main wrote label 35 as the first weekly row.
#[test]
fn a_pgas_origin_inside_a_period_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), BASE).unwrap();
    std::fs::write(dir.join("daily.tsv"), daily_through(25)).unwrap();
    std::fs::write(dir.join("weekly.tsv"), fixture_rows("weekly_cases.tsv", 21.0)).unwrap();
    std::fs::write(dir.join("fit.toml"), r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
daily_cases = "daily.tsv"
weekly_cases = "weekly.tsv"
[estimate]
beta = { bounds = [0.05, 2.0], start = 0.5 }
[fixed]
gamma = 0.25
rho = 0.5
k = 10.0
N0 = 1000
I0 = 5
[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 50
sweeps = 20
burn_in = 10
thin = 1
n_trajectories = 4
"#).unwrap();
    let fit = run(dir, &["fit".into(), "run".into(), "fit.toml".into(), "--seed".into(), "1".into()]);
    assert!(fit.status.success(), "fit run: {}", stderr(&fit));
    let segment = std::fs::read_dir(dir.join("results/fits")).unwrap().flatten()
        .map(|e| e.path()).find(|p| p.is_dir()).expect("one fit segment");
    let out = run(dir, &[
        "simulate".into(), "model.camdl".into(), "--fit".into(), s(&segment),
        "--draws".into(), "posterior".into(), "--init-state".into(), "fit".into(),
        "--to".into(), "49".into(), "--obs-dir".into(), "od".into(),
        "--seed".into(), "2".into(), "--output-dir".into(), "cas".into(),
    ]);
    assert_refused(&out, "weekly_cases", "[21, 28)", "25");
    assert!(!dir.join("od").exists(), "a refusal writes no observation directory");
    assert_eq!(leaves(&dir.join("cas")), 0, "a refusal commits no leaf");
}

// ── Rule 2: pfilter --save-final-state ───────────────────────────────────────

fn fixture_rows(file: &str, through: f64) -> String {
    let text = std::fs::read_to_string(manifest().join("tests/fixtures/forecast").join(file)).unwrap();
    let mut lines = text.lines();
    let mut out = format!("{}\n", lines.next().unwrap());
    for l in lines {
        let t: f64 = l.split('\t').next().unwrap().parse().unwrap();
        if t <= through {
            out += l;
            out += "\n";
        }
    }
    out
}

fn daily_through(t: u32) -> String {
    fixture_rows("daily_cases.tsv", t as f64)
}

fn pfilter_save(dir: &Path, model: &str, data: &[(&str, &str)]) -> Output {
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    let mut args = vec![
        "pfilter".into(), "model.camdl".into(), "--params".into(), s(&params()),
        "--particles".into(), "20".into(), "--seed".into(), "1".into(),
        "--save-final-state".into(), "state.tsv".into(),
    ];
    for (stream, text) in data {
        let file = format!("{stream}.tsv");
        std::fs::write(dir.join(&file), text).unwrap();
        args.push("--data".into());
        args.push(format!("{stream}={file}"));
    }
    run(dir, &args)
}

/// F6: weekly data closing on 11, 18, 25 against a schedule giving
/// [0, 7), [7, 14), [14, 21). The save is refused before filtering, with the
/// forecast's D82 text, and no state file is written.
#[test]
fn a_save_whose_data_and_schedule_disagree_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let out = pfilter_save(tmp.path(), BASE, &[
        ("daily_cases", &daily_through(25)),
        ("weekly_cases", "time\tweekly_cases\n11\t40\n18\t90\n25\t120\n"),
    ]);
    let err = stderr(&out);
    assert!(!out.status.success(), "the save must be refused: {err}");
    assert!(err.contains("stream 'weekly_cases' is refused --save-final-state: its data imply \
        the period [4, 11) (training row 0), but its emit_schedule gives [0, 7) over the same \
        reporting interval"), "{err}");
    assert!(!tmp.path().join("state.tsv").exists(), "a refused save writes no state");
}

/// Rule 2 judges the training rows only, under the fit's `holdout_after`:
/// window rows [0, 7), [7, 14), [14, 21), [21, 30) against `every 7 'days`
/// (windows [0, 7), ..., [21, 28)). The row [21, 30) disagrees, so the save
/// is refused without a holdout; with `holdout_after = 21` it is a held-out
/// row, the training rows agree, and the state is saved at 21.
#[test]
fn the_save_check_judges_only_the_training_rows_under_holdout_after() {
    let model = window_columns("every 7 'days");
    let weekly = "ws\twe\tweekly_cases\n0\t7\t10\n7\t14\t40\n14\t21\t90\n21\t30\t150\n";
    let fit_toml = |holdout: &str| format!(r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data]
{holdout}
[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"
[estimate.beta]
bounds = [0.05, 2.0]
start = 0.5
[fixed]
gamma = 0.25
rho = 0.5
k = 10.0
N0 = 1000
I0 = 5
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 20
iterations = 1
cooling = 0.5
"#);
    for (holdout, saves) in [("", false), ("holdout_after = 21", true)] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("model.camdl"), &model).unwrap();
        std::fs::write(dir.join("daily_cases.tsv"), daily_through(30)).unwrap();
        std::fs::write(dir.join("weekly_cases.tsv"), weekly).unwrap();
        std::fs::write(dir.join("fit.toml"), fit_toml(holdout)).unwrap();
        let out = run(dir, &[
            "pfilter".into(), "model.camdl".into(), "--fit".into(), "fit.toml".into(),
            "--params".into(), s(&params()),
            "--particles".into(), "20".into(), "--seed".into(), "1".into(),
            "--save-final-state".into(), "state.tsv".into(), "--no-store".into(),
        ]);
        let err = stderr(&out);
        if saves {
            assert!(out.status.success(), "{holdout}: the training rows agree, so it saves: {err}");
            let header = std::fs::read_to_string(dir.join("state.tsv")).unwrap();
            assert!(header.starts_with("# camdl-final-state v1\tt=21\n"), "{header}");
        } else {
            assert!(!out.status.success(), "without a holdout the row [21, 30) is judged: {err}");
            assert!(err.contains("stream 'weekly_cases' is refused --save-final-state: its data \
                imply the period [21, 30) (training row 3), but its emit_schedule gives [21, 28)"),
                "{err}");
            assert!(!dir.join("state.tsv").exists());
        }
    }
}

/// gh#1301 carried into Rule 2: an `ending_on` weekly stream whose data
/// (labelled 7, 14, 21: [1, 8), [8, 15), [15, 22)) agree with the periods its
/// schedule's labels give saves, and notes the week [22, 29) under way at 25.
#[test]
fn an_ending_on_stream_whose_declarations_agree_saves() {
    let tmp = tempfile::tempdir().unwrap();
    let model = variant(&[(WEEKLY_BLOCK_COVERS, "    covers        = ending_on(time, 7 'days)")]);
    let out = pfilter_save(tmp.path(), &model, &[
        ("daily_cases", &daily_through(25)),
        ("weekly_cases", &fixture_rows("weekly_cases.tsv", 21.0)),
    ]);
    assert!(out.status.success(), "agreeing declarations save: {}", stderr(&out));
    assert!(tmp.path().join("state.tsv").exists());
    assert!(stderr(&out).contains("[22, 29)"), "{}", stderr(&out));
}

/// The two-cadence save agrees and succeeds, and says which period of which
/// stream is under way at the saved time.
#[test]
fn a_save_inside_a_period_succeeds_with_a_note_naming_it() {
    let tmp = tempfile::tempdir().unwrap();
    let out = pfilter_save(tmp.path(), BASE, &[
        ("daily_cases", &daily_through(25)),
        ("weekly_cases", &fixture_rows("weekly_cases.tsv", 21.0)),
    ]);
    let err = stderr(&out);
    assert!(out.status.success(), "the save succeeds: {err}");
    assert!(err.contains("note: stream 'weekly_cases': its period [21, 28) is under way at the \
        saved time t = 25"), "{err}");
    assert!(err.contains("simulate --init-state"), "{err}");
    assert!(!err.contains("'daily_cases': its period"), "the daily stream is on a boundary: {err}");
}

// ── Byte comparisons against output recorded from main ───────────────────────

fn baseline_dir(case: &str) -> PathBuf {
    manifest().join("tests/fixtures/restart_guard").join(case)
}

/// The file with its first line dropped when it is the build-version comment
/// (`# 0.1.0+<hash>`), which names the commit and so differs between builds.
fn content(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if bytes.starts_with(b"# ") {
        let nl = bytes.iter().position(|&b| b == b'\n').map(|i| i + 1).unwrap_or(bytes.len());
        bytes[nl..].to_vec()
    } else {
        bytes
    }
}

/// Compare (or, under `CAMDL_RECORD_RESTART_BASELINE=1`, record) each of
/// `files`, relative to the run directory, against `case`'s recorded copy.
fn compare_with_main(case: &str, dir: &Path, files: &[&str]) {
    let base = baseline_dir(case);
    let record = std::env::var_os("CAMDL_RECORD_RESTART_BASELINE").is_some();
    for f in files {
        let got = content(&dir.join(f));
        let want = base.join(f.replace('/', "__"));
        if record {
            std::fs::create_dir_all(&base).unwrap();
            std::fs::write(&want, &got).unwrap();
        } else {
            let expected = std::fs::read(&want).unwrap_or_else(|e| panic!("{}: {e}", want.display()));
            assert!(got == expected, "{case}/{f} differs from the output recorded from main:\n\
                got:\n{}\nmain:\n{}", String::from_utf8_lossy(&got), String::from_utf8_lossy(&expected));
        }
    }
}

/// A restart on a boundary of every period (T = 21) writes what main wrote.
#[test]
fn baseline_a_boundary_restart_is_byte_identical_to_main() {
    let r = restart(BASE, 21.0);
    let out = r.simulate(&["--to", "49", "-o", &r.path("traj.tsv"), "--obs-dir", &r.path("od")]);
    assert!(out.status.success(), "{}", stderr(&out));
    compare_with_main("boundary_21", &r.dir, &["traj.tsv", "od/daily_cases.tsv", "od/weekly_cases.tsv"]);
}

/// A restart inside the week that writes no stream (trajectory only) is not
/// judged, and writes what main wrote.
#[test]
fn baseline_a_trajectory_only_restart_inside_a_period_is_byte_identical_to_main() {
    let r = restart(BASE, 25.0);
    let out = r.simulate(&["--to", "49", "-o", &r.path("traj.tsv")]);
    assert!(out.status.success(), "{}", stderr(&out));
    compare_with_main("trajectory_only_25", &r.dir, &["traj.tsv"]);
}

/// An instant stream (prevalence read every 7 days) has no period, so a
/// restart between its read times is not refused.
#[test]
fn baseline_an_instant_stream_is_unaffected() {
    let model = variant(&[(WEEKLY_BLOCK, "  prevalence {
    columns       { time : time, prevalence : count }
    projected     = I
    emit_schedule = every 7 'days
    prevalence ~ poisson(rate = rho * projected)
  }
")]);
    let r = restart(&model, 25.0);
    let out = r.simulate(&["--to", "49", "-o", &r.path("traj.tsv"), "--obs-dir", &r.path("od")]);
    assert!(out.status.success(), "{}", stderr(&out));
    compare_with_main("instant_25", &r.dir, &["traj.tsv", "od/daily_cases.tsv", "od/prevalence.tsv"]);
}

/// Window columns with `emit_schedule = at [10, 20]`: no emit time follows
/// T = 25, so no window closes after it and the writer writes none; the
/// restart is not refused and writes what main wrote.
#[test]
fn baseline_a_window_stream_with_no_emit_time_after_t_is_unaffected() {
    let r = restart(&window_columns("at [10 'days, 20 'days]"), 25.0);
    let out = r.simulate(&["-o", &r.path("traj.tsv"), "--obs-dir", &r.path("od")]);
    assert!(out.status.success(), "{}", stderr(&out));
    compare_with_main("window_at_list_25", &r.dir,
        &["traj.tsv", "od/daily_cases.tsv", "od/weekly_cases.tsv"]);
}

/// A fit-only stream (no `emit_schedule`) is never written, so its week
/// under way does not refuse a run that writes a quantity over another
/// stream.
#[test]
fn baseline_a_fit_only_stream_is_unaffected() {
    let model = variant(&[
        ("    emit_schedule = every 7 'days\n", ""),
        ("simulate { from", "quantities { peak_daily = max(observations.daily_cases) }\n\nsimulate { from"),
    ]);
    let r = restart(&model, 25.0);
    let out = r.simulate(&["--to", "49", "-o", &r.path("traj.tsv"), "--quantities-out", &r.path("q")]);
    assert!(out.status.success(), "{}", stderr(&out));
    fn walk(root: &Path, p: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(p).unwrap().flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.push(s(path.strip_prefix(root).unwrap()));
            }
        }
    }
    let mut files = Vec::new();
    walk(&r.dir, &r.dir.join("q"), &mut files);
    files.sort();
    assert!(!files.is_empty(), "--quantities-out wrote files");
    files.push("traj.tsv".into());
    compare_with_main("fit_only_25", &r.dir, &files.iter().map(String::as_str).collect::<Vec<_>>());
}
