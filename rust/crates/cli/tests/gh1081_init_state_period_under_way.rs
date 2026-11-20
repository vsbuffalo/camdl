//! gh#1081, gh#1085 — `camdl simulate --init-state` refuses to write
//! observations when a reporting period is under way at the restart.
//!
//! A saved state (`pfilter --save-final-state`) holds compartment counts only.
//! It does not hold what each stream has counted since its current period
//! began, so a restart inside a period cannot produce that period's row: the
//! part of the count before the restart is not in the file. A stream with a
//! uniform `covers` form used to drop the row without a word, and the
//! forecaster's first week went missing; a `window_start`/`window_stop` stream
//! wrote a short first window instead. Both are refused by name here
//! (`docs/dev/temporal-architecture.md` §4.4, "Restarts"; §5.2), and the
//! message points at `fit predict --mode forecast`, which carries the period
//! under way.
//!
//! These tests shell out to the release binary.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let b = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        b.display()
    );
    b
}

fn run(bin: &Path, args: &[String]) -> std::process::Output {
    Command::new(bin)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// An SIR whose one stream is described by `stream`, run from day 0 to day 49.
fn model(stream: &str) -> String {
    format!(
        r#"
time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}}

transitions {{
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990  I = 10  R = 0 }}

observations {{
{stream}
}}

simulate {{ from = 0 'days  to = 49 'days }}
"#
    )
}

/// Weekly incidence, each row labelled with the day its week closes: the row
/// labelled 28 counts days 21 to 27.
const WEEKLY_CLOSING_AT: &str = r#"
  weekly {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected + 0.1)
  }"#;

/// Weekly incidence, each row labelled with the first day of its week: the row
/// labelled 21 counts days 21 to 27. Its label is before a restart at day 25
/// while its period runs past it.
const WEEKLY_STARTING_ON: &str = r#"
  weekly {
    columns       { time : time, cases : count }
    covers        = starting_on(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected + 0.1)
  }"#;

/// Weekly incidence whose rows state their own period in two columns.
const WEEKLY_WINDOW_COLUMNS: &str = r#"
  weekly {
    columns       { start : window_start, stop : window_stop, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected + 0.1)
  }"#;

/// A census of the infectious compartment: an instant, with no period.
const WEEKLY_CENSUS: &str = r#"
  weekly {
    columns       { time : time, infectious : count }
    projected     = I
    emit_schedule = every 7 'days
    infectious    ~ poisson(rate = projected + 0.1)
  }"#;

/// Weekly incidence, each row labelled with the last day of its week: the row
/// labelled 7 counts days 1 to 7, and the row labelled 0 would count days -6
/// to 0, a week that begins before the model does and that no run writes.
const WEEKLY_ENDING_ON: &str = r#"
  weekly {
    columns       { time : time, cases : count }
    covers        = ending_on(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected + 0.1)
  }"#;

/// Window columns on a listed schedule with no emit time at the model's
/// start: the first window is days 0 to 6, opening where the run starts.
const WINDOW_COLUMNS_AT_LIST: &str = r#"
  weekly {
    columns       { start : window_start, stop : window_stop, cases : count }
    projected     = incidence(infection)
    emit_schedule = at [7 'days, 14 'days, 21 'days, 28 'days, 35 'days, 42 'days, 49 'days]
    cases         ~ poisson(rate = projected + 0.1)
  }"#;

/// A one-particle saved state at day `t`.
fn write_state(path: &Path, t: f64) {
    std::fs::write(
        path,
        format!("# camdl-final-state v1\tt={t}\nparticle\tS\tI\tR\n0\t800\t60\t140\n"),
    )
    .unwrap();
}

struct Restart {
    _tmp: tempfile::TempDir,
    out: std::process::Output,
    obs_dir: PathBuf,
    /// The run store the command was given.
    store: PathBuf,
}

/// Restart `stream`'s model from a saved state at day `origin` and ask for
/// its observations, one file per stream.
fn restart(stream: &str, origin: f64) -> Restart {
    restart_with(stream, origin, &[])
}

/// [`restart`], with further command-line arguments.
fn restart_with(stream: &str, origin: f64, extra: &[&str]) -> Restart {
    let bin = binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, model(stream)).unwrap();
    let state = dir.join("final.tsv");
    write_state(&state, origin);
    let obs_dir = dir.join("obs");
    let store = dir.join("cas");
    let mut args: Vec<String> = vec![
        "simulate".into(), s(&model_path),
        "--param".into(), "beta=0.4".into(),
        "--param".into(), "gamma=0.15".into(),
        "--init-state".into(), s(&state),
        "--seed".into(), "3".into(),
        "--obs-dir".into(), s(&obs_dir),
        "--output-dir".into(), s(&store),
    ];
    args.extend(extra.iter().map(|a| a.to_string()));
    let out = run(&bin, &args);
    Restart { _tmp: tmp, out, obs_dir, store }
}

/// Whether anything was stored under `root`: a `run.json` at any depth.
fn stored_a_run(root: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(root) else { return false };
    rd.flatten().any(|e| {
        let p = e.path();
        if p.is_dir() { stored_a_run(&p) } else { p.file_name().is_some_and(|n| n == "run.json") }
    })
}

/// The data rows of a written observation file.
fn data_rows(r: &Restart) -> Vec<String> {
    let txt = std::fs::read_to_string(r.obs_dir.join("weekly.tsv")).unwrap();
    txt.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .skip(1)
        .map(str::to_string)
        .collect()
}

fn assert_refused(r: &Restart, period: &str) {
    assert_refused_at(r, period, "t = 25");
}

fn assert_refused_at(r: &Restart, period: &str, at: &str) {
    let err = String::from_utf8_lossy(&r.out.stderr);
    assert!(
        !r.out.status.success(),
        "a restart inside a reporting period must be refused, not written: {err}"
    );
    assert!(err.contains("'weekly'"), "the refusal names the stream: {err}");
    assert!(err.contains(period), "the refusal names the period {period}: {err}");
    assert!(err.contains(at), "the refusal names the restart time, {at}: {err}");
    assert!(
        !err.contains("chain_binomial: running"),
        "the refusal comes before any cell is simulated: {err}"
    );
    assert!(
        !stored_a_run(&r.store),
        "a refused restart stores no run, so no leaf declares observations it never wrote"
    );
    assert!(
        err.contains("fit predict") && err.contains("--mode forecast"),
        "the refusal points at the forecast that carries the period: {err}"
    );
    assert!(
        !r.obs_dir.join("weekly.tsv").exists(),
        "no observation file may be written for the refused stream"
    );
}

/// The headline (gh#1081): a restart at day 25 is inside the week of days 21
/// to 27. That week's row was dropped silently and the series began at the
/// row labelled 35.
#[test]
fn a_closing_at_week_under_way_at_the_restart_is_refused() {
    assert_refused(&restart(WEEKLY_CLOSING_AT, 25.0), "[21, 28)");
}

/// The same week under a start label. The label, 21, is before the restart, so
/// the row is not among the emit times after it; its period still straddles.
#[test]
fn a_starting_on_week_under_way_at_the_restart_is_refused() {
    assert_refused(&restart(WEEKLY_STARTING_ON, 25.0), "[21, 28)");
}

/// gh#1085: a window-column stream wrote the row `25 28 <count>`, a short
/// window that is honest about its own extent but is not the stream's week,
/// so a reader lining it up against weekly data compares four days with seven.
#[test]
fn a_window_column_week_under_way_at_the_restart_is_refused() {
    assert_refused(&restart(WEEKLY_WINDOW_COLUMNS, 25.0), "[21, 28)");
}

/// A restart on a period boundary loses nothing: every week after it is whole.
#[test]
fn a_restart_on_a_period_boundary_still_writes_every_week() {
    for stream in [WEEKLY_CLOSING_AT, WEEKLY_STARTING_ON, WEEKLY_WINDOW_COLUMNS] {
        let r = restart(stream, 28.0);
        let err = String::from_utf8_lossy(&r.out.stderr);
        assert!(r.out.status.success(), "a restart at a week boundary must run: {err}");
        let txt = std::fs::read_to_string(r.obs_dir.join("weekly.tsv")).unwrap();
        let rows = txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).count() - 1;
        assert_eq!(rows, 3, "the weeks of days 28-34, 35-41 and 42-48:\n{txt}");
    }
}

/// A state read has no period, so a restart between two of its times loses
/// nothing and is not refused.
#[test]
fn an_instant_stream_restarts_anywhere() {
    let r = restart(WEEKLY_CENSUS, 25.0);
    let err = String::from_utf8_lossy(&r.out.stderr);
    assert!(r.out.status.success(), "a census has no period under way: {err}");
    let txt = std::fs::read_to_string(r.obs_dir.join("weekly.tsv")).unwrap();
    assert!(txt.lines().any(|l| l.starts_with("28\t")), "the census at day 28 is written:\n{txt}");
}

/// A week that begins before the model does is not a period under way: no
/// run writes the row labelled 0 of an `ending_on` stream (days -6 to 0), so a
/// restart at the model's own start loses nothing and writes every later week.
#[test]
fn a_restart_at_the_models_start_is_not_inside_a_period() {
    let r = restart(WEEKLY_ENDING_ON, 0.0);
    let err = String::from_utf8_lossy(&r.out.stderr);
    assert!(r.out.status.success(), "a restart at simulate.from must run: {err}");
    let rows = data_rows(&r);
    assert_eq!(rows.len(), 6, "the weeks ending on days 7, 14, ..., 42 (the week ending on day 49 closes after the run):\n{rows:?}");
    assert!(rows[0].starts_with("7\t"), "the first week written ends on day 7: {rows:?}");
}

/// The same stream one day into its first week is inside the week of days 1
/// to 7, and is refused.
#[test]
fn an_ending_on_week_under_way_is_refused() {
    assert_refused_at(&restart(WEEKLY_ENDING_ON, 3.0), "[1, 8)", "t = 3");
}

/// A window-column stream's first window opens where the model starts. A
/// restart before the first emit time is inside it: unrefused, the row would
/// read `3 7 <count>` where a run from the start writes `0 7 <count>`.
#[test]
fn a_window_column_first_window_under_way_is_refused() {
    assert_refused_at(&restart(WINDOW_COLUMNS_AT_LIST, 3.0), "[0, 7)", "t = 3");
}

/// `--emit-every` re-widens the periods, so a restart on a declared weekly
/// boundary can be inside a fortnight: day 21 is inside days 14 to 27.
#[test]
fn a_period_widened_by_emit_every_is_refused() {
    for stream in [WEEKLY_CLOSING_AT, WEEKLY_STARTING_ON, WEEKLY_WINDOW_COLUMNS] {
        assert_refused_at(&restart_with(stream, 21.0, &["--emit-every", "14"]), "[14, 28)", "t = 21");
    }
    // Day 28 is a fortnight boundary too, and runs.
    let r = restart_with(WEEKLY_CLOSING_AT, 28.0, &["--emit-every", "14"]);
    let err = String::from_utf8_lossy(&r.out.stderr);
    assert!(r.out.status.success(), "a restart on the widened boundary must run: {err}");
    assert_eq!(data_rows(&r).len(), 1, "the fortnight of days 28 to 41");
}
