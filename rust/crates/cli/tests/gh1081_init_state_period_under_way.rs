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
}

/// Restart `stream`'s model from a saved state at day `origin` and ask for
/// its observations, one file per stream.
fn restart(stream: &str, origin: f64) -> Restart {
    let bin = binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, model(stream)).unwrap();
    let state = dir.join("final.tsv");
    write_state(&state, origin);
    let obs_dir = dir.join("obs");
    let out = run(&bin, &[
        "simulate".into(), s(&model_path),
        "--param".into(), "beta=0.4".into(),
        "--param".into(), "gamma=0.15".into(),
        "--init-state".into(), s(&state),
        "--seed".into(), "3".into(),
        "--obs-dir".into(), s(&obs_dir),
        "--output-dir".into(), s(&dir.join("cas")),
    ]);
    Restart { _tmp: tmp, out, obs_dir }
}

fn assert_refused(r: &Restart, period: &str) {
    let err = String::from_utf8_lossy(&r.out.stderr);
    assert!(
        !r.out.status.success(),
        "a restart inside a reporting period must be refused, not written: {err}"
    );
    assert!(err.contains("'weekly'"), "the refusal names the stream: {err}");
    assert!(err.contains(period), "the refusal names the period {period}: {err}");
    assert!(err.contains("t = 25"), "the refusal names the restart time: {err}");
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
