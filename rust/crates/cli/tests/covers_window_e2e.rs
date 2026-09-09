//! End-to-end: a declared observation window (gh#833) actually moves the
//! scoring boundary, through the real CLI data path.
//!
//! The claim this pins is the one the whole arc exists to make. `covers =
//! day(time)` says row `D` covers `[D, D+1)`; `closing_at(time, 1 'days)` says
//! row `D` covers `[D−1, D)`. Stated as equivalences, which are stronger than
//! "the number changed" and do not depend on any particular fixture's
//! sensitivity:
//!
//!   day(time) on labels 3,4,5,…   ==   closing_at(time, 1 'days) on labels 4,5,6,…
//!
//! Both declare the periods [3,4), [4,5), … so the logliks agree EXACTLY, and
//! `closing_at` on the SAME file as `day` does not — the shift is real.
//!
//! The committed fixture already declares `closing_at(time, 1 'days)`; the
//! variants are made by editing that declaration in the IR as JSON, so this
//! test does not depend on the OCaml compiler being on PATH (mirroring
//! `dated_data_loader.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn seed_timing_ir() -> serde_json::Value {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_covers_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// The fixture with its one stream's `covers` replaced. `None` removes the
/// declaration — an IR the compiler cannot produce, which the loader refuses.
fn model_with_covers(dir: &Path, name: &str, covers: Option<serde_json::Value>) -> PathBuf {
    let mut v = seed_timing_ir();
    let obs = v["model"]["observations"][0].as_object_mut().expect("one stream");
    match covers {
        Some(c) => { obs.insert("covers".into(), c); }
        None => { obs.remove("covers"); }
    }
    let p = dir.join(name);
    std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    p
}

/// `covers = day(time)` in its lowered form: open at the label, one day wide.
fn day_model(dir: &Path) -> PathBuf {
    model_with_covers(dir, "day.ir.json",
        Some(serde_json::json!({ "kind": "from", "offset": 0.0, "span": 1.0 })))
}

/// `covers = closing_at(time, 1 'days)`: the committed fixture's own
/// declaration, closing at the label, one day wide.
fn closing_model(dir: &Path) -> PathBuf {
    model_with_covers(dir, "closing.ir.json",
        Some(serde_json::json!({ "kind": "until", "offset": 0.0, "span": 1.0 })))
}

/// No declaration on an incidence stream: an IR that did not come from the
/// compiler (E350 refuses it there).
fn undeclared_model(dir: &Path) -> PathBuf {
    model_with_covers(dir, "undeclared.ir.json", None)
}

/// Replace a fixture's `: time` column with a `win_start`/`win_stop` pair and
/// declare `covers = window_columns` — the per-row form, the only one that can
/// state a gap or a row wider than the file's usual spacing.
fn to_window_columns(v: &mut serde_json::Value) {
    let obs = &mut v["model"]["observations"][0];
    obs["covers"] = serde_json::json!({ "kind": "window_columns" });
    let cols = obs["columns"].as_array_mut().expect("columns");
    let time_idx = cols.iter().position(|c| c["role"] == "time").expect("a time column");
    cols.splice(time_idx..=time_idx, [
        serde_json::json!({ "name": "win_start", "role": "window_start" }),
        serde_json::json!({ "name": "win_stop", "role": "window_stop" }),
    ]);
}

/// The fixture with its time column replaced by a `window_start`/`window_stop`
/// pair and `covers` set to `window_columns`.
fn windowed_model(dir: &Path) -> PathBuf {
    let mut v = seed_timing_ir();
    to_window_columns(&mut v);
    let p = dir.join("windowed.ir.json");
    std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    p
}

/// The calendar-anchored sibling fixture (`origin = date("2020-02-24")`), same
/// dynamics, with its stream put in the per-row window form. An `origin` is
/// what lets a boundary cell be a date.
const DATED_ORIGIN: &str = "2020-02-24";

fn dated_windowed_model(dir: &Path) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../sim/tests/fixtures/seed_timing_dated.ir.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    assert_eq!(v["model"]["origin"], DATED_ORIGIN, "the dated fixture carries the origin");
    to_window_columns(&mut v);
    let p = dir.join("windowed_dated.ir.json");
    std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    p
}

/// A daily counts file over 40 rows labelled `3 + k + shift`, with a bump so
/// the series is not all zeros (an all-zero series would agree under any
/// window and prove nothing).
fn write_counts(path: &Path, shift: f64) {
    let mut s = String::from("time\tcases\n");
    for k in 0..40u32 {
        let v = if k >= 20 { (k - 19) * 3 } else { 0 };
        s.push_str(&format!("{}\t{}\n", 3.0 + f64::from(k) + shift, v));
    }
    std::fs::write(path, s).unwrap();
}

/// Daily one-day windows [D, D+1) for D in 3..43, with the row for
/// `[skip, skip+1)` OMITTED — a stated gap, legal under the per-row form.
fn write_window_counts(path: &Path, skip: u32) {
    let mut s = String::from("win_start\twin_stop\tcases\n");
    for k in 0..40u32 {
        let d = 3 + k;
        if d == skip {
            continue;
        }
        let v = if k >= 20 { (k - 19) * 3 } else { 0 };
        s.push_str(&format!("{d}\t{}\t{v}\n", d + 1));
    }
    std::fs::write(path, s).unwrap();
}

/// Per-row windows over days 3..43, one day wide except that days 30, 31 and
/// 32 arrive as a single three-day row — the "publication slipped" shape
/// `docs/camdl-data-spec.md` prints. Each row's value is the counts over the
/// days it covers, so the two renderings below carry identical numbers.
fn window_rows() -> Vec<(u32, u32, u32)> {
    let per_day = |d: u32| if d >= 23 { (d - 22) * 3 } else { 0 };
    let mut rows = Vec::new();
    let mut d = 3u32;
    while d < 43 {
        let stop = if d == 30 { 33 } else { d + 1 };
        rows.push((d, stop, (d..stop).map(per_day).sum()));
        d = stop;
    }
    rows
}

/// Those windows with numeric boundaries — day offsets from the model origin.
fn write_numeric_windows(path: &Path) {
    let mut s = String::from("win_start\twin_stop\tcases\n");
    for (a, b, v) in window_rows() {
        s.push_str(&format!("{a}\t{b}\t{v}\n"));
    }
    std::fs::write(path, s).unwrap();
}

/// The same windows with ISO-date boundaries, rendered through the model's own
/// origin — the form the data spec's window examples are written in.
fn write_dated_windows(path: &Path) {
    let day = |t: u32| ir::caltime::internal_to_date(DATED_ORIGIN, f64::from(t), "days").unwrap();
    let mut s = String::from("win_start\twin_stop\tcases\n");
    for (a, b, v) in window_rows() {
        s.push_str(&format!("{}\t{}\t{v}\n", day(a), day(b)));
    }
    std::fs::write(path, s).unwrap();
}

/// The counts a windowed file states with ISO dates are read over exactly the
/// periods those dates name. Stated as an equivalence, which needs no fixture
/// sensitivity to be meaningful:
///
///   the same windows written as dates  ==  written as day offsets from origin
///
/// The two files carry the same rows — including the three-day row — so if the
/// dates are converted through the model's `origin` and `time_unit` the two
/// score identically, bit for bit; if a boundary landed anywhere else the
/// periods would differ and so would the number.
///
/// Before this, the dated file did not load at all: its start column was put
/// through the value parser on the way to the time conversion, and `f64` has
/// no reading for `2020-02-27`.
#[test]
fn a_dated_windowed_file_is_read_over_the_periods_its_dates_name() {
    let camdl = camdl_bin();
    let tmp = tempdir("dated_windows");
    let numeric = tmp.join("numeric.tsv");
    let dated = tmp.join("dated.tsv");
    write_numeric_windows(&numeric);
    write_dated_windows(&dated);

    // Non-vacuous: the "dated" file really is dated, and the rows really do
    // line up — day 3 from a 2020-02-24 origin is 27 February.
    let first = std::fs::read_to_string(&dated).unwrap();
    let first = first.lines().nth(1).unwrap().to_string();
    assert!(first.starts_with("2020-02-27\t2020-02-28\t"), "dated first row: {first}");
    assert!(window_rows().iter().any(|&(a, b, _)| b - a > 1),
        "the fixture must contain a row wider than a day");

    let model = dated_windowed_model(&tmp);
    let by_date = pfilter_loglik(&camdl, &model, &dated);
    let by_number = pfilter_loglik(&camdl, &model, &numeric);
    assert_eq!(
        by_date, by_number,
        "a boundary written as a date names the same period as the day offset it \
         converts to (dated={by_date}, numeric={by_number})",
    );
}

/// The `closing_at` reading of the same counts: labels shifted +1 (a closing
/// row is labelled by its stop), with the row that closes the gap present as
/// `NA` — a hole, which closes the bin without scoring it.
fn write_counts_with_hole(path: &Path, hole_label: u32) {
    let mut s = String::from("time\tcases\n");
    for k in 0..40u32 {
        let label = 4 + k;
        let v = if k >= 20 { (k - 19) * 3 } else { 0 };
        if label == hole_label {
            s.push_str(&format!("{label}\tNA\n"));
        } else {
            s.push_str(&format!("{label}\t{v}\n"));
        }
    }
    std::fs::write(path, s).unwrap();
}

const BASE_PARAMS: &[&str] = &[
    "--param", "beta=0.6",
    "--param", "gamma=0.2",
    "--param", "lambda=2.0",
    "--param", "w=3.0",
    "--param", "N0=5000",
    "--param", "rho=0.5",
    "--param", "k=20",
    "--param", "tau=2",
];

fn pfilter_loglik(camdl: &Path, model: &Path, data: &Path) -> f64 {
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "500", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    let out = run(camdl, &args);
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!(
            "no loglik in output:\nSTDOUT:{stdout}\nSTDERR:{}",
            String::from_utf8_lossy(&out.stderr)))
}

/// The complete meaning of `covers = day(time)`, stated as an equivalence
/// between two declarations of the same periods:
///
///   day(time) on labels 3,4,5,…   ==   closing_at(time, 1 'days) on labels 4,5,6,…
///
/// Both declare [3,4), [4,5), …: the same bins, the same first period opening at
/// 3 (the warm-up before it simulated but not scored), so the numbers agree
/// EXACTLY. This pins the one-bucket SHIFT (row D under `day` closes at D+1)
/// and the LEADING EDGE at once, with no conditioning knob anywhere — the
/// declaration itself says where scoring begins.
#[test]
fn a_declared_day_window_scores_one_bucket_later_and_opens_at_its_own_start() {
    let camdl = camdl_bin();
    let tmp = tempdir("day");
    let labelled_by_day = tmp.join("by_day.tsv");
    let labelled_by_close = tmp.join("by_close.tsv");
    write_counts(&labelled_by_day, 0.0);
    write_counts(&labelled_by_close, 1.0);

    let day = pfilter_loglik(&camdl, &day_model(&tmp), &labelled_by_day);
    let closing = pfilter_loglik(&camdl, &closing_model(&tmp), &labelled_by_close);
    assert_eq!(
        day, closing,
        "day(time) on labels 3.. must equal closing_at(time, 1 day) on labels 4.., \
         bit for bit — they declare the same periods (day={day}, closing={closing})",
    );

    // Non-vacuous: the two forms on the SAME file declare different periods.
    let closing_same_file = pfilter_loglik(&camdl, &closing_model(&tmp), &labelled_by_day);
    assert_ne!(
        day, closing_same_file,
        "day and closing_at on one file must NOT agree — the shift is real",
    );
}

/// `closing_at(time, 1 'days)` and `day(time)` on one file labelled 1,2,3,…
/// declare different periods — `[t−1, t)` against `[t, t+1)` — and score
/// differently. The migration recorded in the proposal rests on the closing
/// form being what every in-repo file meant; that it reproduced the numbers
/// those files were fitted under was checked against the pinned inference
/// baselines and the pomp fixtures when the migration landed.
#[test]
fn closing_at_and_day_on_one_file_declare_different_periods() {
    let camdl = camdl_bin();
    let tmp = tempdir("closing");
    let data = tmp.join("counts.tsv");
    // Labels 1..=40: the first row closes one day after t_start = 0.
    write_counts(&data, -2.0);

    let closing = pfilter_loglik(&camdl, &closing_model(&tmp), &data);
    let day = pfilter_loglik(&camdl, &day_model(&tmp), &data);
    assert!(closing.is_finite() && day.is_finite(), "closing={closing}, day={day}");
    assert_ne!(
        closing, day,
        "day(time) on the same file must NOT agree — it scores one bucket later",
    );
}

/// An incidence stream whose IR says nothing about what its rows cover has no
/// reading: the compiler never produces such an IR (E350), and the loader
/// refuses one rather than score the file under a convention it never stated.
#[test]
fn an_incidence_stream_with_no_declaration_is_refused_by_the_loader() {
    let camdl = camdl_bin();
    let tmp = tempdir("undeclared");
    let data = tmp.join("counts.tsv");
    write_counts(&data, -2.0);
    let model = undeclared_model(&tmp);
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "50", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    let out = run(&camdl, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "must refuse; stderr:\n{stderr}");
    assert!(stderr.contains("'cases'") && stderr.contains("E350"),
        "names the stream and the rule:\n{stderr}");
}

/// Proposal Testing item 4 through the real loader and filter. A stream stating
/// its windows per row, with the window [10,11) simply absent, is a declared
/// GAP: the flow over (10,11) belongs to no bin and is discarded. Stated as an
/// equivalence with a uniform declaration and an `NA` row:
///
///   windowed, [10,11) omitted  ==  closing_at(time, 1 day) on labels 4..43,
///                                  the row labelled 11 present as NA
///
/// Under the uniform form the NA row is a hole: no likelihood term, but it
/// closes the bin at 11 without scoring it — discarding exactly (10,11). Both
/// sides open at 3, score at 4..10 and 12..43, and discard the same span, so
/// the numbers agree EXACTLY.
#[test]
fn a_stated_gap_under_window_columns_discards_exactly_that_span() {
    let camdl = camdl_bin();
    let tmp = tempdir("gap");

    let windowed = tmp.join("windowed.tsv");
    let holed = tmp.join("holed.tsv");
    write_window_counts(&windowed, 10);
    write_counts_with_hole(&holed, 11);

    let declared_gap = pfilter_loglik(&camdl, &windowed_model(&tmp), &windowed);
    let uniform_with_hole = pfilter_loglik(&camdl, &closing_model(&tmp), &holed);
    assert_eq!(
        declared_gap, uniform_with_hole,
        "a stated gap must discard exactly the uncovered span — the same numbers \
         as an NA row closing that bin unscored (windowed={declared_gap}, \
         uniform={uniform_with_hole})",
    );

    // Non-vacuous: without the hole the uniform side scores (10,11] against a
    // real count, so the gap genuinely removes a term.
    let plain = tmp.join("plain.tsv");
    write_counts_with_hole(&plain, u32::MAX);
    let uniform_no_hole = pfilter_loglik(&camdl, &closing_model(&tmp), &plain);
    assert_ne!(
        declared_gap, uniform_no_hole,
        "the gap must change the scoring; agreeing with the gapless reading means \
         the omitted window was not discarded",
    );
}
