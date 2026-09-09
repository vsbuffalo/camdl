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

// ── The real-shaped fixture (gh#878) ────────────────────────────────────────
//
// Every fixture above is numeric-time, regular-grid, one stream, no holes, no
// data-supplied columns. `tests/fixtures/real_shaped/` is not: a model with an
// `origin` and the three files it is fitted to, carrying the properties a
// published surveillance table has — ISO dates in both window boundaries,
// one-day rows with a three-day and a five-day row among them, a scheduled row
// with no count, a stratified stream in long form beside the wide one, a
// denominator the file supplies, and a prevalence reading taken at the
// windows' own closing column.

/// The compiler, which the real-shaped fixture is compiled with rather than
/// carrying a second committed IR to keep in sync. Built by `build-ocaml`,
/// which both `make test-rust` and `make test-fast` depend on.
fn camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(
        p.exists(),
        "camdlc missing: {} - run `make build-ocaml`, or gate with `make test-rust`",
        p.display()
    );
    p
}

fn real_shaped(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/real_shaped").join(name)
}

/// Compile the committed fixture model into `dir`.
fn compile_real_shaped(dir: &Path) -> PathBuf {
    let ir = dir.join("surveillance.ir.json");
    let out = Command::new(camdlc())
        .arg(real_shaped("surveillance.camdl"))
        .arg("-o")
        .arg(&ir)
        .output()
        .expect("camdlc must invoke");
    assert!(out.status.success(), "camdlc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    ir
}

/// The fixture model's `origin`. Asserted against the model file itself below,
/// so a change there cannot silently leave this reading the wrong calendar.
const REAL_SHAPED_ORIGIN: &str = "2026-06-29";

/// An ISO date as the fixture's own model time: days since `origin`.
fn day_of(date: &str) -> f64 {
    ir::caltime::date_to_internal(REAL_SHAPED_ORIGIN, date, "days")
        .unwrap_or_else(|e| panic!("cannot read '{date}' through the origin: {e:?}"))
}

fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let header = lines.next().expect("header").split('\t').map(str::to_string).collect();
    let rows = lines.map(|l| l.split('\t').map(str::to_string).collect()).collect();
    (header, rows)
}

/// What one scored observation covers: a period, or an instant.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cov {
    Interval(f64, f64),
    Instant(f64),
}

impl Cov {
    /// Where on the time axis the observation sits — a period's close, an
    /// instant's own time. Used only to order the two lists the same way.
    fn at(self) -> f64 {
        match self {
            Cov::Interval(_, stop) => stop,
            Cov::Instant(t) => t,
        }
    }
}

fn sorted(mut v: Vec<(String, Cov)>) -> Vec<(String, Cov)> {
    v.sort_by(|a, b| {
        a.1.at().partial_cmp(&b.1.at()).unwrap().then_with(|| a.0.cmp(&b.0))
    });
    v
}

/// Every observation the fixture's own files state, read off the files rather
/// than written out here — the dates converted through the model's `origin`,
/// the uniform form's rule applied to its labels, holes left out.
fn periods_the_files_state() -> Vec<(String, Cov)> {
    let mut want: Vec<(String, Cov)> = Vec::new();

    let (header, rows) = read_tsv(&real_shaped("bulletin.tsv"));
    assert_eq!(header, ["window_start", "window_stop", "cases", "in_care"],
        "the bulletin's columns are what the model declares");
    assert!(rows.iter().all(|r| r[0].contains('-') && r[1].contains('-')),
        "both boundaries are written as ISO dates, which is the point of the fixture");
    assert!(rows.iter().any(|r| day_of(&r[1]) - day_of(&r[0]) > 1.0),
        "at least one row is wider than a day — no uniform `covers` form states this file");
    assert_eq!(rows.iter().filter(|r| r[2] == "NA").count(), 1,
        "exactly one scheduled row carries no count");
    for r in &rows {
        let (start, stop) = (day_of(&r[0]), day_of(&r[1]));
        if r[2] != "NA" {
            want.push(("cases".to_string(), Cov::Interval(start, stop)));
        }
        // The prevalence stream reads this file's stop column as its `: time`.
        want.push(("in_care".to_string(), Cov::Instant(stop)));
    }

    // `covers = ending_on(week_ending, 7 'days)`: the row labelled D covers
    // [D − 6 days, D + 1 day), the label being the last day included.
    let (header, rows) = read_tsv(&real_shaped("province_cases.tsv"));
    assert_eq!(header, ["week_ending", "province", "province_cases"]);
    assert!(rows.iter().any(|r| r[2] == "NA"), "one province misses a week");
    for r in &rows {
        if r[2] == "NA" {
            continue;
        }
        let label = day_of(&r[0]);
        want.push((format!("province_cases_{}", r[1]), Cov::Interval(label - 6.0, label + 1.0)));
    }

    let (header, rows) = read_tsv(&real_shaped("survey.tsv"));
    assert_eq!(header, ["time", "positives", "tested"]);
    for r in &rows {
        want.push(("survey".to_string(), Cov::Instant(day_of(&r[0]))));
    }

    sorted(want)
}

/// The periods a run actually scored, off the prequential trace.
fn periods_the_run_scored(trace: &serde_json::Value) -> Vec<(String, Cov)> {
    let mut got: Vec<(String, Cov)> = Vec::new();
    for step in trace["steps"].as_array().expect("steps") {
        let t = step["t"].as_f64().expect("a step time");
        for ps in step["per_stream"].as_array().expect("per_stream") {
            let name = ps["stream"].as_str().expect("a stream name").to_string();
            let c = &ps["coverage"];
            let cov = match c["kind"].as_str() {
                Some("interval") => Cov::Interval(
                    c["start"].as_f64().expect("start"),
                    c["stop"].as_f64().expect("stop"),
                ),
                Some("instant") => Cov::Instant(t),
                other => panic!("stream '{name}' has coverage kind {other:?}"),
            };
            got.push((name, cov));
        }
    }
    sorted(got)
}

/// The loader binds the real-shaped fixture over exactly the periods its files
/// state, row by row, with the dates read through the model's own `origin`.
///
/// Everything the fixture carries is under test at once, because it is one
/// bind: the dated window pair (`cases`), the same file's closing column read
/// as an instant by a second stream (`in_care`), a long-format file routed to
/// two stratum leaves by name (`province_cases`), and a stream whose
/// likelihood reads a column the file supplies (`survey`). The expected
/// periods are computed from the files themselves, so the assertion cannot
/// drift from the fixture — only from the loader.
#[test]
fn the_real_shaped_files_are_bound_over_the_periods_their_dates_name() {
    let camdl = camdl_bin();
    let tmp = tempdir("real_shaped");
    assert!(
        std::fs::read_to_string(real_shaped("surveillance.camdl"))
            .unwrap()
            .contains(&format!("origin    = date(\"{REAL_SHAPED_ORIGIN}\")")),
        "the fixture's origin is the calendar this test converts through"
    );
    let ir = compile_real_shaped(&tmp);
    let stem = tmp.join("preq");

    let bind = |stream: &str, file: &str| format!("{stream}={}", real_shaped(file).display());
    let (cases, in_care) = (bind("cases", "bulletin.tsv"), bind("in_care", "bulletin.tsv"));
    let provinces = bind("province_cases", "province_cases.tsv");
    let survey = bind("survey", "survey.tsv");
    let mut args = vec![
        "pfilter", ir.to_str().unwrap(),
        "--particles", "100", "--dt", "0.5", "--seed", "1",
        "--data", &cases,
        "--data", &in_care,
        "--data", &provinces,
        "--data", &survey,
        "--save-prequential", stem.to_str().unwrap(),
    ];
    let params = [
        "--param", "beta=0.35", "--param", "gamma=0.2", "--param", "rho=0.5",
        "--param", "psi=0.3", "--param", "k=20",
    ];
    args.extend_from_slice(&params);
    let out = run(&camdl, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the real-shaped fixture must bind:\n{stderr}");
    assert!(
        !stderr.contains("W326"),
        "every temporal cell in the fixture is a date; a numeric-cell warning means \
         one was read as a day offset:\n{stderr}"
    );

    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}.json", stem.display())).unwrap())
            .unwrap();
    let want = periods_the_files_state();
    let got = periods_the_run_scored(&trace);
    let missing: Vec<_> = want.iter().filter(|w| !got.contains(w)).collect();
    let extra: Vec<_> = got.iter().filter(|g| !want.contains(g)).collect();
    assert!(
        missing.is_empty() && extra.is_empty() && got.len() == want.len(),
        "the bound periods are the ones the files state\n  \
         stated but not scored: {missing:#?}\n  scored but not stated: {extra:#?}\n  \
         counts: scored {} vs stated {}",
        got.len(),
        want.len(),
    );
    assert_eq!(got, want, "the bound periods are the ones the files state, in order");

    // The row with no count states its period and scores nothing: no `cases`
    // term closes at its stop, and the row after it opens at its own start
    // rather than reaching back across the hole.
    let (_, rows) = read_tsv(&real_shaped("bulletin.tsv"));
    let hole = rows.iter().position(|r| r[2] == "NA").expect("a hole");
    let hole_stop = day_of(&rows[hole][1]);
    assert!(
        !got.iter().any(|(s, c)| s == "cases" && c.at() == hole_stop),
        "the row with no count must contribute no likelihood term"
    );
    let next = &rows[hole + 1];
    assert!(
        got.contains(&(
            "cases".to_string(),
            Cov::Interval(day_of(&next[0]), day_of(&next[1]))
        )),
        "the row after the hole opens at its own start, not across it"
    );
}
