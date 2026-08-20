//! End-to-end tests for the dated-data loader (2026-05-22 calendar-time,
//! phase 2). Exercised through the `camdl` CLI so the whole boundary —
//! column detection, date→internal-time conversion, the distinct-substep
//! check, and the origin-missing / mixed-column errors — runs in the
//! production data path, not just the unit-tested core.
//!
//! The model is a copy of `crates/sim/tests/fixtures/seed_timing.ir.json`
//! with an `origin` injected at runtime (the committed fixture has none, so
//! the same file also serves the origin-missing test).
//!
//! Silent-skip if the release `camdl` binary is not built (mirrors
//! seed_timing_e2e.rs). `CAMDL_SKIP_VERSION_CHECK=1` avoids a stale globally
//! installed `camdlc` making the test flaky.

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

fn seed_timing_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json")
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_dated_{}_{}_{}", tag, std::process::id(), ns));
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

/// Write a copy of the seed_timing IR with an `origin` field injected.
fn model_with_origin(dir: &Path, origin: &str) -> PathBuf {
    let src = std::fs::read_to_string(seed_timing_ir()).unwrap();
    // Inject `"origin": "..."` right after the time_unit line.
    let injected = src.replacen(
        "\"time_unit\": \"days\",",
        &format!("\"time_unit\": \"days\",\n    \"origin\": \"{origin}\","),
        1,
    );
    assert!(injected.contains("\"origin\""), "origin injection failed");
    let p = dir.join("seed_timing_origin.ir.json");
    std::fs::write(&p, injected).unwrap();
    p
}

const BASE_PARAMS: &[&str] = &[
    "--param", "beta=0.6",
    "--param", "gamma=0.2",
    "--param", "lambda=2.0",
    "--param", "w=3.0",
    "--param", "N0=5000",
    "--param", "rho=0.5",
    "--param", "k=20",
    "--param", "tau=30",
];

fn pfilter_loglik(camdl: &Path, model: &Path, data: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "500", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.extend_from_slice(extra);
    run(camdl, &args)
}

fn parse_loglik(out: &std::process::Output) -> f64 {
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik in output:\nSTDOUT:{stdout}\nSTDERR:{}",
            String::from_utf8_lossy(&out.stderr)))
}

/// gh#621: pfilter runs the same W329 wide-first-window enforcer as `fit run`,
/// and this fixture's first window is 40 days against a ~5-day cadence. The
/// loglik-comparison tests below declare the SAME window on both sides — they
/// are about dated-vs-numeric equivalence, not conditioning, and this also
/// pins that the flag treats dated and numeric time columns identically.
/// Tests whose subject is a DIFFERENT error (empty file, missing origin) must
/// NOT pass it, or they hit the conditioning resolver first.
const COND: [&str; 2] = ["--condition-from", "first_obs - 5 days"];

/// §9.4 byte-identity: a dated TSV yields the same pfilter loglik as the same
/// data hand-converted to day-numbers against the origin.
#[test]
fn dated_loglik_matches_numeric() {
    let camdl = camdl_bin();
    let tmp = tempdir("byteid");
    let model = model_with_origin(&tmp, "2020-02-28");

    // The data must land inside the seeded epidemic (the model seeds at
    // tau=30, so it predicts ~0 cases before day 33). Earlier this fixture
    // placed the data at days 2..23 — before seeding — where every count is
    // near-impossible, the particle filter's ESS collapses, and the loglik
    // is -inf. That made the byte-identity check *vacuous*: -inf == -inf
    // holds even if the date→day-number conversion were wrong. The gh#110
    // degeneracy watchdog (2026-05-26) then turned that silent -inf into a
    // hard PFDegenerate error, surfacing the latent problem. Placing the
    // data in the epidemic window (days 40..55) gives a finite, time-
    // sensitive loglik (~-53.1 at 500 particles), so the test now actually
    // exercises the conversion.
    //
    // origin = 2020-02-28. Dates → day-numbers:
    //   2020-04-08 → 40, 2020-04-13 → 45, 2020-04-18 → 50, 2020-04-23 → 55
    let dated = tmp.join("dated.tsv");
    std::fs::write(&dated,
        "time\tcases\n2020-04-08\t11\n2020-04-13\t75\n2020-04-18\t212\n2020-04-23\t73\n").unwrap();
    let numeric = tmp.join("numeric.tsv");
    std::fs::write(&numeric, "time\tcases\n40\t11\n45\t75\n50\t212\n55\t73\n").unwrap();

    let ll_dated = parse_loglik(&pfilter_loglik(&camdl, &model, &dated, &COND));
    let ll_numeric = parse_loglik(&pfilter_loglik(&camdl, &model, &numeric, &COND));
    assert_eq!(ll_dated, ll_numeric,
        "dated and hand-converted-numeric logliks must be bit-identical");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// An observation file with a header but no data rows must produce a
/// clean, actionable error — not a panic. Regression: a header-only TSV
/// built a `MultiStreamObsModel` with empty `obs_times`, and the bootstrap
/// filter then indexed `obs_times[0]` in `mean()`, panicking with
/// `index out of bounds: the len is 0 but the index is 0`.
#[test]
fn empty_observation_file_errors_cleanly() {
    let camdl = camdl_bin();
    let tmp = tempdir("emptyobs");
    let model = model_with_origin(&tmp, "2020-02-28");
    let empty = tmp.join("empty.tsv");
    std::fs::write(&empty, "time\tcases\n").unwrap(); // header only

    let out = pfilter_loglik(&camdl, &model, &empty, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(!out.status.success(),
        "pfilter on an empty observation file must fail, not succeed");
    // The clean error names the problem and is actionable.
    assert!(stderr.to_lowercase().contains("no observation"),
        "error should explain there are no observations; got:\n{stderr}");
    // And it must NOT be the old index-out-of-bounds panic.
    assert!(!stderr.contains("index out of bounds") && !stderr.to_lowercase().contains("panic"),
        "must be a clean error, not a panic; got:\n{stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.4 origin-missing: dated cells against a model with no origin → error.
#[test]
fn dated_without_origin_errors() {
    let camdl = camdl_bin();
    let tmp = tempdir("noorigin");
    // seed_timing.ir.json (committed) declares no origin.
    let model = seed_timing_ir();
    let dated = tmp.join("dated.tsv");
    std::fs::write(&dated, "time\tcases\n2020-03-01\t3\n2020-03-08\t40\n").unwrap();

    let out = pfilter_loglik(&camdl, &model, &dated, &[]);
    assert!(!out.status.success(), "must fail when dated data has no origin");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("origin"), "error should mention origin: {stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.4 mixed column: numeric + date in one column → hard error naming rows.
#[test]
fn mixed_column_errors() {
    let camdl = camdl_bin();
    let tmp = tempdir("mixed");
    let model = model_with_origin(&tmp, "2020-02-28");
    let mixed = tmp.join("mixed.tsv");
    std::fs::write(&mixed, "time\tcases\n2\t3\n2020-03-08\t40\n").unwrap();

    let out = pfilter_loglik(&camdl, &model, &mixed, &[]);
    assert!(!out.status.success(), "must fail on a mixed time column");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mixed"), "error should flag mixed column: {stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.4 distinct-substep collision: two distinct times within dt → error.
#[test]
fn distinct_substep_collision_errors() {
    let camdl = camdl_bin();
    let tmp = tempdir("collide");
    let model = model_with_origin(&tmp, "2020-02-28");
    // Numeric times 10.0 and 10.4 at dt=1 both round to step 10.
    let data = tmp.join("collide.tsv");
    std::fs::write(&data, "time\tcases\n10.0\t3\n10.4\t40\n20\t60\n").unwrap();

    let out = pfilter_loglik(&camdl, &model, &data, &[]);
    assert!(!out.status.success(), "must fail on a sub-dt observation collision");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("same integrator substep"),
        "error should flag the substep collision: {stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.4 `--time-format numeric` forbids date cells.
#[test]
fn time_format_numeric_forbids_dates() {
    let camdl = camdl_bin();
    let tmp = tempdir("fmtnum");
    let model = model_with_origin(&tmp, "2020-02-28");
    let dated = tmp.join("dated.tsv");
    std::fs::write(&dated, "time\tcases\n2020-03-01\t3\n2020-03-08\t40\n").unwrap();

    let out = pfilter_loglik(&camdl, &model, &dated, &["--time-format", "numeric"]);
    assert!(!out.status.success(), "--time-format numeric must reject date cells");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Write the seed_timing IR with `simulation.t_start` set to `c`.
fn model_with_t_start(dir: &Path, c: f64) -> PathBuf {
    let src = std::fs::read_to_string(seed_timing_ir()).unwrap();
    let injected = src.replacen("\"t_start\": 0.0,", &format!("\"t_start\": {c},"), 1);
    assert!(injected.contains(&format!("\"t_start\": {c},")), "t_start injection failed");
    let p = dir.join(format!("seed_timing_tstart_{c}.ir.json"));
    std::fs::write(&p, injected).unwrap();
    p
}

/// §9.0.1 shift-invariance (numeric engine): shifting `(t_start, data times,
/// the time-typed param tau)` together by `c` leaves the pfilter loglik
/// bit-identical. This is the property the dated loader relies on (a change
/// of origin is exactly such a shift). Includes a negative shift (origin
/// after the first obs → negative internal times).
#[test]
fn numeric_shift_invariance() {
    let camdl = camdl_bin();
    let tmp = tempdir("shift");

    // Baseline data at tau=30: numeric daily cases (chosen to give a real
    // epidemic so the loglik is non-degenerate).
    let base_rows: &[(f64, i64)] =
        &[(20.0, 2), (30.0, 25), (40.0, 110), (50.0, 70), (60.0, 20)];

    let loglik_at_shift = |c: f64| -> f64 {
        let model = model_with_t_start(&tmp, c);
        let data = tmp.join(format!("shift_{c}.tsv"));
        let mut s = String::from("time\tcases\n");
        for (t, v) in base_rows {
            s.push_str(&format!("{}\t{}\n", t + c, v));
        }
        std::fs::write(&data, s).unwrap();
        let tau = format!("tau={}", 30.0 + c);
        // All BASE_PARAMS except tau, then the shifted tau.
        let params: Vec<&str> = BASE_PARAMS
            .iter()
            .copied()
            .take(BASE_PARAMS.len() - 2) // drop the trailing "--param", "tau=30"
            .collect();
        let mut args = vec![
            "pfilter", model.to_str().unwrap(),
            "--particles", "1000", "--dt", "1", "--seed", "7",
            "--data", data.to_str().unwrap(),
        ];
        args.extend_from_slice(&params);
        args.push("--param");
        args.push(&tau);
        let out = run(&camdl, &args);
        assert!(out.status.success(), "pfilter (shift {c}) failed: {}",
            String::from_utf8_lossy(&out.stderr));
        parse_loglik(&out)
    };

    let base = loglik_at_shift(0.0);
    for c in [-20.0, -11.0, 11.0, 20.0] {
        let ll = loglik_at_shift(c);
        assert_eq!(ll, base,
            "loglik must be shift-invariant: c={c} gave {ll}, c=0 gave {base}");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.8 output rendering: `--dates` adds a calendar `date` column to obs
/// output, rendered from the model origin; the numeric `time` column is
/// unchanged (byte-identical to a run without `--dates`).
#[test]
fn dates_flag_adds_calendar_column() {
    let camdl = camdl_bin();
    let tmp = tempdir("datesout");
    let model = model_with_origin(&tmp, "2020-02-28");

    let common: Vec<&str> = {
        let mut v = vec![
            "simulate", model.to_str().unwrap(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "7",
        ];
        v.extend_from_slice(BASE_PARAMS);
        v
    };

    // Without --dates.
    let plain = tmp.join("plain.tsv");
    let mut a1 = common.clone();
    a1.extend(["--obs-only", plain.to_str().unwrap()]);
    assert!(run(&camdl, &a1).status.success());

    // With --dates.
    let dated = tmp.join("dated.tsv");
    let mut a2 = common.clone();
    a2.extend(["--dates", "--obs-only", dated.to_str().unwrap()]);
    assert!(run(&camdl, &a2).status.success());

    let plain_txt = std::fs::read_to_string(&plain).unwrap();
    let dated_txt = std::fs::read_to_string(&dated).unwrap();

    // Header gains a `date` column.
    let dated_hdr = dated_txt.lines().next().unwrap();
    assert!(dated_hdr.starts_with("time\tdate\t"), "header: {dated_hdr}");
    assert!(plain_txt.lines().next().unwrap().starts_with("time\t"));

    // t=0 → origin date; the numeric `time` column matches the plain run.
    let plain_times: Vec<&str> = plain_txt.lines().skip(1)
        .map(|l| l.split('\t').next().unwrap()).collect();
    let dated_rows: Vec<Vec<&str>> = dated_txt.lines().skip(1)
        .map(|l| l.split('\t').collect()).collect();
    assert_eq!(dated_rows[0][1], "2020-02-28", "t=0 renders to origin");
    assert_eq!(dated_rows[1][1], "2020-02-29", "t=1 is the leap day");
    let dated_times: Vec<&str> = dated_rows.iter().map(|r| r[0]).collect();
    assert_eq!(plain_times, dated_times, "numeric time column must be unchanged");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.8 `--dates` with no origin → clear error.
#[test]
fn dates_flag_requires_origin() {
    let camdl = camdl_bin();
    let tmp = tempdir("datesnoorigin");
    let ir = seed_timing_ir();
    let mut args = vec![
        "simulate", ir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "7", "--dates",
    ];
    args.extend_from_slice(BASE_PARAMS);
    let out = tmp.join("o.tsv");
    args.extend(["--obs-only", out.to_str().unwrap()]);
    let o = run(&camdl, &args);
    assert!(!o.status.success(), "--dates without origin must fail");
    assert!(String::from_utf8_lossy(&o.stderr).contains("origin"),
        "error should mention origin");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §7 backward-compat: a run *without* `--dates` is byte-identical to the
/// pre-feature behaviour — no `date` column anywhere.
#[test]
fn no_dates_flag_is_unchanged() {
    let camdl = camdl_bin();
    let tmp = tempdir("nodates");
    let model = model_with_origin(&tmp, "2020-02-28");
    let mut args = vec![
        "simulate", model.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "7",
    ];
    args.extend_from_slice(BASE_PARAMS);
    let out = tmp.join("o.tsv");
    args.extend(["--obs-only", out.to_str().unwrap()]);
    assert!(run(&camdl, &args).status.success());
    let txt = std::fs::read_to_string(&out).unwrap();
    assert!(!txt.contains("date"), "no --dates → no date column even with origin set");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// §9.7 multi-timezone civil-date alignment: a TSV whose cells carry mixed
/// trailing offsets loads to the *same* internal time as the offset-stripped
/// sibling — the offset is discarded, every row maps to the same civil date.
#[test]
fn multitz_offsets_collapse_to_civil_date() {
    let camdl = camdl_bin();
    let tmp = tempdir("multitz");
    let model = model_with_origin(&tmp, "2020-02-28");

    // The dates land in the seeded epidemic window (the model seeds at
    // tau=30), so the filter is non-degenerate and the loglik is finite
    // AND date-sensitive. Earlier this fixture put all five rows on one
    // *pre-seeding* civil date (2020-03-15): the filter degenerated to
    // -inf, and comparing two -inf values is vacuous — it would pass even
    // if the offsets were parsed wrongly. (gh#110's watchdog later turned
    // that silent -inf into a hard error, surfacing the latent problem.)
    //
    // Each row carries a different tz-offset form (+01:00, +06:00, -03:00,
    // Z, +05:45). A daily-cadence loader must DROP the offset and keep the
    // civil date, so the offset-bearing file and the naive file resolve to
    // the same internal times and the same loglik. origin = 2020-02-28;
    // day-numbers: 2020-04-06→38, -11→43, -16→48, -21→53, -26→58.
    let tz = tmp.join("tz.tsv");
    std::fs::write(&tz,
        "time\tcases\n\
         2020-04-06+01:00\t4\n\
         2020-04-11+06:00\t64\n\
         2020-04-16-03:00\t174\n\
         2020-04-21Z\t144\n\
         2020-04-26+05:45\t75\n").unwrap();
    let naive = tmp.join("naive.tsv");
    std::fs::write(&naive,
        "time\tcases\n2020-04-06\t4\n2020-04-11\t64\n2020-04-16\t174\n\
         2020-04-21\t144\n2020-04-26\t75\n").unwrap();
    // Negative control: one civil date shifted a single day. Used only to
    // prove the loglik genuinely depends on the parsed dates — so the
    // tz == naive equality below is meaningful, not a vacuous -inf == -inf.
    let shifted = tmp.join("shifted.tsv");
    std::fs::write(&shifted,
        "time\tcases\n2020-04-07\t4\n2020-04-11\t64\n2020-04-16\t174\n\
         2020-04-21\t144\n2020-04-26\t75\n").unwrap();

    let ll_tz = parse_loglik(&pfilter_loglik(&camdl, &model, &tz, &COND));
    let ll_naive = parse_loglik(&pfilter_loglik(&camdl, &model, &naive, &COND));
    let ll_shifted = parse_loglik(&pfilter_loglik(&camdl, &model, &shifted, &COND));

    // Anti-vacuity guard 1: the loglik must be finite, else the equality
    // below proves nothing (any two -inf are equal).
    assert!(ll_naive.is_finite(),
        "loglik must be finite for the offset-collapse check to be \
         meaningful; got {ll_naive}");
    // The property under test: tz offsets are dropped, civil dates kept, so
    // the offset-bearing data scores identically to the naive sibling.
    assert_eq!(ll_tz, ll_naive,
        "offset-bearing dates must collapse to the same civil date as the \
         naive sibling");
    // Anti-vacuity guard 2: shifting a date by one day *does* change the
    // loglik — so the equality above is a real test of date resolution.
    assert_ne!(ll_shifted, ll_naive,
        "a one-day date shift must change the loglik; if it doesn't, the \
         loglik is insensitive to dates and the equality above is vacuous");

    let _ = std::fs::remove_dir_all(&tmp);
}
