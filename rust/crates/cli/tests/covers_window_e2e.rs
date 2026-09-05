//! End-to-end: a declared observation window (gh#833) actually moves the
//! scoring boundary, through the real CLI data path.
//!
//! The claim this pins is the one the whole arc exists to make. An UNDECLARED
//! incidence stream scores row `k` over `(t[k-1], t[k]]` — a window nobody
//! wrote down. `covers = day(time)` says row `D` covers `[D, D+1)`, so it is
//! scored one bucket LATER. Stated as an equivalence, which is stronger than
//! "the number changed" and does not depend on any particular fixture's
//! sensitivity:
//!
//!   declared `day(time)` on a file       ==       undeclared on the same file
//!   labelled 0,1,2,…                              with every label shifted +1
//!
//! Both sides then close their bins at 1,2,3,…, so the logliks must agree
//! EXACTLY, not approximately.
//!
//! `covers` is injected into the committed IR rather than compiled from a
//! `.camdl`, so this test does not depend on the OCaml compiler being on PATH
//! (mirroring `dated_data_loader.rs`).

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

/// The committed fixture, unchanged — an incidence stream that declares
/// nothing about what its rows cover.
fn undeclared_model(dir: &Path) -> PathBuf {
    let p = dir.join("undeclared.ir.json");
    std::fs::copy(seed_timing_ir(), &p).unwrap();
    p
}

/// The same fixture with `covers = day(time)` injected, in its lowered form:
/// open at the row's own label, one day wide.
fn declared_model(dir: &Path) -> PathBuf {
    let src = std::fs::read_to_string(seed_timing_ir()).unwrap();
    let injected = src.replacen(
        "\"projection\":",
        "\"covers\":{\"kind\":\"from\",\"offset\":0.0,\"span\":1.0},\"projection\":",
        1,
    );
    assert!(injected.contains("\"covers\""), "covers injection failed");
    let p = dir.join("declared.ir.json");
    std::fs::write(&p, injected).unwrap();
    p
}

/// A daily counts file over `times`, with a bump so the series is not all
/// zeros (an all-zero series would agree under any window and prove nothing).
fn write_counts(path: &Path, shift: f64) {
    let mut s = String::from("time\tcases\n");
    for k in 0..40u32 {
        let v = if k >= 20 { (k - 19) * 3 } else { 0 };
        s.push_str(&format!("{}\t{}\n", 3.0 + f64::from(k) + shift, v));
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

fn pfilter_loglik(camdl: &Path, model: &Path, data: &Path, extra: &[&str]) -> f64 {
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "500", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    args.extend_from_slice(extra);
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

/// The complete meaning of `covers = day(time)` on a file labelled 3,4,5,…
/// with `t_start = 0`, stated as an equivalence with the undeclared machinery:
///
///   declared day(time)  ==  undeclared, labels shifted +1, AND `condition_from`
///                           opening the first bin at 3
///
/// Both then reset at 3 and close bins at 4,5,6,…, so the numbers agree
/// EXACTLY. This pins two things at once: the one-bucket SHIFT (row D closes at
/// D+1) and the LEADING EDGE (the first period opens where it says, at 3, not
/// at `t_start`). A declaration is the thing that makes `condition_from`
/// unnecessary — it states where scoring begins.
#[test]
fn a_declared_day_window_scores_one_bucket_later_and_opens_at_its_own_start() {
    let camdl = camdl_bin();
    let tmp = tempdir("shift");

    let plain = tmp.join("counts.tsv");
    let shifted = tmp.join("counts_shifted.tsv");
    write_counts(&plain, 0.0); // labels 3..42
    write_counts(&shifted, 1.0); // labels 4..43

    let declared = pfilter_loglik(&camdl, &declared_model(&tmp), &plain, &[]);
    // The shifted file's first observation is 4; opening one day before it
    // puts the reset-only boundary at 3 — exactly where the declared stream's
    // first period opens.
    let undeclared_equiv = pfilter_loglik(
        &camdl, &undeclared_model(&tmp), &shifted,
        &["--condition-from", "first_obs - 1 days"],
    );
    assert_eq!(
        declared, undeclared_equiv,
        "a declared day-window must score exactly as the undeclared reading of \
         the same data shifted one day later with its first bin opened at the \
         declared start (declared={declared}, equivalent={undeclared_equiv})",
    );

    // Non-vacuous, twice over. (1) The undeclared reading of the UNSHIFTED file
    // is a different scoring, so this is not "any two runs agree".
    let undeclared_plain = pfilter_loglik(&camdl, &undeclared_model(&tmp), &plain, &[]);
    assert_ne!(
        declared, undeclared_plain,
        "declaring a window must change the scoring of the same file; if these \
         agree the declaration is not reaching the filter",
    );
    // (2) The shifted file WITHOUT conditioning opens its first bin at t_start,
    // so it must NOT match: if it does, the declared stream is also opening at
    // t_start — the leading-window defect — rather than at its declared start.
    let undeclared_shifted_wide = pfilter_loglik(&camdl, &undeclared_model(&tmp), &shifted, &[]);
    assert_ne!(
        declared, undeclared_shifted_wide,
        "the declared stream's first period must open at its own start (3), not \
         at t_start; agreeing with the un-conditioned shifted reading means the \
         first bin spans the whole warm-up",
    );
}

#[test]
fn condition_from_on_a_declared_stream_is_refused() {
    // Conditioning opens the first scored bin at a boundary of the USER's
    // choosing, by prepending a reset-only row. On a stream that has stated
    // what its first row covers, that silently truncates the stated window —
    // two sources disagreeing about one period. Refuse, naming both.
    let camdl = camdl_bin();
    let tmp = tempdir("cond");
    let plain = tmp.join("counts.tsv");
    write_counts(&plain, 0.0);

    let model = declared_model(&tmp);
    let mut args: Vec<String> = vec![
        "pfilter".into(), model.to_str().unwrap().into(),
        "--particles".into(), "50".into(), "--dt".into(), "1".into(),
        "--seed".into(), "5".into(),
        "--data".into(), plain.to_str().unwrap().into(),
        "--condition-from".into(), "first_obs - 2 days".into(),
    ];
    args.extend(BASE_PARAMS.iter().map(|s| s.to_string()));
    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
    let out = run(&camdl, &borrowed);

    assert!(!out.status.success(), "a declared stream + condition_from must fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("declares what each of its rows covers")
            && err.contains("silently truncated"),
        "the refusal must explain the conflict, got: {err}",
    );
}
