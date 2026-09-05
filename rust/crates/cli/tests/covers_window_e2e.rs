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
        "\"covers\":{\"kind\":\"from\",\"offset\":0.0,\"span\":{\"const\":1.0}},\"projection\":",
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

#[test]
fn a_declared_day_window_scores_one_bucket_later() {
    let camdl = camdl_bin();
    let tmp = tempdir("shift");

    let plain = tmp.join("counts.tsv");
    let shifted = tmp.join("counts_shifted.tsv");
    write_counts(&plain, 0.0);
    write_counts(&shifted, 1.0);

    let declared = pfilter_loglik(&camdl, &declared_model(&tmp), &plain);
    let undeclared_shifted = pfilter_loglik(&camdl, &undeclared_model(&tmp), &shifted);

    // The equivalence: declaring `day(time)` puts row D in the bucket closing
    // at D+1, which is exactly where an undeclared stream puts a row LABELLED
    // D+1. Same bins, same values, so the same number — exactly.
    assert_eq!(
        declared, undeclared_shifted,
        "a declared day-window must score exactly as the undeclared reading of \
         the same data shifted one day later (declared={declared}, \
         undeclared-on-shifted={undeclared_shifted})",
    );

    // Non-vacuous: the undeclared reading of the UNSHIFTED file is a different
    // scoring, so the equivalence above is not just "any two runs agree".
    let undeclared_plain = pfilter_loglik(&camdl, &undeclared_model(&tmp), &plain);
    assert_ne!(
        declared, undeclared_plain,
        "declaring a window must change the scoring of the same file; if these \
         agree the declaration is not reaching the filter",
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
