//! gh#174, restated under gh#833 — an incidence row whose period opens before
//! the run must fail with a clear, named error, NOT a silent `-Inf` or a bare
//! `PFDegenerate`.
//!
//! Before declarations existed, a row at the model origin was a zero-width
//! window: the accumulator held 0 and a positive count scored `-Inf`, which
//! surfaced as a degenerate particle filter indistinguishable from a bad
//! parameter point. The stream in `seed_timing.ir.json` now declares
//! `covers = closing_at(time, 1 'days)`, so a row labelled 0 covers `[-1, 0)`
//! — time the run never simulated — and the loader refuses it before the
//! filter runs, naming the period and `t_start`.
//!
//! Silent-skip if the release binary is not built (mirrors dated_data_loader).

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
    let p = std::env::temp_dir().join(format!("camdl_inc_t0_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
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

fn pfilter(camdl: &Path, model: &Path, data: &Path) -> std::process::Output {
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "500", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(BASE_PARAMS);
    Command::new(camdl)
        .args(&args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// A positive incidence count at model time 0 must produce a hard, named
/// error — not a silent `-Inf` loglik and not a generic `PFDegenerate` bail
/// (which looks like a bad parameter point).
#[test]
fn positive_incidence_at_origin_is_named_error() {
    let camdl = camdl_bin();
    let tmp = tempdir("posinc");
    let model = seed_timing_ir();

    // First row at time 0 (= t_start = the model origin) with a positive
    // count; the remaining rows are consecutive days, as the stream's
    // `closing_at(time, 1 'days)` declares, so the ONLY defect is the row
    // whose period `[-1, 0)` opens before the run.
    let data = tmp.join("t0.tsv");
    std::fs::write(&data, "time\tcases\n0\t11\n40\t11\n41\t75\n42\t212\n43\t73\n").unwrap();

    let out = pfilter(&camdl, &model, &data);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // It must FAIL, not print `-inf` and exit 0.
    assert!(
        !out.status.success(),
        "a positive incidence obs at time 0 must be a hard error, not a \
         silent success. stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // It must NOT silently emit a `-inf` loglik to stdout.
    assert!(
        !stdout.lines().any(|l| l.trim().eq_ignore_ascii_case("-inf")),
        "must not print a bare `-inf` loglik; got stdout:\n{stdout}"
    );
    // The error must name the period and the run's start, so the user knows
    // the fix is data alignment, not a bad parameter point.
    assert!(
        stderr.contains("opens at -1") && stderr.contains("t_start = 0"),
        "error must name the period that opens before the run and t_start; got:\n{stderr}"
    );
    // And it must NOT masquerade as a particle-filter degeneracy.
    let combined = format!("{stdout}\n{stderr}").to_lowercase();
    assert!(
        !combined.contains("pfdegenerate") && !combined.contains("esscollapsed"),
        "the refusal must come from the loader, BEFORE the filter runs, \
         not surface as a PFDegenerate bail; got:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// An `NA` at the origin is not an observation: its period `[-1, 0)` precedes
/// the run, but there is nothing to score there and no reset the run could
/// reach, so the loader drops the row rather than refusing the file. This is
/// the shape a wide synthetic file has when an instant stream sits beside an
/// incidence one (gh#833).
#[test]
fn a_hole_at_the_origin_is_dropped_not_refused() {
    let camdl = camdl_bin();
    let tmp = tempdir("hole0");
    let model = seed_timing_ir();

    let data = tmp.join("hole0.tsv");
    std::fs::write(&data, "time\tcases\n0\tNA\n40\t11\n41\t75\n42\t212\n43\t73\n").unwrap();

    let out = pfilter(&camdl, &model, &data);
    assert!(
        out.status.success(),
        "a hole whose period precedes the run must be dropped, not refused; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ll: f64 = stdout
        .lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik in stdout:\n{stdout}"));
    assert!(ll.is_finite(), "loglik must be finite; got {ll}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Control: the same data with the t=0 row dropped scores a finite loglik.
/// This pins that the error above is specifically about that row's period,
/// not the data or parameters. No conditioning window is needed: the first
/// declared period opens at 39, and the warm-up before it is simulated but
/// not scored.
#[test]
fn dropping_origin_row_scores_finite() {
    let camdl = camdl_bin();
    let tmp = tempdir("drop0");
    let model = seed_timing_ir();

    let data = tmp.join("drop0.tsv");
    std::fs::write(&data, "time\tcases\n40\t11\n41\t75\n42\t212\n43\t73\n").unwrap();

    let out = pfilter(&camdl, &model, &data);
    assert!(
        out.status.success(),
        "dropping the t=0 row must score cleanly; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ll: f64 = stdout
        .lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik in stdout:\n{stdout}"));
    assert!(ll.is_finite(), "loglik must be finite after dropping t=0; got {ll}");

    let _ = std::fs::remove_dir_all(&tmp);
}
