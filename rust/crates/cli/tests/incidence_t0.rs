//! gh#174 — a positive incidence observation at model time 0 must fail with a
//! clear, named convention diagnostic, NOT a silent `-Inf` / bare PFDegenerate.
//!
//! Incidence over `[t-1, t]` is undefined at the model origin: the first
//! observation window `[t_start, obs_times[0]]` is zero-width, so the flow
//! accumulator is 0 and a positive count scores `-Inf` under the likelihood.
//! Pre-fix, this surfaced as a degenerate particle filter (loglik `-inf` or a
//! `PFDegenerate` bail), indistinguishable from a genuinely bad parameter
//! point — a non-expert would discard valid params. The fix is a hard error,
//! emitted before the filter runs, that names the convention and the remedies.
//!
//! The model is the committed `seed_timing.ir.json`, whose `cases` observation
//! is an incidence projection (`cumulative_flow: infection`) on a regular
//! schedule starting at 0.0 — so a data file with a positive count at time 0
//! exercises exactly the degenerate origin window.
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
    pfilter_with(camdl, model, data, &[])
}

/// gh#621: pfilter now runs the same W329 wide-first-window enforcer as
/// `fit run`. The degenerate-origin test below must still reach its OWN
/// (gh#174) error, so conditioning is opt-in per call rather than baked into
/// the shared helper.
fn pfilter_with(camdl: &Path, model: &Path, data: &Path, extra: &[&str])
    -> std::process::Output
{
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "500", "--dt", "1", "--seed", "5",
        "--data", data.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    args.extend_from_slice(BASE_PARAMS);
    Command::new(camdl)
        .args(&args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// A positive incidence count at model time 0 must produce a hard, named
/// convention error — not a silent `-Inf` loglik and not a generic
/// `PFDegenerate` bail (which looks like a bad parameter point).
#[test]
fn positive_incidence_at_origin_is_named_error() {
    let camdl = camdl_bin();
    let tmp = tempdir("posinc");
    let model = seed_timing_ir();

    // First row at time 0 (= t_start = the model origin) with a positive
    // count; remaining rows land in the seeded epidemic (the model seeds at
    // tau=30) so the *only* degeneracy is the t=0 window.
    let data = tmp.join("t0.tsv");
    std::fs::write(&data, "time\tcases\n0\t11\n40\t11\n45\t75\n50\t212\n55\t73\n").unwrap();

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
    // The error must name the t=0 incidence convention so the user knows the
    // fix is data alignment, not a bad parameter point.
    let combined = format!("{stdout}\n{stderr}").to_lowercase();
    assert!(
        combined.contains("incidence") && combined.contains("time 0"),
        "error must name the t=0 incidence convention; got:\n{stderr}"
    );
    // And it must NOT masquerade as a particle-filter degeneracy.
    assert!(
        !combined.contains("pfdegenerate") && !combined.contains("esscollapsed"),
        "the t=0 incidence convention must be caught BEFORE the filter runs, \
         not surfaced as a PFDegenerate bail; got:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Control: the same data with the degenerate t=0 row dropped scores a finite
/// loglik. This pins that the error above is specifically about the origin
/// window, not the data or parameters.
#[test]
fn dropping_origin_row_scores_finite() {
    let camdl = camdl_bin();
    let tmp = tempdir("drop0");
    let model = seed_timing_ir();

    let data = tmp.join("drop0.tsv");
    std::fs::write(&data, "time\tcases\n40\t11\n45\t75\n50\t212\n55\t73\n").unwrap();

    // The 40-day first window needs a declared conditioning window (gh#621);
    // the SUBJECT of this test is that dropping the t=0 row scores finite.
    let out = pfilter_with(&camdl, &model, &data, &["--condition-from", "first_obs - 5 days"]);
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
