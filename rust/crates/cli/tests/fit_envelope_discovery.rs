//! End-to-end acceptance for the fit OUTPUT ENVELOPE projection in
//! `camdl show` / `camdl cat`: a completed fit (a CAS segment with no fit-wide
//! `run.json`) is discoverable by a fit handle. After `fit run` + `fit predict`,
//! `camdl show @label` lists the discoverable output files, and `camdl cat
//! @label --stream <rel>` streams an individual output (with the `.tsv`
//! convenience for an extensionless stream).
//!
//! Phase 1c of
//! docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation + one generated quantity.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  peak = max(I / N)
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

/// A short observed weekly series (rise-and-fall), times on the weekly grid.
const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(algorithm_block: &str, output_dir: &str) -> String {
    format!(
        r#"output_dir = "{output_dir}"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

{algorithm_block}
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent). Mirrors the runbook's ad-hoc guidance.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

#[test]
fn fit_envelope_show_and_cat_by_label() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_fit_envelope_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    let pgas = r#"[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(pgas, "results")).unwrap();

    // Run the fit with a label, then predict (free-forward keeps it fast).
    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--label", "foo", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"],
    );
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // ── (1) `camdl show @foo` lists predictive/ and quantities/ outputs ──
    let out = run(&bin, &tmp, &["show", "@foo", "--root", "results"]);
    assert!(
        out.status.success(),
        "show @foo failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let show = String::from_utf8_lossy(&out.stdout);
    assert!(
        show.contains("predictive/weekly_cases.tsv"),
        "show envelope must list a predictive/ output path; got:\n{show}"
    );
    assert!(
        show.contains("quantities/peak.tsv"),
        "show envelope must list a quantities/ output path; got:\n{show}"
    );
    // The header fields the proposal specifies are present.
    assert!(show.contains("kind") && show.contains("fit"), "kind=fit header; got:\n{show}");
    assert!(show.contains("foo"), "label is surfaced; got:\n{show}");

    // ── (2) `camdl cat @foo --stream predictive/weekly_cases` → TSV w/ q50 ──
    // Extensionless stream resolves the `.tsv` sibling.
    let out = run(
        &bin,
        &tmp,
        &["cat", "@foo", "--stream", "predictive/weekly_cases", "--root", "results"],
    );
    assert!(
        out.status.success(),
        "cat @foo predictive stream failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let pred = String::from_utf8_lossy(&out.stdout);
    let header = pred.lines().next().unwrap_or("");
    assert!(
        header.split('\t').any(|c| c == "q50"),
        "predictive TSV header carries the q50 band column; got header: {header:?}"
    );

    // ── (3) `camdl cat @foo --stream quantities/peak.tsv` → non-empty ──
    let out = run(
        &bin,
        &tmp,
        &["cat", "@foo", "--stream", "quantities/peak.tsv", "--root", "results"],
    );
    assert!(
        out.status.success(),
        "cat @foo quantities/peak.tsv failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.stdout.is_empty(),
        "quantities/peak.tsv must be non-empty"
    );

    // ── (4) no `--stream` defaults to the fit.meta.json summary record ──
    let out = run(&bin, &tmp, &["cat", "@foo", "--root", "results"]);
    assert!(
        out.status.success(),
        "cat @foo (default) failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta = String::from_utf8_lossy(&out.stdout);
    assert!(
        meta.contains("\"label\"") && meta.contains("foo"),
        "default cat emits the fit.meta.json sidecar; got:\n{meta}"
    );

    // ── (5) a missing stream errors actionably (suggests `camdl show`) ──
    let out = run(
        &bin,
        &tmp,
        &["cat", "@foo", "--stream", "predictive/nope", "--root", "results"],
    );
    assert!(!out.status.success(), "a missing stream must error, not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("camdl show"),
        "miss points the user at `camdl show`; got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
