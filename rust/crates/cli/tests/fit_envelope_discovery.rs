//! `camdl show` and `camdl cat` on a stored fit run named by its label and on
//! the prediction stored inside it. After `fit run --label foo` and
//! `camdl predict @foo`, `camdl show @foo` lists the run's result files,
//! `camdl show <prediction folder>` lists the prediction's, and
//! `camdl cat <handle> <name>` prints one of them, by its logical name
//! (`predictive/weekly_cases`) or its stored name.

mod fit_store_support;

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
    covers        = closing_at(time, 7 'days)
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

fn fit_toml(algorithm_block: &str, store: &str) -> String {
    format!(
        r#"store = "{store}"

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
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent). Mirrors the runbook's ad-hoc guidance.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        // The store `fit.toml` names, for the commands that read it.
        .env("CAMDL_STORE", dir.join("results"))
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

    let pgas = r#"[method]
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
        // Default options: their outputs are the bare paths `show`/`cat`
        // name (gh#896).
        &["predict", "@foo"],
    );
    assert!(
        out.status.success(),
        "predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The prediction `camdl predict` stored, a result of its own inside the run.
    let predictions = fit_store_support::predictions(&tmp.join("results"));
    assert_eq!(predictions.len(), 1, "one stored prediction: {predictions:?}");
    let prediction = predictions[0].to_string_lossy().into_owned();

    // ── (1) `camdl show @foo` lists the run's result files ──
    let out = run(&bin, &tmp, &["show", "@foo", "--store", "results"]);
    assert!(
        out.status.success(),
        "show @foo failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let show = String::from_utf8_lossy(&out.stdout);
    let fields = show_fields(&show);
    assert_eq!(fields.get("kind").map(String::as_str), Some("fit_run"), "kind header; got:\n{show}");
    assert_eq!(fields.get("method").map(String::as_str), Some("pgas"), "method header; got:\n{show}");
    for f in ["draws.tsv", "fit_state.toml", "chain_0/trace.tsv"] {
        assert!(show.lines().any(|l| l.trim() == f), "show @foo lists {f}; got:\n{show}");
    }

    // ── (2) `camdl show <prediction>` lists its predictive and quantities
    // outputs ──
    let out = run(&bin, &tmp, &["show", &prediction, "--store", "results"]);
    assert!(
        out.status.success(),
        "show <prediction> failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let show_pred = String::from_utf8_lossy(&out.stdout);
    assert_eq!(show_fields(&show_pred).get("kind").map(String::as_str), Some("prediction"),
        "kind header; got:\n{show_pred}");
    assert!(
        show_pred.contains("predictive/weekly_cases.tsv"),
        "show must list a predictive/ output path; got:\n{show_pred}"
    );
    assert!(
        show_pred.contains("quantities-model/peak.tsv"),
        "show must list a quantities-model/ output path; got:\n{show_pred}"
    );
    // gh#892: and the file that says whether that predictive is complete.
    // `camdl predict` writes `report.json` on every run — the deterministic
    // failures, empty when nothing refused — so a listing that omits it sends
    // a reader to the band without the record of what did not make it.
    assert!(
        show_pred.contains("report.json"),
        "show must list the predictive's report.json beside it; got:\n{show_pred}"
    );

    // ── (3) `camdl cat <prediction> predictive/weekly_cases` → TSV
    // with q50, named by its logical name ──
    let out = run(
        &bin,
        &tmp,
        &["cat", &prediction, "predictive/weekly_cases", "--store", "results"],
    );
    assert!(
        out.status.success(),
        "cat predictive stream failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let pred = String::from_utf8_lossy(&out.stdout);
    let header = pred.lines().next().unwrap_or("");
    assert!(
        header.split('\t').any(|c| c == "q50"),
        "predictive TSV header carries the q50 band column; got header: {header:?}"
    );

    // ── (4) `camdl cat <prediction> quantities-model/peak.tsv` →
    // non-empty ──
    let out = run(
        &bin,
        &tmp,
        &["cat", &prediction, "quantities-model/peak.tsv", "--store", "results"],
    );
    assert!(
        out.status.success(),
        "cat quantities-model/peak.tsv failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty(), "quantities-model/peak.tsv must be non-empty");

    // ── (5) `@foo` names a run: no name gives its `fit_state.toml` ──
    let out = run(&bin, &tmp, &["cat", "@foo", "--store", "results"]);
    assert!(
        out.status.success(),
        "cat @foo (default) failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let state = String::from_utf8_lossy(&out.stdout);
    assert!(state.contains("method"), "default cat emits the run's fit_state.toml; got:\n{state}");

    // ── (6) a missing stream is refused, listing the files the run holds ──
    let out = run(
        &bin,
        &tmp,
        &["cat", "@foo", "predictive/nope", "--store", "results"],
    );
    assert!(!out.status.success(), "a missing stream must error, not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("has `predictive/nope`") && stderr.contains("fit_state"),
        "the refusal lists the files there are; got: {stderr}"
    );

    // ── (7) the label the run was named by is surfaced ──
    assert!(show.contains("foo"), "label is surfaced; got:\n{show}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The `name → value` pairs of `camdl show`'s text output: each field's name
/// on one line and its value, indented, on the next.
fn show_fields(show: &str) -> std::collections::BTreeMap<String, String> {
    let lines: Vec<&str> = show.lines().collect();
    lines.windows(2)
        .filter(|w| !w[0].starts_with(' ') && w[1].starts_with("  "))
        .map(|w| (w[0].trim().to_string(), w[1].trim().to_string()))
        .collect()
}
