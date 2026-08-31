//! End-to-end acceptance for `camdl fit predict --sweep` — the Phase-3 parameter
//! sweep over the posterior, composing with `--scenario`.
//!
//! A sweep varies one parameter across a grid over the posterior cloud: each
//! sweep cell OVERRIDES the swept parameter in every posterior draw (the draw
//! supplies the rest), so the swept value rides in the same draw/sweep tier as
//! the draw — the resolver still lets a `--scenario` win over it. The cells are
//! keyed by a leading `sweep:<param>` column.
//!
//! Three checks:
//!   (a) compose: `--scenario` on `rho` + `--sweep k=…` (a DISTINCT parameter)
//!       stack into one file, each row tagged with both columns;
//!   (b) collision: a scenario and a sweep on the SAME parameter is a hard error;
//!   (c) no-sweep: the predictive header is byte-identical to the pre-sweep shape.
//!
//! Phase 3 of
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

/// A closed SIR with a weekly NegBinomial observation and two `rho` scenarios.
/// `k` (the NB dispersion) is a real, fixed model parameter distinct from `rho`,
/// so it can be swept while a scenario pins `rho`.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
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

scenarios {
  low_rho  { set = { rho = 0.3 } }
  high_rho { set = { rho = 0.8 } }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

/// A short observed weekly series (rise-and-fall), times on the weekly grid.
const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// `k` is fixed in the fit (the swept parameter must carry a value per draw, and
/// fixed parameters appear in every draw row exactly like estimated ones).
fn fit_toml(output_dir: &str) -> String {
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

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    let fits = root.join("fits");
    let entries = std::fs::read_dir(&fits).ok()?;
    for e in entries.flatten() {
        let p = e.path().join(sub).join(format!("{stream}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Run the shared PGAS fit once into `tmp/results` and return `results`.
fn setup_fit(bin: &Path, tmp: &Path) -> PathBuf {
    let _ = std::fs::remove_dir_all(tmp);
    std::fs::create_dir_all(tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml("results")).unwrap();

    let out = run(bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    tmp.join("results")
}

#[test]
fn fit_predict_sweep_composes_with_scenario_on_distinct_params() {
    // (a) `--scenario low_rho` (pins rho) + `--sweep k=8,12` (varies k, a DISTINCT
    // parameter): the predictive file leads `scenario  sweep:k  time …`, and rows
    // exist for both (low_rho, 8) and (low_rho, 12). A quantity file carries the
    // sweep:k column too.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_sweep_compose_{}", std::process::id()));
    let results = setup_fit(&bin, &tmp);

    let out = run(&bin, &tmp, &[
        "fit", "predict", "--fit", "fit.toml",
        "--horizon", "free_forward",
        "--scenario", "low_rho",
        "--sweep", "k=8,12",
        "--seed", "1",
    ]);
    assert!(
        out.status.success(),
        "compose fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // ── predictive/weekly_cases.tsv: scenario then sweep:k lead the header ──
    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let header = pred_txt.lines().next().unwrap();
    assert_eq!(
        header,
        "scenario\tsweep:k\ttime\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "the sweep:k column follows the scenario column"
    );

    // Group the free-forward rows by (scenario, sweep:k): both grid cells present.
    let cells: std::collections::BTreeSet<(String, String)> = pred_txt
        .lines()
        .skip(1)
        .filter(|l| l.split('\t').nth(3) == Some("free_forward"))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), f[1].to_string())
        })
        .collect();
    assert!(
        cells.contains(&("low_rho".to_string(), "8".to_string())),
        "the (low_rho, k=8) cell is present; saw {cells:?}"
    );
    assert!(
        cells.contains(&("low_rho".to_string(), "12".to_string())),
        "the (low_rho, k=12) cell is present; saw {cells:?}"
    );
    // gh#625: the fitted no-overlay arm is ALWAYS emitted (it is the
    // posterior predictive every scenario overlays); no OTHER scenario leaks
    // in, which is what this assertion is for.
    assert!(
        cells.iter().all(|(s, _)| s == "low_rho" || s == "fitted"),
        "only the requested scenario (plus the fitted reference) is present; \
         saw {cells:?}"
    );
    assert!(
        cells.iter().any(|(s, _)| s == "fitted"),
        "the fitted reference arm is present (gh#625); saw {cells:?}"
    );

    // ── quantities/peak.tsv: the sweep:k column is carried on the quantity too ──
    let peakf = find_artifact(&results, "quantities", "peak")
        .expect("quantities/peak.tsv must be written");
    let peak_txt = std::fs::read_to_string(&peakf).unwrap();
    let qheader = peak_txt.lines().next().unwrap();
    assert_eq!(
        qheader,
        "scenario\tsweep:k\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
        "the quantity header carries the sweep:k column after scenario"
    );
    let qcells: std::collections::BTreeSet<String> = peak_txt
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(1).unwrap().to_string())
        .collect();
    assert_eq!(
        qcells,
        ["8".to_string(), "12".to_string()].into_iter().collect(),
        "one peak row per sweep cell, each tagged with its k value; saw {qcells:?}"
    );

    // ── quantities.json carries the calendar block ──────────────────────────
    // `fit predict` used to build its merged manifest with `schema` +
    // `quantities` only, so a consumer could not map the numeric `time` column
    // to dates without re-parsing the model — the one thing the block exists to
    // prevent, and it bites hardest on dated outbreak work. `simulate`'s
    // manifest always carried it; predict's two OTHER sidecars did too. Only
    // this one dropped it.
    let qmanf = peakf
        .parent()
        .and_then(|d| d.parent())
        .map(|seg| seg.join("quantities.json"))
        .expect("quantities.json sits beside the quantities/ dir");
    let qm: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&qmanf).unwrap()).unwrap();
    assert_eq!(qm["schema"], "camdl.quantities/v1", "manifest schema tag");
    assert_eq!(
        qm["calendar"]["time_unit"], "days",
        "the quantities manifest must carry calendar semantics for the time \
         column, got: {}",
        qm["calendar"]
    );

    // ── predictive.json: the sweep:k coordinate is named in the join contract ──
    let pmf = std::fs::read_dir(results.join("fits"))
        .unwrap()
        .flatten()
        .map(|e| e.path().join("predictive.json"))
        .find(|p| p.is_file())
        .expect("predictive.json must be written");
    let pjson: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pmf).unwrap()).unwrap();
    // Calendar semantics travel with the predictive artifact.
    assert!(
        pjson["calendar"]["time_unit"].is_string(),
        "predictive.json carries calendar semantics"
    );
    let wc = pjson["streams"].as_array().unwrap()
        .iter().find(|s| s["name"] == "weekly_cases").expect("weekly_cases stream entry");
    let coords: Vec<&str> = wc["coordinates"].as_array().unwrap()
        .iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(
        coords, ["scenario", "sweep:k", "time", "horizon", "treatment"],
        "the manifest names sweep:k as a coordinate, after scenario; saw {coords:?}"
    );

    // ── observed.json: the net-new sibling manifest carries calendar semantics ──
    let omf = pmf.parent().unwrap().join("observed.json");
    let ojson: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&omf).expect("observed.json must be written"),
    )
    .unwrap();
    assert_eq!(ojson["schema"], "camdl.observed/v1", "observed.json schema tag");
    assert!(
        ojson["calendar"]["time_unit"].is_string(),
        "observed.json carries calendar semantics"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_sweep_same_param_as_scenario_is_a_hard_error() {
    // (b) A scenario that sets `rho` AND `--sweep rho=…` is contradictory: the
    // scenario pins rho (winning over the sweep) while the sweep varies it. Must
    // exit non-zero with a message naming rho, the scenario, and "sweep".
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_sweep_collide_{}", std::process::id()));
    let _results = setup_fit(&bin, &tmp);

    let out = run(&bin, &tmp, &[
        "fit", "predict", "--fit", "fit.toml",
        "--horizon", "free_forward",
        "--scenario", "low_rho",
        "--sweep", "rho=0.2,0.4",
        "--seed", "1",
    ]);
    assert!(
        !out.status.success(),
        "a scenario and a sweep on the same parameter must be rejected, not run"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rho"),
        "the error names the colliding parameter rho; got: {stderr}"
    );
    assert!(
        stderr.contains("low_rho"),
        "the error names the scenario; got: {stderr}"
    );
    assert!(
        stderr.contains("sweep"),
        "the error names the sweep; got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_no_sweep_header_is_byte_identical() {
    // (c) With no `--sweep`, the predictive header is EXACTLY the pre-sweep shape
    // (no `sweep:` column) — the byte-identity guard for the no-sweep path.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_sweep_none_{}", std::process::id()));
    let results = setup_fit(&bin, &tmp);

    let out = run(&bin, &tmp, &[
        "fit", "predict", "--fit", "fit.toml",
        "--horizon", "free_forward",
    ]);
    assert!(
        out.status.success(),
        "no-sweep fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let header = pred_txt.lines().next().unwrap();
    assert_eq!(
        header,
        "scenario\ttime\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "no --sweep ⇒ no sweep: column ⇒ byte-identical header"
    );
    assert!(
        !header.contains("sweep:"),
        "no sweep column without --sweep; got: {header}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
