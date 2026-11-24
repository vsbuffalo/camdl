//! End-to-end acceptance for `camdl predict` — the free-forward posterior
//! predictive verb. Runs a real (tiny) PGAS fit, then `camdl predict`, and checks
//! the two tidy artifacts have the typed-axis columns the proposal specifies.
//! Also checks the safety property: an optimizer (IF2) fit is refused, never
//! silently turned into a band.
//!
//! Proposal: docs/dev/proposals/2026-06-22-predictive-ergonomics.md

use std::path::{Path, PathBuf};
use std::process::Command;

mod fit_store_support;

/// The prediction folder the latest `camdl predict` in the store `root` wrote:
/// of every `predict/<mode>-<id12>/` folder in it, the one whose
/// `report*.json` (one written by every predict) is newest. A path that does
/// not exist while nothing has been predicted.
fn predict_out(root: &Path) -> PathBuf {
    fit_store_support::predictions(root).into_iter()
        .filter_map(|d| {
            // Newest `report*.json` (keyed by chain selection) in the folder.
            let t = std::fs::read_dir(&d).ok()?.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.starts_with("report") && n.ends_with(".json")
                })
                .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
                .max()?;
            Some((t, d))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, d)| d)
        .unwrap_or_else(|| root.join("no-prediction"))
}

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

/// A closed SIR with a weekly NegBinomial observation — small and well-behaved.
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
  prevalence = I / N                  # series  (one value per snapshot)
  peak       = max(I / N)             # value scalar (no censoring)
  onset      = first_above(I / N, 0.01)   # time scalar (right-censorable)
  onset2     = first_above(I / N, 0.02)   # time scalar
  spread     = onset2 - onset             # Derived over Time scalars; censorable
  peak_obs   = max(observations.weekly_cases)   # v1.1: reduce the per-draw y_sim
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
        // The store every fit file here names, for the commands that read it.
        .env("CAMDL_STORE", dir.join("results"))
        .output()
        .expect("spawn camdl")
}

/// `<sub>/<stream>.tsv` of the latest prediction in the store `root`.
fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    let p = predict_out(root).join(sub).join(format!("{stream}.tsv"));
    p.is_file().then_some(p)
}

/// `file` of the latest prediction in the store `root` (`quantities.json`),
/// else of the one fit problem or compiled model stored there
/// (`model.ir.json`).
fn find_stored_file(root: &Path, file: &str) -> Option<PathBuf> {
    let mut places = vec![predict_out(root)];
    places.extend(fit_store_support::problems(root));
    places.extend(fit_store_support::folders_of(root, "compiled_model"));
    places.into_iter().map(|d| d.join(file)).find(|p| p.is_file())
}

#[test]
fn fit_predict_writes_posterior_predictive_and_observed_artifacts() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_e2e_{}", std::process::id()));
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

    // Run the fit.
    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Predict.
    let out = run(&bin, &tmp, &["predict", "@fit"]);
    assert!(
        out.status.success(),
        "predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // gh#890: the line saying where the draws came from names a *method* —
    // the word the config, the store levels and the CLI all use since the
    // `[stages]` → `[method]` split. It used to read "draws from pgas stage
    // 'pgas'".
    let predict_stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        predict_stderr.contains("draws from pgas method 'pgas'"),
        "the resolution line must name the method it read the cloud \
         from:\n{predict_stderr}"
    );
    assert!(
        !predict_stderr.contains("stage"),
        "and `camdl predict` must not print the word `stage` \
         anywhere:\n{predict_stderr}"
    );

    let results = tmp.join("results");

    // ── predictive/weekly_cases.tsv: typed-axis columns + quantile band ──
    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let mut lines = pred_txt.lines();
    let header = lines.next().unwrap();
    assert_eq!(
        header,
        "scenario\ttime\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "predictive header leads with the scenario overlay axis, then both axes + convergence + band"
    );
    let first = lines.next().expect("at least one predictive row");
    let cols: Vec<&str> = first.split('\t').collect();
    assert_eq!(cols.len(), 19, "row shape matches header");
    assert_eq!(cols[0], "fitted", "no --scenario → the no-overlay row tagged fitted");
    assert_eq!(cols[2], "free_forward", "mode axis is explicit");
    assert_eq!(cols[3], "", "no forecast origin, so no horizon");
    assert_eq!(cols[4], "posterior", "treatment axis is explicit (not a plug-in)");
    // gh#1079: this fit declares no holdout, so every data row trained.
    assert_eq!(cols[5], "trained", "the row's role in the fit");
    assert_eq!(cols[6], "observation", "the band is of the observation");
    // rhat_max is carried (a finite number), never silently blank for a PGAS fit.
    assert!(
        cols[7].parse::<f64>().is_ok(),
        "rhat_max carried on the band, got {:?}",
        cols[7]
    );
    // n_draws is a positive count of the cloud the band was reduced over. It
    // sits after the four per-row convergence cells (gh#794).
    assert!(
        cols[13].parse::<usize>().map(|n| n > 0).unwrap_or(false),
        "n_draws carried and positive, got {:?}",
        cols[13]
    );
    // The quantile band is monotone non-decreasing q05 ≤ q25 ≤ … ≤ q95.
    let qs: Vec<f64> = cols[14..19].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in qs.windows(2) {
        assert!(w[0] <= w[1], "quantiles must be ordered: {qs:?}");
    }

    // ── default emits BOTH horizons for a chain-binomial fit: the same file
    // also carries one_step rows (typed `mode` column distinguishes them).
    // The one-step rows are scenario-agnostic, tagged `fitted`.
    let one_step_rows: Vec<&str> = pred_txt
        .lines()
        .filter(|l| l.split('\t').nth(2) == Some("one_step"))
        .collect();
    assert!(
        !one_step_rows.is_empty(),
        "default predict on a chain-binomial fit must also emit one_step rows; \
         got only:\n{pred_txt}"
    );
    // A one-step row is well-formed: fitted scenario, posterior treatment,
    // positive n_draws, ordered quantile band.
    let osr: Vec<&str> = one_step_rows[0].split('\t').collect();
    assert_eq!(osr.len(), 19, "one_step row shape matches header");
    assert_eq!(osr[0], "fitted", "one_step is scenario-agnostic (fitted model)");
    assert_eq!(osr[2], "one_step", "mode axis");
    assert_eq!(osr[4], "posterior", "one-step is a posterior-treatment band");
    assert_eq!(osr[5], "trained", "no holdout: every one-step row trained");
    assert_eq!(osr[6], "observation", "the band is of the observation");
    assert!(
        osr[13].parse::<usize>().map(|n| n > 0).unwrap_or(false),
        "one_step n_draws carried and positive (the subsample used), got {:?}",
        osr[13]
    );
    let osq: Vec<f64> = osr[14..19].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in osq.windows(2) {
        assert!(w[0] <= w[1], "one_step quantiles must be ordered: {osq:?}");
    }

    // ── observed/weekly_cases.tsv: the observed half, same time keys ──
    let obs = find_artifact(&results, "observed", "weekly_cases")
        .expect("observed/weekly_cases.tsv must be written");
    let obs_txt = std::fs::read_to_string(&obs).unwrap();
    let mut olines = obs_txt.lines();
    assert_eq!(olines.next().unwrap(), "time\tvalue", "observed header");
    // The observed value at t=28 is the planted peak, 1303.
    let peak = obs_txt.lines().find(|l| l.starts_with("28\t"));
    assert_eq!(peak, Some("28\t1303"), "observed series is the recorded data");

    // ── quantities/prevalence.tsv: a series (time + banded columns, no dims) ──
    let prev = find_artifact(&results, "quantities-model", "prevalence")
        .expect("quantities/prevalence.tsv must be written");
    let prev_txt = std::fs::read_to_string(&prev).unwrap();
    let mut plines = prev_txt.lines();
    assert_eq!(
        plines.next().unwrap(),
        "scenario\ttime\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
        "series quantity header: scenario + time + the per-row convergence pair + band"
    );
    let prow: Vec<&str> = plines.next().expect("at least one prevalence row").split('\t').collect();
    assert_eq!(prow.len(), 10, "series row shape matches header");
    assert_eq!(prow[0], "fitted", "quantity rows tagged with the scenario");
    let pq: Vec<f64> = prow[5..10].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in pq.windows(2) {
        assert!(w[0] <= w[1], "prevalence quantiles ordered: {pq:?}");
    }

    // ── quantities/peak.tsv: a value scalar (banded, NO censoring trio) ──
    let peakf = find_artifact(&results, "quantities-model", "peak")
        .expect("quantities/peak.tsv must be written");
    let peak_txt = std::fs::read_to_string(&peakf).unwrap();
    let mut klines = peak_txt.lines();
    assert_eq!(
        klines.next().unwrap(),
        "scenario\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
        "value-scalar header: scenario + convergence + banded columns, no time, no censoring"
    );
    let krow: Vec<&str> = klines.next().expect("a peak row").split('\t').collect();
    assert_eq!(krow.len(), 9, "value-scalar row shape matches header");
    assert_eq!(krow[0], "fitted", "value scalar tagged with the scenario");

    // ── quantities/peak_obs.tsv: an observation-source value scalar ──────────
    // `max(observations.weekly_cases)` reduces the per-draw y_sim — same banded
    // value-scalar shape as a state reduction (a Value reduction never censors),
    // and the band must be finite (the obs series was materialized, not empty).
    let peakobsf = find_artifact(&results, "quantities-model", "peak_obs")
        .expect("quantities/peak_obs.tsv must be written");
    let po_txt = std::fs::read_to_string(&peakobsf).unwrap();
    let mut polines = po_txt.lines();
    assert_eq!(
        polines.next().unwrap(),
        "scenario\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
        "obs value-scalar header: scenario + convergence + banded columns, no censoring trio"
    );
    let porow: Vec<&str> = polines.next().expect("a peak_obs row").split('\t').collect();
    assert_eq!(porow.len(), 9, "obs value-scalar row shape matches header");
    let q50: f64 = porow[6].parse().expect("peak_obs q50 parses");
    assert!(
        q50.is_finite() && q50 > 0.0,
        "peak_obs median must be a finite positive count (y_sim materialized), got {q50}"
    );

    // ── quantities/onset.tsv: a time scalar (censorable → the censoring trio) ──
    let onsetf = find_artifact(&results, "quantities-model", "onset")
        .expect("quantities/onset.tsv must be written");
    let onset_txt = std::fs::read_to_string(&onsetf).unwrap();
    let mut olines2 = onset_txt.lines();
    assert_eq!(
        olines2.next().unwrap(),
        "scenario\tn_draws\trhat\tess\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95",
        "censorable scalar header: scenario + convergence + the censoring trio"
    );
    let orow: Vec<&str> = olines2.next().expect("an onset row").split('\t').collect();
    assert_eq!(orow.len(), 12, "censorable row shape matches header");

    // ── quantities/spread.tsv: a Derived over Time scalars inherits censoring ──
    // `spread = onset2 - onset` propagates a censored endpoint, so it must carry
    // the censoring trio (not silently drop censored draws under a plain header).
    let spreadf = find_artifact(&results, "quantities-model", "spread")
        .expect("quantities/spread.tsv must be written");
    let spread_txt = std::fs::read_to_string(&spreadf).unwrap();
    assert_eq!(
        spread_txt.lines().next().unwrap(),
        "scenario\tn_draws\trhat\tess\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95",
        "a Derived transitively referencing a Time scalar inherits the censoring trio (scenario-tagged)"
    );

    // ── quantities.json: lists all three logical quantities, typed ──
    let manifest = find_stored_file(&results, "quantities.json")
        .expect("quantities.json manifest must be written");
    let mtxt = std::fs::read_to_string(&manifest).unwrap();
    let mjson: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    assert_eq!(mjson["schema"], "camdl.quantities/v1", "manifest schema tag");
    let qs = mjson["quantities"].as_array().expect("quantities array");
    let lookup = |n: &str| qs.iter().find(|q| q["name"] == n).unwrap_or_else(|| panic!("manifest missing {n}"));
    assert_eq!(lookup("prevalence")["shape"], "series");
    assert_eq!(lookup("peak")["shape"], "scalar");
    assert_eq!(lookup("peak")["reduce"], "max");
    assert!(lookup("peak")["censoring"].is_null(), "a value reduction is not censorable");
    assert_eq!(lookup("onset")["shape"], "scalar");
    assert_eq!(lookup("onset")["reduce"], "first_above");
    assert!(lookup("onset")["censoring"].is_object(), "a time reduction records right-censoring");
    assert_eq!(lookup("peak_obs")["shape"], "scalar");
    assert_eq!(lookup("peak_obs")["reduce"], "max");
    assert!(lookup("peak_obs")["censoring"].is_null(), "an obs value reduction is not censorable");
    // Every manifest entry carries the scenario overlay field (no overlay →
    // `fitted`).
    assert_eq!(
        lookup("prevalence")["scenario"], "fitted",
        "manifest entry tagged with the scenario (no overlay → fitted)"
    );

    // ── predictive.json: the per-stream join contract (coordinates vs band) ──
    let pmf = find_stored_file(&results, "predictive.json")
        .expect("predictive.json manifest must be written");
    let pjson: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pmf).unwrap()).unwrap();
    // The tag has to move with the fields it describes. v1 → v2 changed the two
    // method-provenance columns from classic Gelman-Rubin R̂ and a Geyer sum to
    // rank-normalized split R̂ and bulk-ESS (gh#84) WITHOUT renaming them, so
    // for those two versions the tag is the only signal that two artifacts in
    // one store hold different quantities. v3 renames them `fit_rhat_max` /
    // `fit_ess_min` and adds the per-row channels (gh#794). v4 adds the `role`
    // coordinate and the two information sets (gh#1079).
    assert_eq!(pjson["schema"], "camdl.predictive/v5", "predictive manifest schema tag");
    let streams = pjson["streams"].as_array().expect("streams array");
    let wc = streams
        .iter()
        .find(|s| s["name"] == "weekly_cases")
        .expect("manifest lists the weekly_cases stream");
    assert_eq!(wc["file"], "predictive/weekly_cases.tsv", "stream file path");
    assert_eq!(wc["value_kind"], "neg_binomial", "value kind = the obs likelihood family");
    // No --sweep, no dims → coordinates are exactly
    // scenario/time/mode/horizon/treatment/role/quantity, matching the
    // predictive TSV header's join keys.
    let coords: Vec<&str> = wc["coordinates"].as_array().unwrap()
        .iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(coords, ["scenario", "time", "mode", "horizon", "treatment", "role", "quantity"],
        "coordinate columns name the group-by keys, in header order");
    let band: Vec<&str> = wc["band"].as_array().unwrap()
        .iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(band, ["q05", "q25", "q50", "q75", "q95"], "band columns are the quantile labels");
    let quantiles: Vec<f64> = wc["quantiles"].as_array().unwrap()
        .iter().map(|q| q.as_f64().unwrap()).collect();
    assert_eq!(quantiles, [0.05, 0.25, 0.50, 0.75, 0.95], "band quantile levels");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A model with two param-overlay scenarios (`set = {...}`) — the supported set.
/// `low_rho` / `high_rho` change the reporting rate, so the predictive bands
/// shift between scenarios while the latent dynamics stay coupled (paired-seed CRN).
const MODEL_WITH_SCENARIOS: &str = r#"
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
    covers        = closing_at(time, 7 'days)
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

#[test]
fn fit_predict_two_scenarios_stack_into_one_file_each_tagged() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_scen_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL_WITH_SCENARIOS).unwrap();
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

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // Two scenarios + free_forward only (keeps the test fast and the assertion
    // about scenario stacking unambiguous).
    let out = run(&bin, &tmp, &[
        "predict", "@fit",
        "--mode", "free_forward",
        "--scenario", "low_rho",
        "--scenario", "high_rho",
    ]);
    assert!(
        out.status.success(),
        "two-scenario predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let results = tmp.join("results");

    // ── ONE predictive/weekly_cases.tsv carries BOTH scenarios' rows, tagged ──
    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let mut lines = pred_txt.lines();
    assert_eq!(
        lines.next().unwrap(),
        "scenario\ttime\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "one header, scenario leads"
    );
    let scenarios_seen: std::collections::BTreeSet<&str> =
        lines.map(|l| l.split('\t').next().unwrap()).collect();
    assert!(
        scenarios_seen.contains("low_rho") && scenarios_seen.contains("high_rho"),
        "both scenarios' rows are in the one file; saw {scenarios_seen:?}"
    );
    // gh#625: the fitted no-overlay arm is ALWAYS emitted — it is the
    // posterior predictive every scenario overlays. Naming scenarios used to
    // silently drop it, leaving downstream sidecars with no reference arm to
    // delta against (and the `--scenario fitted` diagnostic's "emitted
    // automatically" claim false).
    assert!(
        scenarios_seen.contains("fitted"),
        "the fitted reference arm must be present alongside named scenarios; \
         saw {scenarios_seen:?}"
    );

    // ── ONE quantities/peak.tsv carries BOTH scenarios' rows, tagged ──
    let peakf = find_artifact(&results, "quantities-model", "peak")
        .expect("quantities/peak.tsv must be written");
    let peak_txt = std::fs::read_to_string(&peakf).unwrap();
    let mut klines = peak_txt.lines();
    assert_eq!(
        klines.next().unwrap(),
        "scenario\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
        "quantity header leads with scenario"
    );
    let qscen: std::collections::BTreeSet<&str> =
        klines.map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(
        qscen,
        ["fitted", "high_rho", "low_rho"].into_iter().collect(),
        "one quantity row per scenario plus the fitted reference (gh#625), \
         each tagged"
    );

    // ── quantities.json: a `scenario` field per entry (one entry per scenario) ──
    let manifest = find_stored_file(&results, "quantities.json")
        .expect("quantities.json manifest");
    let mtxt = std::fs::read_to_string(&manifest).unwrap();
    let mjson: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    let entries = mjson["quantities"].as_array().expect("quantities array");
    let scen_fields: std::collections::BTreeSet<String> = entries
        .iter()
        .filter(|e| e["name"] == "peak")
        .filter_map(|e| e["scenario"].as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(
        scen_fields,
        ["fitted".to_string(), "high_rho".to_string(), "low_rho".to_string()]
            .into_iter().collect(),
        "manifest carries one peak entry per scenario plus the fitted \
         reference (gh#625), each with its scenario field"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A model whose scenario SCALES the reporting rate `rho` (fixed at 0.2 in the
/// fit). A `scale = { rho = 2.0 }` applied ONCE gives effective rho = 0.4 (mean
/// y_rep ∝ 0.4·projected); applied TWICE (the deleted fold-hack double-apply,
/// which folded scale into each draw AND let the resolver re-apply it) gives
/// rho = 0.8 — a 4× shift from baseline, not 2×. The bands' median ratio
/// distinguishes the two: ≈2 if correct, ≈4 if double-applied.
const MODEL_WITH_SCALE_SCENARIO: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.01, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
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

scenarios {
  double_rho { scale = { rho = 2.0 } }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

fn fit_toml_rho_fixed(algorithm_block: &str, store: &str, rho: f64) -> String {
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
rho = {rho}
k   = 10.0

{algorithm_block}
"#
    )
}

/// Read a `predictive/<stream>.tsv`: for each scenario, the per-time q50
/// (median) cells. Returns scenario → Vec<(time, q50)>.
fn read_q50_by_scenario(path: &Path) -> std::collections::BTreeMap<String, Vec<(String, f64)>> {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let scen_i = header.iter().position(|c| *c == "scenario").unwrap();
    let time_i = header.iter().position(|c| *c == "time").unwrap();
    let q50_i = header.iter().position(|c| *c == "q50").unwrap();
    let horizon_i = header.iter().position(|c| *c == "mode").unwrap();
    let mut out: std::collections::BTreeMap<String, Vec<(String, f64)>> =
        std::collections::BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        // Compare like-for-like horizons only (free_forward).
        if f[horizon_i] != "free_forward" {
            continue;
        }
        out.entry(f[scen_i].to_string())
            .or_default()
            .push((f[time_i].to_string(), f[q50_i].parse::<f64>().unwrap_or(0.0)));
    }
    out
}

#[test]
fn fit_predict_scale_scenario_applied_exactly_once_not_squared() {
    // Regression guard for the deleted `ScenarioOverlay::apply_to_draw` fold-hack.
    // With the resolver doing scenario > draw, the engine applies the scenario
    // `scale` ONCE; the old hand-fold + resolver re-apply double-applied it
    // (×2.0 → ×4.0). We emit the `double_rho` scale scenario AND the no-overlay
    // baseline in one run and compare median bands.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_scale_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL_WITH_SCALE_SCENARIO).unwrap();
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
    // rho fixed at 0.2: ×2.0 = 0.4 (in bounds); a double-apply would be 0.8.
    std::fs::write(tmp.join("fit.toml"), fit_toml_rho_fixed(pgas, "results", 0.2)).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // BOTH the scale scenario and the no-overlay baseline (fitted) in one run.
    let out = run(&bin, &tmp, &[
        "predict", "@fit",
        "--mode", "free_forward",
        "--scenario", "double_rho",
        "--seed", "1",
    ]);
    assert!(
        out.status.success(),
        "scale-scenario predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let by_scen = read_q50_by_scenario(&pred);

    // Run a SECOND predict with NO scenario to get the fitted baseline (the
    // explicit --scenario above suppresses the no-overlay row).
    let out2 = run(&bin, &tmp, &[
        "predict", "@fit",
        "--mode", "free_forward",
        "--seed", "1",
    ]);
    assert!(out2.status.success(),
        "baseline predict failed:\nstderr={}", String::from_utf8_lossy(&out2.stderr));
    let base_pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv (baseline) must be written");
    let base_by_scen = read_q50_by_scenario(&base_pred);

    let scaled = by_scen.get("double_rho").expect("double_rho rows present");
    let baseline = base_by_scen.get("fitted").expect("fitted rows present");

    // Pair times and take the ratio of medians at peak weeks (where the signal
    // is strong, so Poisson/NB noise is a small relative perturbation). Pool the
    // ratio across the high-count weeks to average out per-week noise.
    let base_map: std::collections::BTreeMap<&str, f64> =
        baseline.iter().map(|(t, v)| (t.as_str(), *v)).collect();
    let mut ratios: Vec<f64> = Vec::new();
    for (t, v_scaled) in scaled {
        if let Some(&v_base) = base_map.get(t.as_str()) {
            // Only weeks with a meaningful baseline count (avoid 0/0 and tiny tails).
            if v_base >= 50.0 {
                ratios.push(v_scaled / v_base);
            }
        }
    }
    assert!(!ratios.is_empty(), "no comparable high-count weeks; bands: {scaled:?} / {baseline:?}");
    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

    // Correct (applied once): ratio ≈ 2.0. Double-applied: ≈ 4.0. A generous
    // band around 2.0 that EXCLUDES 4.0 is the discriminating assertion.
    assert!(
        mean_ratio > 1.5 && mean_ratio < 2.8,
        "scale = {{ rho = 2.0 }} must shift the median band ≈2× (applied once), \
         not ≈4× (double-applied fold-hack). mean ratio = {mean_ratio:.3} over \
         {} weeks; ratios = {ratios:?}",
        ratios.len()
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_scenario_named_fitted_in_model_is_rejected() {
    // The reserved-name guard at the compiler boundary: a model `scenarios {}`
    // preset named `fitted` is an E291 error naming the reservation + fix.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_reserved_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let bad_model = MODEL_WITH_SCENARIOS.replace(
        "scenarios {\n  low_rho  { set = { rho = 0.3 } }\n  high_rho { set = { rho = 0.8 } }\n}",
        "scenarios {\n  fitted { set = { rho = 0.3 } }\n}",
    );
    assert!(bad_model.contains("fitted { set"), "the substitution applied");
    std::fs::write(tmp.join("model.camdl"), &bad_model).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    // Compiling the model (via `simulate --dry-run`, which only needs the model)
    // surfaces the E291 reservation diagnostic.
    let out = run(&bin, &tmp, &["simulate", "model.camdl", "--dry-run"]);
    assert!(!out.status.success(), "a model preset named fitted must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E291") && stderr.contains("reserved") && stderr.contains("fitted"),
        "E291 names the reservation and the offending name; got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_is_self_contained_after_loose_camdl_removed() {
    // gh#322: `fit run` stores the compiled IR in the run store, so
    // `camdl predict` resolves the model from that copy — not from the loose
    // `.camdl`, which may have moved. We delete the source `.camdl` after the
    // fit; predict must still succeed.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_portable_{}", std::process::id()));
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

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The compiled IR is stored, non-empty.
    let archive = find_stored_file(&tmp.join("results"), "model.ir.json")
        .expect("fit run must store model.ir.json");
    assert!(
        std::fs::metadata(&archive).unwrap().len() > 0,
        "archived model.ir.json is non-empty"
    );

    // Remove the loose source model — the run must remain self-contained.
    std::fs::remove_file(tmp.join("model.camdl")).unwrap();

    let out = run(
        &bin,
        &tmp,
        &["predict", "@fit", "--mode", "free_forward"],
    );
    assert!(
        out.status.success(),
        "predict must resolve the model from the archived IR after the loose \
         .camdl is removed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        find_artifact(&tmp.join("results"), "predictive", "weekly_cases").is_some(),
        "predictive artifact written from the archived model"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_resolves_at_label_and_hash_prefix() {
    // A fit run named by `@label` and by a prefix of its run id, not just its
    // folder path. Both resolve to the same stored run.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_handle_{}", std::process::id()));
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

    let out = run(
        &bin,
        &tmp,
        &["fit", "run", "fit.toml", "--label", "jigawa-baseline", "--seed", "1"],
    );
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    let the_run = fit_store_support::the_run(&tmp.join("results"));
    let hash8: String = fit_store_support::run_id(&the_run).chars().take(8).collect();

    // (1) Resolve by @label.
    let out = run(&bin, &tmp, &["predict", "@jigawa-baseline", "--mode", "free_forward"]);
    assert!(
        out.status.success(),
        "predict @label failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        find_artifact(&tmp.join("results"), "predictive", "weekly_cases").is_some(),
        "@label resolved to the fit and predicted"
    );

    // (2) Resolve by run id prefix (positional handle).
    let out = run(&bin, &tmp, &["predict", &hash8, "--mode", "free_forward"]);
    assert!(
        out.status.success(),
        "predict <hash> failed for prefix {hash8}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // (3) An unknown label is a typed not-found error, not a panic.
    let out = run(&bin, &tmp, &["predict", "@nope", "--mode", "free_forward"]);
    assert!(!out.status.success(), "unknown @label must error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no stored result matches `@nope`"), "actionable not-found, got: {stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn a_second_fit_under_a_held_label_takes_it() {
    // A fit label moves: a second fit asking for a label another run holds
    // runs, takes the label, says which run it displaced, and `@label` then
    // names the second run while `@label~1` names the first.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_handle_ambig_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    // Two fits that differ in their method (→ two runs of one problem)
    // asking for the same label.
    let if2_a = r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 20
cooling = 0.7
"#;
    let if2_b = r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 30
cooling = 0.7
"#;
    std::fs::write(tmp.join("a.toml"), fit_toml(if2_a, "results")).unwrap();
    std::fs::write(tmp.join("b.toml"), fit_toml(if2_b, "results")).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "a.toml", "--label", "dup", "--seed", "1"]);
    assert!(out.status.success(), "fit run a.toml failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));
    let out = run(&bin, &tmp, &["fit", "run", "b.toml", "--label", "dup", "--seed", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "a held label moves to the new run:\n{stderr}");
    assert!(stderr.contains("note: @dup now names run"), "the move is said: {stderr}");

    // The run folder of each fit file, by the config its run records.
    let run_of = |config: &str| -> String {
        let run = fit_store_support::runs(&tmp.join("results")).into_iter()
            .find(|r| fit_store_support::run_inputs(r)["header"]["config"] == config)
            .unwrap_or_else(|| panic!("no run of {config}"));
        run.file_name().unwrap().to_string_lossy().into_owned()
    };
    let out = run(&bin, &tmp, &["fit", "summary", "@dup"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains(&run_of("b.toml")) && stderr.contains("(1 earlier holder)"),
        "@dup names b.toml's run, and the resolution line says so: {stderr}");
    let out = run(&bin, &tmp, &["fit", "summary", "@dup~1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success() && stderr.contains(&run_of("a.toml")),
        "@dup~1 names a.toml's run: {stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The distinct `n_draws` values on the free_forward rows of a predictive TSV.
fn free_forward_n_draws(path: &Path) -> std::collections::BTreeSet<usize> {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let hz = header.iter().position(|c| *c == "mode").unwrap();
    let nd = header.iter().position(|c| *c == "n_draws").unwrap();
    lines
        .filter(|l| l.split('\t').nth(hz) == Some("free_forward"))
        .map(|l| l.split('\t').nth(nd).unwrap().parse::<usize>().unwrap())
        .collect()
}

#[test]
fn fit_predict_free_forward_honors_n_draws() {
    // gh#387: the free-forward horizon must honor `--n-draws` via an even
    // subsample of the posterior cloud, not silently replay EVERY draw. Before
    // the fix, free-forward built its replay from the full cloud and reported
    // `posterior.n_draws()` regardless of `--n-draws` — so a long-burn-in ODE
    // fit (thousands of ~seconds-each solves) never finished and the
    // `predictive/` artifact was never written. Red→green: pre-fix the capped
    // run reports the full cloud (≠ 10); post-fix it reports exactly 10.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_ndraws_{}", std::process::id()));
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

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));

    // (1) No cap → the default (200) ≥ the cloud, so free-forward replays the
    // whole cloud and reports its full size ( > 10 ). Establishes the baseline
    // and confirms the cloud is bigger than the cap we test with.
    let out = run(&bin, &tmp, &["predict", "@fit", "--mode", "free_forward"]);
    assert!(out.status.success(), "uncapped predict failed:\nstderr={}", String::from_utf8_lossy(&out.stderr));
    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv (uncapped) must be written");
    let full = free_forward_n_draws(&pred);
    assert_eq!(full.len(), 1, "one n_draws value across free_forward rows, got {full:?}");
    let full_n = *full.iter().next().unwrap();
    assert!(full_n > 10, "the posterior cloud ({full_n}) must exceed the --n-draws cap for this test");

    // (2) --n-draws 10 → free-forward replays exactly 10 draws and reports 10.
    let out = run(&bin, &tmp, &[
        "predict", "@fit", "--mode", "free_forward", "--n-draws", "10",
    ]);
    assert!(
        out.status.success(),
        "capped predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv (capped) must be written");
    let capped = free_forward_n_draws(&pred);
    assert_eq!(
        capped,
        [10].into_iter().collect(),
        "free_forward must honor --n-draws 10 (report the subsample count, not the full cloud), got {capped:?}"
    );
    // The artifact IS written and carries data rows (the watcher's Predictive tab
    // is no longer empty).
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let ff_rows = pred_txt.lines().filter(|l| l.split('\t').nth(2) == Some("free_forward")).count();
    assert!(ff_rows > 0, "capped predictive artifact must have free_forward data rows");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// An optimizer run is predicted at its one winning point, labelled
/// `plug_in`, never as a posterior band; more than one copy is refused.
#[test]
fn fit_predict_of_an_optimizer_fit_is_its_winning_point() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_refuse_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    let if2 = r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 25
cooling = 0.7
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(if2, "results")).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "if2 fit run failed");

    let out = run(&bin, &tmp, &["predict", "@fit", "--n-draws", "2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an optimizer fit has one point: {stderr}");
    assert!(stderr.contains("--n-draws 2") && stderr.contains("optimizer fit"), "{stderr}");

    let out = run(&bin, &tmp, &["predict", "@fit"]);
    assert!(out.status.success(), "predict of an optimizer fit: {}", String::from_utf8_lossy(&out.stderr));
    let band = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("a predictive band of the winning point");
    let text = std::fs::read_to_string(&band).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let c = header.iter().position(|h| *h == "treatment").expect("a treatment column");
    let rows: Vec<&str> = lines.filter(|l| !l.is_empty()).map(|l| l.split('\t').nth(c).unwrap()).collect();
    assert!(!rows.is_empty() && rows.iter().all(|t| *t == "plug_in"), "labelled plug_in, not posterior: {rows:?}");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A prediction made under an engine switch (`CAMDL_EVAL_UNRESOLVED`, which
/// routes every rate through the other evaluator) is a different computation
/// from one made without it, so it is stored at a different address: the
/// address key folds the engine identity the other commands key, build and
/// switches, not the build alone. Under the build alone the two predictions
/// shared one directory, the second overwrote the first, and a reader of that
/// address (the `fit table` quantity fast path) could not tell which engine
/// made what it read.
#[test]
fn a_prediction_under_an_engine_switch_has_its_own_address() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_switch_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    let pgas = r#"[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 30
burn_in = 10
thin = 1
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(pgas, "results")).unwrap();
    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run failed:\n{}", String::from_utf8_lossy(&out.stderr));

    // The run's folder, named directly, so the switched predict reads the
    // run the default engine made.
    let run_dir = fit_store_support::the_run(&tmp.join("results"));
    let leaf = run_dir.to_string_lossy().into_owned();
    let addresses = || -> std::collections::BTreeSet<String> {
        std::fs::read_dir(run_dir.join("predict")).into_iter().flatten().flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    };

    let plain = run(&bin, &tmp, &["predict", &leaf]);
    assert!(plain.status.success(), "predict failed:\n{}", String::from_utf8_lossy(&plain.stderr));
    let first = addresses();
    assert_eq!(first.len(), 1, "{first:?}");

    let switched = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["predict", &leaf])
        .current_dir(&tmp)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", tmp.join("results"))
        .env("CAMDL_EVAL_UNRESOLVED", "1")
        .output()
        .expect("spawn camdl");
    assert!(switched.status.success(), "switched predict failed:\n{}", String::from_utf8_lossy(&switched.stderr));
    let both = addresses();
    assert_eq!(both.len(), 2, "the switched prediction must not share the default one's address: {both:?}");
    assert!(both.is_superset(&first), "and the default one's address is where it was");

    // Each manifest of a prediction names the engine that made it, as the
    // run's `identity.json` spells the build: the switched prediction's name
    // its switch, the default one's name none, and both name this build.
    let build = fit_store_support::json(&run_dir.join("identity.json"))["ambient"]["build"].clone();
    assert!(build.is_string(), "the run records its build");
    let switched_dir = both.difference(&first).next().expect("the switched address");
    let default_dir = first.iter().next().unwrap();
    for file in ["predictive.json", "observed.json", "quantities.json", "report.json"] {
        let read = |dir: &str| -> serde_json::Value {
            let p = run_dir.join("predict").join(dir).join(file);
            serde_json::from_str(&std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))).unwrap()
        };
        let (on, off) = (read(switched_dir), read(default_dir));
        assert_eq!(on["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]),
            "{file} of the switched prediction names its switch: {on}");
        assert!(off.get("engine_switches").is_none(), "{file} of the default prediction names none: {off}");
        assert_eq!(on["engine_version"], build, "{file}: the build, as the run's identity spells it");
        assert_eq!(off["engine_version"], build, "{file}");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
