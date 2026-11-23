//! gh#708: the one-step band is filtered on the step the FIT walked.
//!
//! Under the one dt rule (`dt_rule`) a fit walks the model's declared
//! `simulate { dt }` when there is one, else its `[config] dt`. `camdl predict`
//! re-filters the observed series for the one-step band, so it must walk the
//! same step: under `chain_binomial` the step sets the transition probability
//! `1 - exp(-rate·dt)`, and a band on another step is a replay of a different
//! process plotted against the data as if the two agreed.
//!
//! ## The oracle
//!
//! Two fits of the same problem at the same step, 0.2, reached the two ways
//! the rule allows: one model declares `simulate { dt = 0.2 }` and its fit
//! writes no `[config] dt`; the other model declares no step and its fit
//! writes `[config] dt = 0.2`. Both fits walk 0.2, so their posterior clouds
//! are byte-identical — asserted first, as the premise. The two
//! `camdl predict --mode one_step` runs then draw from identical clouds at
//! the same seed, so their bands must be identical too. Reading the archived
//! `[config] dt` instead (its default 1.0 in the first arm) filters the first
//! band on 1.0 and the bands differ. No fitted value enters the comparison.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

/// The stored prediction under `seg` (a run folder, or any folder above
/// runs) whose `report*.json` is newest: the latest `camdl predict`'s output. A
/// path that does not exist while nothing has been predicted.
fn predict_out(seg: &std::path::Path) -> std::path::PathBuf {
    fit_store_support::predictions(seg)
        .into_iter()
        .filter_map(|d| {
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
        .unwrap_or_else(|| seg.join("predict").join("none"))
}


fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// The SIR model, declaring `simulate { dt = <sim_dt> }` when given.
fn model(sim_dt: Option<&str>) -> String {
    let sim_dt = sim_dt.map(|d| format!(" dt = {d} 'days")).unwrap_or_default();
    format!(
        r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = 6 'days {sim_dt} }}
"#
    )
}

const DATA: &str = "time\tcases\n1\t20\n2\t35\n3\t60\n4\t90\n5\t120\n6\t150\n";

/// The fit half, writing `[config] dt = <config_dt>` when given.
fn fit_toml(model_file: &str, config_dt: Option<&str>) -> String {
    let config = config_dt.map(|d| format!("[config]\ndt = {d}\n")).unwrap_or_default();
    format!(
        r#"store = "results"
[model]
camdl = "{model_file}"
[data.observations]
cases = "cases.tsv"
{config}[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[method]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 2
particles = 8
sweeps    = 8
burn_in   = 2
thin      = 1
starts    = "uniform_unconstrained"
"#
    )
}

/// The `one_step` rows of a predictive TSV, as text.
fn one_step_rows(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let mode = header.iter().position(|c| *c == "mode")
        .unwrap_or_else(|| panic!("no `mode` column in {header:?}"));
    lines
        .filter(|l| l.split('\t').nth(mode) == Some("one_step"))
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn the_one_step_band_is_filtered_on_the_dt_the_fit_used() {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    let tmp = std::env::temp_dir().join(format!("camdl_gh708_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);

    // One directory per arm: the fits are content-addressed on the model, so
    // they would land in different leaves anyway; separate roots keep the
    // `find` helpers unambiguous.
    let mut predictive: Vec<PathBuf> = Vec::new();
    let mut draws: Vec<String> = Vec::new();
    for (arm, sim_dt, config_dt) in [("model", Some("0.2"), None), ("config", None, Some("0.2"))] {
        let dir = tmp.join(arm);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), model(sim_dt)).unwrap();
        std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
        std::fs::write(dir.join("fit.toml"), fit_toml("sir.camdl", config_dt)).unwrap();

        let run = |args: &[&str]| -> std::process::Output {
            Command::new(&bin).env_remove("CLICOLOR_FORCE")
                .args(args)
                .current_dir(&dir)
                .env("CAMDL_STORE", dir.join("results"))
                .env("CAMDL_SKIP_VERSION_CHECK", "1")
                .output()
                .expect("spawn camdl")
        };
        let out = run(&["fit", "run", "fit.toml", "--seed", "1", "--no-progress", "--label", "fit"]);
        assert!(
            out.status.success(),
            "[{arm}] fit run failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let out = run(&["predict", "@fit", "--mode", "one_step"]);
        assert!(
            out.status.success(),
            "[{arm}] predict failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );

        // The run folder holds `draws.tsv`, and its predictions sit inside it.
        let fit_run = fit_store_support::the_run(&dir.join("results"));
        draws.push(std::fs::read_to_string(fit_run.join("draws.tsv")).unwrap());
        predictive.push(predict_out(&fit_run).join("predictive").join("cases.tsv"));
    }

    // Premise: both fits walk 0.2 — one from the model's declaration, one
    // from the fit's `[config] dt` — so they are the same inference.
    assert_eq!(
        draws[0], draws[1],
        "both arms walk dt = 0.2 (the model's declared step in one, the fit's \
         [config] dt in the other), so their posterior clouds must be identical \
         for the band comparison to mean anything"
    );

    let by_model = one_step_rows(&predictive[0]);
    let by_config = one_step_rows(&predictive[1]);
    assert!(!by_model.is_empty(), "the one-step horizon produced no rows");
    assert_eq!(
        by_model, by_config,
        "the one-step band must walk the step the fit walked (0.2 in both arms), \
         so two fits with identical posteriors must produce identical bands. \
         Reading the archived [config] dt (its default 1.0 in the first arm) \
         makes that band a replay of a different process."
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
