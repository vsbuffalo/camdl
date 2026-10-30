//! gh#708: the one-step band is filtered on the step the FIT walked.
//!
//! Under the one dt rule (`dt_rule`) a fit walks the model's declared
//! `simulate { dt }` when there is one, else its `[config] dt`. `fit predict`
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
//! `fit predict --horizon one_step` runs then draw from identical clouds at
//! the same seed, so their bands must be identical too. Reading the archived
//! `[config] dt` instead (its default 1.0 in the first arm) filters the first
//! band on 1.0 and the bands differ. No fitted value enters the comparison.

use std::path::{Path, PathBuf};
use std::process::Command;

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
    projected  = prevalence(I)
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
        r#"output_dir = "results"
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

/// The one file named `name` anywhere under `root`.
fn find(root: &Path, name: &str) -> PathBuf {
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|f| f == name) {
                hits.push(p);
            }
        }
    }
    assert_eq!(hits.len(), 1, "expected one {name} under {}, found {hits:?}", root.display());
    hits.pop().unwrap()
}

/// The `one_step` rows of a predictive TSV, as text.
fn one_step_rows(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let horizon = header.iter().position(|c| *c == "horizon")
        .unwrap_or_else(|| panic!("no `horizon` column in {header:?}"));
    lines
        .filter(|l| l.split('\t').nth(horizon) == Some("one_step"))
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
            Command::new(&bin)
                .args(args)
                .current_dir(&dir)
                .env("CAMDL_SKIP_VERSION_CHECK", "1")
                .output()
                .expect("spawn camdl")
        };
        let out = run(&["fit", "run", "fit.toml", "--seed", "1", "--no-progress"]);
        assert!(
            out.status.success(),
            "[{arm}] fit run failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let out = run(&["fit", "predict", "--fit", "fit.toml", "--horizon", "one_step"]);
        assert!(
            out.status.success(),
            "[{arm}] fit predict failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );

        // `<segment>/<method>-<h8>/seed_<n>-<h8>/draws.tsv`, so the segment —
        // where the predictive artifacts land — is three levels up.
        let draws_path = find(&dir.join("results"), "draws.tsv");
        let segment = draws_path.parent().unwrap().parent().unwrap().parent().unwrap();
        draws.push(std::fs::read_to_string(&draws_path).unwrap());
        predictive.push(segment.join("predictive").join("cases.tsv"));
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
