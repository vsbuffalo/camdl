//! gh#708: the one-step band is filtered on the step the FIT walked, not on
//! the step the model file happens to declare.
//!
//! `fit run` walks its filter on the fit toml's `[config] dt`; `fit predict`
//! derived its own from `model.simulation.dt`, and nothing syncs the two. They
//! agree on most fits — both default to 1.0 — and diverge exactly when a model
//! declares `simulate { dt = X }` and the fit declares a different `[config]
//! dt`. Under `chain_binomial` the step sets the transition probability
//! `1 - exp(-rate·dt)`, so the one-step predictive was then drawn from a
//! *different process* than the posterior it replays, silently, and plotted
//! against the observed series as if the two agreed.
//!
//! ## The oracle
//!
//! Two fits of the same problem at the same `[config] dt = 1.0`, differing
//! only in what their model files declare for `simulate { dt }`: 0.2 in one,
//! 1.0 in the other. A `chain_binomial` fit never reads `model.simulation.dt`
//! — it resolves its step from the fit config — so the two fits are the same
//! inference and their posterior clouds are byte-identical. That equality is
//! asserted first, as the premise: it is what makes the one-step comparison a
//! statement about the filter's step rather than about two different
//! posteriors.
//!
//! The two `fit predict --horizon one_step` runs then draw from identical
//! clouds at the same seed, so their bands must be identical too. Before the
//! fix the first filtered at 0.2 and the second at 1.0 and the bands differed.
//! No fitted value enters the comparison, so nothing here depends on either
//! fit converging.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// The SIR model, with the `simulate { dt = ... }` the fit's own `[config] dt`
/// is supposed to override for the one-step band.
fn model(sim_dt: &str) -> String {
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
simulate {{ from = 0 'days  to = 6 'days  dt = {sim_dt} 'days }}
"#
    )
}

const DATA: &str = "time\tcases\n1\t20\n2\t35\n3\t60\n4\t90\n5\t120\n6\t150\n";

/// The fit half, at a fixed `[config] dt = 1.0` whatever the model declares.
fn fit_toml(model_file: &str) -> String {
    format!(
        r#"output_dir = "results"
[model]
camdl = "{model_file}"
[data.observations]
cases = "cases.tsv"
[config]
dt = 1.0
[estimate]
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
    for (arm, sim_dt) in [("coarse", "1.0"), ("fine", "0.2")] {
        let dir = tmp.join(arm);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), model(sim_dt)).unwrap();
        std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
        std::fs::write(dir.join("fit.toml"), fit_toml("sir.camdl")).unwrap();

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

    // Premise: a chain_binomial fit reads its step from `[config] dt`, so the
    // model's declaration does not touch the inference and the two arms are
    // the same posterior. If this ever stops holding, the band comparison
    // below stops being about the filter's step.
    assert_eq!(
        draws[0], draws[1],
        "the two arms differ only in the model's `simulate {{ dt }}`, which a \
         chain_binomial fit does not read — their posterior clouds must be \
         identical for the band comparison to mean anything"
    );

    let coarse = one_step_rows(&predictive[0]);
    let fine = one_step_rows(&predictive[1]);
    assert!(!coarse.is_empty(), "the one-step horizon produced no rows");
    assert_eq!(
        coarse, fine,
        "the one-step band must be filtered on the fit's `[config] dt` (1.0 in \
         both arms), so two fits with identical posteriors must produce \
         identical bands. Taking the model's `simulate {{ dt }}` instead makes \
         the band on the right a replay of a different process — under \
         chain_binomial the step sets `1 - exp(-rate*dt)` — plotted against \
         the same observed series."
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
