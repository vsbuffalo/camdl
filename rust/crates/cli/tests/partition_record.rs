//! Every `fit run`, `survey` and `profile` records its partition — which
//! parameters it fixed, estimated or swept, and what declared each — and the
//! legacy `estimated` / `fixed` fields of its record are generated from it
//! (proposal `docs/dev/proposals/2026-10-26-parameter-roles.md`, Stage 1,
//! §7.2, D4).
//!
//! Stage 1 changes no computed result: these tests read records only.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

/// The SIR recovery model with a typed constant `omega` that the dynamics
/// read. A constant is not a parameter: no fit config names it.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95]
}
let omega : rate = 0.01
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  waning    : R --> S  @ omega * R
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = 10.0)
  }
}
simulate { from = 0 'days  to = 56 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// A fit config: `estimate` and `fixed` are the bodies of the two blocks.
fn fit_toml(estimate: &str, fixed: &str) -> String {
    format!(r#"output_dir = "res"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
{estimate}
[fixed]
{fixed}
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 2
cooling = 0.7
"#)
}

const ESTIMATE: &str = "beta  = { bounds = [0.1, 0.9], start = 0.4 }\ngamma = { start = 0.15 }";
const FIXED: &str = "N0 = 10000\nI0 = 10\nrho = 0.6";

fn fixture(fit: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), fit).unwrap();
    tmp
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    let out = Command::new(binary()).args(args).current_dir(dir).output().expect("spawn");
    assert!(out.status.success(), "camdl {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    out
}

/// Every file named `name` under `root`, sorted.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|f| f == name) {
                found.push(p);
            }
        }
    }
    found.sort();
    found
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The legacy fields a record's `partition` generates: estimated names in
/// order, and every held value.
fn legacy_of(partition: &Value) -> (Value, Value) {
    let mut estimated = Vec::new();
    let mut fixed = serde_json::Map::new();
    for (name, rec) in partition["parameters"].as_object().unwrap() {
        match rec["role"].as_str().unwrap() {
            "estimated" => estimated.push(Value::from(name.clone())),
            "fixed" => {
                fixed.insert(name.clone(), rec["value"].clone());
            }
            _ => {}
        }
    }
    let mut sorted: Vec<(String, Value)> = fixed.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    (Value::Array(estimated), Value::Object(sorted.into_iter().collect()))
}

#[test]
fn fit_run_records_its_partition_and_generates_the_legacy_fields() {
    let tmp = fixture(&fit_toml(ESTIMATE, FIXED));
    camdl(tmp.path(), &["fit", "run", "fit.toml", "--seed", "1"]);
    let metas = find_all(&tmp.path().join("res"), "fit.meta.json");
    assert_eq!(metas.len(), 1, "{metas:?}");
    let meta = json(&metas[0]);
    let part = &meta["partition"];
    assert_eq!(part["status"], "resolved", "{meta:#}");
    let beta = &part["parameters"]["beta"];
    assert_eq!(beta["role"], "estimated");
    assert_eq!(beta["declared_by"], "fit_estimate");
    assert_eq!(beta["search"], serde_json::json!([0.1, 0.9]));
    assert_eq!(beta["search_from"], "fit_config");
    assert_eq!(beta["support"], serde_json::json!([0.05, 1.0]));
    assert_eq!(beta["transform"], "log_interval");
    assert_eq!(beta["prior"]["from"], "model_ir");
    let gamma = &part["parameters"]["gamma"];
    assert_eq!(gamma["search_from"], "support");
    assert_eq!(gamma["start"], 0.15);
    let rho = &part["parameters"]["rho"];
    assert_eq!(rho["role"], "fixed");
    assert_eq!(rho["declared_by"], "fit_fixed");
    assert_eq!(rho["via"], "inline");
    assert_eq!(rho["value"], 0.6);
    assert_eq!(rho["value_from"]["source"], "fit_toml_fixed");
    // A typed constant is not a parameter: it is in no partition and no
    // legacy field, and the fit ran without a `[fixed]` entry for it (gh#965).
    assert!(part["parameters"].get("omega").is_none(), "{part:#}");
    assert!(meta["fixed"].get("omega").is_none(), "{meta:#}");
    let (estimated, fixed) = legacy_of(part);
    assert_eq!(meta["estimated"], estimated);
    assert_eq!(meta["fixed"], fixed);
    assert_eq!(meta["estimated"], serde_json::json!(["beta", "gamma"]));
}

#[test]
fn a_swept_fit_records_each_points_cell_partition() {
    let tmp = fixture(&fit_toml(ESTIMATE, FIXED));
    camdl(tmp.path(), &["fit", "run", "fit.toml", "--seed", "1", "--sweep", "rho=0.4,0.6"]);
    let metas = find_all(&tmp.path().join("res"), "fit.meta.json");
    assert_eq!(metas.len(), 2, "one segment per sweep point: {metas:?}");
    let mut seen = Vec::new();
    for m in &metas {
        let meta = json(m);
        let rho = &meta["partition"]["parameters"]["rho"];
        assert_eq!(rho["role"], "fixed", "{meta:#}");
        assert_eq!(rho["job_role"], "design");
        assert_eq!(rho["special"], "sweep_over_fixed");
        assert_eq!(rho["declared_by"], "cli_sweep");
        assert_eq!(rho["value_from"]["source"], "design_point");
        assert_eq!(meta["fixed"]["rho"], rho["value"]);
        let (estimated, fixed) = legacy_of(&meta["partition"]);
        assert_eq!((&meta["estimated"], &meta["fixed"]), (&estimated, &fixed));
        seen.push(rho["value"].as_f64().unwrap());
    }
    seen.sort_by(f64::total_cmp);
    assert_eq!(seen, vec![0.4, 0.6]);
}

/// A run today's rules accept and the partition refuses — here a simplex
/// group that `--fixed` splits — records why, and keeps the legacy fields as
/// the command wrote them before.
#[test]
fn a_survey_whose_roles_form_no_partition_records_why_and_still_runs() {
    let fit = fit_toml(ESTIMATE, FIXED) + "\n[[simplex_groups]]\nparams = [\"beta\", \"gamma\"]\n";
    let tmp = fixture(&fit);
    camdl(tmp.path(), &["survey", "model.camdl", "--fit", "fit.toml", "--fixed", "gamma=0.15",
        "--n-points", "4", "--eval-particles", "20", "--output", "srv"]);
    let inputs = &json(&find_all(&tmp.path().join("srv"), "run.json")[0])["inputs"];
    assert_eq!(inputs["partition"]["status"], "unresolved", "{inputs:#}");
    assert!(inputs["partition"]["reason"].as_str().unwrap().contains("simplex"), "{inputs:#}");
    assert_eq!(inputs["estimated"], serde_json::json!(["beta"]));
    assert_eq!(inputs["fixed"]["gamma"], 0.15);
}

#[test]
fn survey_records_its_partition() {
    let tmp = fixture(&fit_toml(ESTIMATE, FIXED));
    camdl(tmp.path(), &["survey", "model.camdl", "--fit", "fit.toml", "--n-points", "4",
        "--eval-particles", "20", "--output", "srv"]);
    let runs = find_all(&tmp.path().join("srv"), "run.json");
    assert_eq!(runs.len(), 1, "{runs:?}");
    let inputs = &json(&runs[0])["inputs"];
    let part = &inputs["partition"];
    assert_eq!(part["status"], "resolved", "{inputs:#}");
    assert_eq!(part["parameters"]["beta"]["role"], "estimated");
    assert_eq!(part["parameters"]["N0"]["value"], 10000.0);
    let (estimated, fixed) = legacy_of(part);
    assert_eq!((&inputs["estimated"], &inputs["fixed"]), (&estimated, &fixed));

    // Inline: `--estimate` declares, an undeclared parameter is fixed by default.
    camdl(tmp.path(), &["survey", "model.camdl", "--data", "weekly_cases.tsv",
        "--estimate", "beta=0.1:0.9", "--fixed", "gamma=0.15", "--fixed", "N0=10000",
        "--fixed", "I0=10", "--fixed", "rho=0.6", "--n-points", "4",
        "--eval-particles", "20", "--output", "srv_inline"]);
    let inputs = &json(&find_all(&tmp.path().join("srv_inline"), "run.json")[0])["inputs"];
    let part = &inputs["partition"];
    assert_eq!(part["parameters"]["beta"]["declared_by"], "cli_estimate", "{inputs:#}");
    assert_eq!(part["parameters"]["beta"]["search_from"], "cli");
    assert_eq!(part["parameters"]["gamma"]["declared_by"], "cli_fixed");
    let (estimated, fixed) = legacy_of(part);
    assert_eq!((&inputs["estimated"], &inputs["fixed"]), (&estimated, &fixed));
}

/// `profile` stores under the default output root, `results/`.
#[test]
fn profile_records_the_jobs_partition_and_each_cells() {
    let tmp = fixture(&fit_toml(ESTIMATE, FIXED));
    camdl(tmp.path(), &["profile", "model.camdl", "--fit", "fit.toml", "--sweep", "beta=0.3,0.5",
        "--rw-sd", "auto", "--particles", "30", "--iterations", "2", "--n-starts", "1"]);
    // The base segment's sidecar: the job's partition, `beta` the focal design
    // parameter, its `[estimate]` entry not used.
    let metas = find_all(&tmp.path().join("results"), "fit.meta.json");
    assert_eq!(metas.len(), 1, "{metas:?}");
    let meta = json(&metas[0]);
    let part = &meta["partition"];
    assert_eq!(part["status"], "resolved", "{meta:#}");
    assert_eq!(part["parameters"]["beta"]["role"], "design");
    assert_eq!(part["parameters"]["beta"]["special"], "profile_focal");
    assert_eq!(part["parameters"]["gamma"]["role"], "estimated");
    assert_eq!(part["parameters"]["gamma"]["declared_by"], "fit_estimate");
    assert_eq!(part["parameters"]["rho"]["role"], "fixed");
    let (estimated, fixed) = legacy_of(part);
    assert_eq!((&meta["estimated"], &meta["fixed"]), (&estimated, &fixed));
    assert_eq!(meta["estimated"], serde_json::json!(["gamma"]));

    // Each leaf: the cell's partition, `beta` fixed at its grid value.
    let leaves = find_all(&tmp.path().join("results"), "run.json");
    assert_eq!(leaves.len(), 2, "{leaves:?}");
    let mut seen = Vec::new();
    for leaf in &leaves {
        let run = json(leaf);
        let beta = &run["inputs"]["provenance"]["partition"]["parameters"]["beta"];
        assert_eq!(beta["role"], "fixed", "{run:#}");
        assert_eq!(beta["job_role"], "design");
        assert_eq!(beta["special"], "profile_focal");
        assert_eq!(beta["value_from"]["source"], "design_point");
        seen.push(beta["value"].as_f64().unwrap());
    }
    seen.sort_by(f64::total_cmp);
    assert_eq!(seen, vec![0.3, 0.5]);
}

/// Without a fit config, `--rw-sd`'s names are the estimated set: recorded as
/// estimate declarations, the rest fixed by `--fixed`.
#[test]
fn a_cli_only_profile_records_its_rw_sd_selection_as_estimate_declarations() {
    let tmp = fixture(&fit_toml(ESTIMATE, FIXED));
    // Today a swept parameter still needs a value from elsewhere (proposal
    // §3.2 item 10): the starts file gives `beta` one.
    std::fs::write(tmp.path().join("start.toml"), "beta = 0.4\ngamma = 0.15\n").unwrap();
    camdl(tmp.path(), &["profile", "model.camdl", "--data", "weekly_cases.tsv",
        "--sweep", "beta=0.4", "--rw-sd", "gamma=0.02", "--fixed", "N0=10000",
        "--fixed", "I0=10", "--fixed", "rho=0.6", "--starts", "from_params=start.toml",
        "--particles", "30", "--iterations", "2", "--n-starts", "1"]);
    let meta = json(&find_all(&tmp.path().join("results"), "fit.meta.json")[0]);
    let part = &meta["partition"];
    assert_eq!(part["status"], "resolved", "{meta:#}");
    assert_eq!(part["parameters"]["gamma"]["role"], "estimated");
    assert_eq!(part["parameters"]["gamma"]["declared_by"], "cli_estimate");
    assert_eq!(part["parameters"]["N0"]["declared_by"], "cli_fixed");
    assert_eq!(part["parameters"]["beta"]["role"], "design");
    assert!(part["parameters"]["beta"].get("special").is_none(), "{part:#}");
    let (estimated, fixed) = legacy_of(part);
    assert_eq!((&meta["estimated"], &meta["fixed"]), (&estimated, &fixed));
}

/// The legacy `estimated` list keeps the order of the survey's columns, which
/// follow `[estimate]`, not the model.
#[test]
fn survey_keeps_the_estimated_order_of_its_columns() {
    let estimate = "gamma = { start = 0.15 }\nbeta  = { bounds = [0.1, 0.9], start = 0.4 }";
    let tmp = fixture(&fit_toml(estimate, FIXED));
    camdl(tmp.path(), &["survey", "model.camdl", "--fit", "fit.toml", "--n-points", "4",
        "--eval-particles", "20", "--output", "srv"]);
    let run = &find_all(&tmp.path().join("srv"), "run.json")[0];
    let inputs = &json(run)["inputs"];
    assert_eq!(inputs["estimated"], serde_json::json!(["gamma", "beta"]), "{inputs:#}");
    let header = std::fs::read_to_string(run.parent().unwrap().join("landscape.tsv")).unwrap();
    let cols: Vec<&str> = header.lines().find(|l| !l.starts_with('#')).unwrap().split('\t')
        .filter(|c| *c == "gamma" || *c == "beta").collect();
    assert_eq!(cols, vec!["gamma", "beta"], "{header}");
}
