//! Every route a user has to set one parameter gives the run the same value,
//! and each verb that takes parameter values goes through the one resolver
//! (`params_resolver::resolve_parameters`, docs/camdl-run-spec.md §1.3).
//!
//! Two kinds of test:
//!
//! - **Route equivalence.** `beta = 0.3` set through each route that can
//!   express it — a params file, `--param`, a second params file, a scenario
//!   `set`, a scenario `scale` of a base value, a scenario inherited through
//!   `extends`, a one-point `--sweep`, a one-row `--sweep-file`, a one-row
//!   `--draws FILE` — must give byte-identical trajectory and observation
//!   output under one seed. Where a route legitimately differs (a design
//!   coordinate adds columns; a scenario is its own identity level) the
//!   difference is stated and pinned rather than ignored.
//! - **Each verb reports through the resolver.** A replacement of one tier's
//!   value by another's prints the resolver's notice, once, and is recorded
//!   in the verb's `run.json` (or, for `fit run`, `fit.meta.json`)
//!   `parameters_provenance`; a verb that resolved parameters some other way
//!   would print nothing.
//!
//! The tier matrix itself (every pair, severity, the record) is unit-tested in
//! `params_resolver.rs`; these tests prove the verbs reach it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Where `fit predict` wrote for the one run in `seg`: `<seg>/predict/<run id>/`
/// (gh#896). A path that does not exist while nothing has been predicted.
fn predict_out(seg: &std::path::Path) -> std::path::PathBuf {
    let runs: Vec<std::path::PathBuf> = std::fs::read_dir(seg.join("predict"))
        .into_iter().flatten().flatten().map(|e| e.path()).collect();
    assert!(runs.len() <= 1, "expected one predicted run under {}: {runs:?}", seg.display());
    runs.into_iter().next().unwrap_or_else(|| seg.join("predict").join("none"))
}


fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).args(args).current_dir(dir).output().expect("spawn camdl")
}

fn ok(out: &Output, what: &str) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{what} failed:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout));
    stderr
}

/// The lines of `stderr` that report a parameter override.
fn override_lines(stderr: &str) -> Vec<String> {
    stderr.lines().filter(|l| l.contains(" replaced ")).map(str::to_string).collect()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `run.json` under `root`, parsed.
fn run_jsons(root: &Path) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rj = dir.join("run.json");
        if rj.is_file() {
            out.push(serde_json::from_str(&read(&rj)).unwrap());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    out
}

/// The single leaf `run.json` under `root`.
fn one_leaf(root: &Path) -> serde_json::Value {
    let mut leaves = run_jsons(root);
    assert_eq!(leaves.len(), 1, "expected one leaf under {}", root.display());
    leaves.remove(0)
}

/// A leaf's identity: its `run_id` and each level's hash by level name.
#[derive(Debug, Clone, PartialEq)]
struct Identity {
    run_id: String,
    levels: BTreeMap<String, String>,
}

fn identity(leaf: &serde_json::Value) -> Identity {
    Identity {
        run_id: leaf["run_id"].as_str().expect("run_id").to_string(),
        levels: leaf["levels"].as_array().expect("levels").iter()
            .map(|l| (l["name"].as_str().unwrap().to_string(),
                      l["hash"].as_str().unwrap().to_string()))
            .collect(),
    }
}

// ─── simulate: every route to beta = 0.3 ────────────────────────────────────

/// An SIR with a daily Poisson observation. Neither parameter has a value in
/// the model, so every route supplies `gamma` through `base.toml` (or
/// `p.toml`), and `beta` through the route under test.
///
/// `set_b` sets beta; `scale_b` doubles base.toml's 0.15; `child` inherits
/// `parent`'s set through `extends`; `doubled` inherits a set of 0.15 and
/// doubles it; `expr` computes beta from the parent's value. Doubling is
/// exact in floating point, so 2 × 0.15 is the same double as 0.3.
const ROUTES_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.0, 5.0]
  gamma : rate in [0.0, 5.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10 }
observations {
  cases {
    columns { time : time, cases : count }
    covers = closing_at(time, 1 'days)
    projected = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = 0.5 * projected)
  }
}
scenarios {
  both    { set = { beta = 0.3  gamma = 0.1 } }
  set_b   { set = { beta = 0.3 } }
  scale_b { scale = { beta = 2.0 } }
  parent  { set = { beta = 0.3 } }
  child   { extends = parent }
  half    { set = { beta = 0.15 } }
  doubled { extends = half  scale = { beta = 2.0 } }
  expr    { extends = half  set = { beta = beta * 2.0 } }
}
simulate { from = 0 'days  to = 20 'days }
"#;

fn routes_fixture(dir: &Path) {
    std::fs::write(dir.join("m.camdl"), ROUTES_MODEL).unwrap();
    std::fs::write(dir.join("p.toml"), "beta = 0.3\ngamma = 0.1\n").unwrap();
    std::fs::write(dir.join("base.toml"), "beta = 0.15\ngamma = 0.1\n").unwrap();
    std::fs::write(dir.join("point.tsv"), "beta\n0.3\n").unwrap();
    // A draws file needs two columns (`main.rs` refuses one), so the row
    // carries gamma at base.toml's value: an equal value, not an override.
    std::fs::write(dir.join("draw.tsv"), "beta\tgamma\n0.3\t0.1\n").unwrap();
}

/// What one simulate route produced: the trajectory and observation mirrors
/// with any design-coordinate columns removed, the leaf identity, and the
/// override lines it printed.
struct RouteRun {
    traj: String,
    obs: String,
    id: Identity,
    overrides: Vec<String>,
}

/// Strip the leading design-coordinate columns a sweep adds
/// (`point_id`, `sweep:beta`, `seed`), after checking they say what the route
/// asked for: point 0, beta 0.3, seed 7.
fn strip_design_columns(table: &str) -> String {
    let mut out = String::new();
    for (i, line) in table.lines().enumerate() {
        if line.starts_with('#') {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let cols: Vec<&str> = line.splitn(4, '\t').collect();
        assert_eq!(cols.len(), 4, "short row: {line}");
        if cols[0] == "point_id" {
            assert_eq!(&cols[..3], ["point_id", "sweep:beta", "seed"], "header {i}");
        } else {
            assert_eq!(&cols[..3], ["0", "0.3", "7"], "row {i}: {line}");
        }
        out.push_str(cols[3]);
        out.push('\n');
    }
    out
}

fn simulate_route(dir: &Path, name: &str, args: &[&str]) -> RouteRun {
    let store = format!("st_{name}");
    let traj = format!("traj_{name}.tsv");
    let obs = format!("obs_{name}.tsv");
    let mut argv = vec!["simulate", "m.camdl", "--backend", "chain_binomial", "--dt", "1",
        "--seed", "7", "--output-dir", &store, "-o", &traj, "--obs", &obs];
    argv.extend_from_slice(args);
    let stderr = ok(&camdl(dir, &argv), &format!("simulate route {name} {args:?}"));
    let design = args.iter().any(|a| a.starts_with("--sweep"));
    let (mut t, mut o) = (read(&dir.join(&traj)), read(&dir.join(&obs)));
    if design {
        t = strip_design_columns(&t);
        o = strip_design_columns(&o);
    }
    RouteRun {
        traj: t,
        obs: o,
        id: identity(&one_leaf(&dir.join(&store).join("sims"))),
        overrides: override_lines(&stderr),
    }
}

#[test]
fn simulate_every_route_to_one_value_gives_one_trajectory() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    routes_fixture(dir);

    // (route, argv, whether the route is a scenario arm)
    let routes: &[(&str, &[&str], bool)] = &[
        ("params", &["--params", "p.toml"], false),
        ("param", &["--params", "base.toml", "--param", "beta=0.3"], false),
        ("second_file", &["--params", "base.toml", "--params", "p.toml"], false),
        // `lin(0.3,0.3,1)` is refused (lin needs n >= 2); a one-value list is
        // the one-point sweep.
        ("sweep", &["--params", "base.toml", "--sweep", "beta=0.3"], false),
        ("sweep_file", &["--params", "base.toml", "--sweep-file", "point.tsv"], false),
        ("draws_file", &["--params", "base.toml", "--draws", "draw.tsv"], false),
        ("scenario_set", &["--params", "base.toml", "--scenario", "set_b"], true),
        ("scenario_scale", &["--params", "base.toml", "--scenario", "scale_b"], true),
        ("extends_set", &["--params", "base.toml", "--scenario", "child"], true),
        ("extends_scale", &["--params", "base.toml", "--scenario", "doubled"], true),
        ("extends_expr", &["--params", "base.toml", "--scenario", "expr"], true),
    ];
    let runs: Vec<(&str, bool, RouteRun)> = routes.iter()
        .map(|(name, args, scen)| (*name, *scen, simulate_route(dir, name, args)))
        .collect();
    let reference = &runs[0].2;
    assert!(reference.overrides.is_empty(), "{:?}", reference.overrides);

    for (name, is_scenario, run) in &runs {
        // One trajectory and one observation series, byte for byte.
        assert_eq!(run.traj, reference.traj, "route {name}: trajectory differs");
        assert_eq!(run.obs, reference.obs, "route {name}: observations differ");

        // One resolved parameter set → one `params` level; no route moves the
        // process seed (a single cell runs at the base seed, and a design
        // point never enters the seed) → one `seed` level.
        for level in ["model", "config", "params", "seed"] {
            assert_eq!(run.id.levels[level], reference.id.levels[level],
                "route {name}: `{level}` level differs");
        }

        // Every route but the plain params file replaced base.toml's 0.15,
        // and said so, naming its own source.
        if *name != "params" {
            assert_eq!(run.overrides.len(), 1, "route {name}: {:?}", run.overrides);
            assert!(run.overrides[0].contains("replaced beta = 0.15 (from base.toml) with 0.3"),
                "route {name}: {:?}", run.overrides);
        }

        // The no-scenario routes are one leaf. A scenario route is its own
        // `scenario` level — the level hashes the scenario's effective delta
        // (`set`/`scale`/enable/disable), which a scenario that sets beta has
        // and a `--param` does not — so its run_id differs while its output
        // does not. That split is the identity rule, not a defect.
        if *is_scenario {
            assert_ne!(run.id.levels["scenario"], reference.id.levels["scenario"],
                "route {name}");
            assert_ne!(run.id.run_id, reference.id.run_id, "route {name}");
        } else {
            assert_eq!(run.id, reference.id, "route {name}: identity differs");
        }
    }

    // Scenarios with the same effective delta share the scenario level:
    // `set_b`, `child` (inherits `set = { beta = 0.3 }`) and `expr` (computes
    // 0.3 at compile time) all reach the run as `set = { beta = 0.3 }`. A
    // `scale` delta is a different delta even where it lands on the same
    // value.
    let by: BTreeMap<&str, &Identity> = runs.iter().map(|(n, _, r)| (*n, &r.id)).collect();
    assert_eq!(by["extends_set"], by["scenario_set"]);
    assert_eq!(by["extends_expr"], by["scenario_set"]);
    assert_ne!(by["scenario_scale"].levels["scenario"], by["scenario_set"].levels["scenario"]);
    assert_ne!(by["extends_scale"].levels["scenario"], by["scenario_scale"].levels["scenario"]);
}

// ─── pfilter: --params, --param and a scenario score one likelihood ─────────

#[test]
fn pfilter_routes_to_one_point_score_one_likelihood_under_one_run_id() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    routes_fixture(dir);
    // Data: the reference route's own observations.
    let sim = simulate_route(dir, "data", &["--params", "p.toml"]);
    std::fs::write(dir.join("cases.tsv"), &sim.obs).unwrap();

    // `pfilter` refuses `--scenario` with `--params`/`--param` (gh#194), so the
    // scenario route carries the whole point.
    let routes: &[(&str, &[&str])] = &[
        ("params", &["--params", "p.toml"]),
        ("param", &["--param", "beta=0.3", "--param", "gamma=0.1"]),
        ("file_then_param", &["--params", "base.toml", "--param", "beta=0.3"]),
        ("scenario", &["--scenario", "both"]),
    ];
    let mut results: Vec<(&str, String, Identity, Vec<String>)> = Vec::new();
    for (name, args) in routes {
        let store = dir.join(format!("pf_{name}"));
        let mut argv = vec!["pfilter", "m.camdl", "--data", "cases.tsv",
            "--particles", "200", "--seed", "5"];
        argv.extend_from_slice(args);
        let out = Command::new(binary()).args(&argv).current_dir(dir)
            .env("CAMDL_OUTPUT_DIR", &store).output().expect("spawn");
        let stderr = ok(&out, &format!("pfilter route {name}"));
        let loglik = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let leaf = one_leaf(&store.join("pfilters"));
        assert_eq!(leaf["inputs"]["params"],
            serde_json::json!([["beta", 0.3], ["gamma", 0.1]]), "route {name}");
        results.push((name, loglik, identity(&leaf), override_lines(&stderr)));
    }
    let (_, ref_ll, ref_id, _) = &results[0];
    for (name, ll, id, overrides) in &results {
        assert_eq!(ll, ref_ll, "route {name}: log-likelihood differs");
        assert_eq!(id, ref_id, "route {name}: identity differs");
        let want: Vec<String> = if *name == "file_then_param" {
            vec!["info: --param/--fixed replaced beta = 0.15 (from base.toml) with 0.3".into()]
        } else {
            vec![]
        };
        assert_eq!(overrides, &want, "route {name}");
    }
}

#[test]
fn pfilter_reports_a_later_params_file_over_an_earlier_one() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    routes_fixture(dir);
    let sim = simulate_route(dir, "data", &["--params", "p.toml"]);
    std::fs::write(dir.join("cases.tsv"), &sim.obs).unwrap();
    let out = Command::new(binary())
        .args(["pfilter", "m.camdl", "--data", "cases.tsv", "--particles", "50",
               "--params", "base.toml", "--params", "p.toml"])
        .current_dir(dir).env("CAMDL_OUTPUT_DIR", dir.join("pf")).output().expect("spawn");
    let stderr = ok(&out, "pfilter");
    assert_eq!(override_lines(&stderr),
        vec!["info: params file p.toml replaced beta = 0.15 (from base.toml) with 0.3"]);
    let leaf = one_leaf(&dir.join("pf/pfilters"));
    let prov = &leaf["inputs"]["parameters_provenance"];
    assert_eq!(prov["beta"]["value"], 0.3);
    assert_eq!(prov["beta"]["source"], "fixed_file");
    assert_eq!(prov["beta"]["overrides"], serde_json::json!([
        { "replaced": { "source": "fixed_file", "value": 0.15, "path": "base.toml" },
          "by": { "source": "fixed_file", "value": 0.3, "path": "p.toml" } },
    ]));
    // gamma: base.toml's 0.1, then p.toml's equal 0.1 — not an override.
    assert_eq!(prov["gamma"]["value"], 0.1);
    assert!(prov["gamma"].get("overrides").is_none(), "{}", prov["gamma"]);
    assert_eq!(leaf["inputs"]["notices"], serde_json::json!([{
        "severity": "info", "code": "param-override",
        "message": "params file p.toml replaced beta = 0.15 (from base.toml) with 0.3",
    }]));
}

// ─── dev eval ────────────────────────────────────────────────────────────────

#[test]
fn dev_eval_reports_param_over_a_params_file_and_evaluates_the_winner() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    routes_fixture(dir);
    let out = camdl(dir, &["dev", "eval", "m.camdl", "--params", "base.toml",
        "--param", "beta=0.3", "--expr", "beta", "--at", "0"]);
    let stderr = ok(&out, "dev eval");
    assert_eq!(override_lines(&stderr),
        vec!["info: --param/--fixed replaced beta = 0.15 (from base.toml) with 0.3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.lines().nth(1), Some("0\t0.300000"), "stdout: {stdout}");
}

// ─── profile and survey: --fixed-file, scenario, --fixed ────────────────────

const PROFILE_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
scenarios {
  base { set = { beta = 0.3  gamma = 0.1  N0 = 1000 } }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;

fn profile_fixture(dir: &Path) {
    std::fs::write(dir.join("sir.camdl"), PROFILE_MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n")
        .unwrap();
    std::fs::write(dir.join("ff.toml"), "gamma = 0.12\n").unwrap();
}

#[test]
fn profile_reports_and_records_each_replacement_of_a_fixed_value() {
    // --fixed-file sets gamma 0.12, the scenario replaces it with 0.1, --fixed
    // replaces that with 0.15: two replacements, in tier order.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    profile_fixture(dir);
    let out = Command::new(binary())
        .args(["profile", "sir.camdl", "--data", "cases.tsv", "--obs", "cases",
               "--sweep", "beta=lin(0.2,0.4,2)", "--particles", "30", "--iterations", "3",
               "--n-starts", "1", "--rw-sd", "auto", "--output", "p.tsv", "--seed", "1",
               "--scenario", "base",
               "--fixed-file", "ff.toml", "--fixed", "gamma=0.15"])
        .current_dir(dir).env("CAMDL_OUTPUT_DIR", dir.join("out")).output().expect("spawn");
    let stderr = ok(&out, "profile");
    assert_eq!(override_lines(&stderr), vec![
        "info: scenario `base` replaced gamma = 0.12 (from ff.toml) with 0.1",
        "info: --param/--fixed replaced gamma = 0.1 (from scenario `base`) with 0.15",
    ]);
    let leaves = run_jsons(&dir.join("out"));
    let leaves: Vec<&serde_json::Value> = leaves.iter()
        .filter(|l| l["inputs"]["provenance"].is_object()).collect();
    assert!(!leaves.is_empty(), "no profile-point leaves");
    for leaf in leaves {
        let gamma = &leaf["inputs"]["provenance"]["parameters_provenance"]["gamma"];
        assert_eq!(gamma["value"], 0.15);
        assert_eq!(gamma["source"], "fixed_cli");
        assert_eq!(gamma["overrides"], serde_json::json!([
            { "replaced": { "source": "fixed_file", "value": 0.12, "path": "ff.toml" },
              "by": { "source": "scenario", "value": 0.1, "scenario": "base" } },
            { "replaced": { "source": "scenario", "value": 0.1, "scenario": "base" },
              "by": { "source": "fixed_cli", "value": 0.15 } },
        ]));
    }
}

#[test]
fn survey_reports_fixed_over_a_scenario_and_fixes_the_winner() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    profile_fixture(dir);
    let out = camdl(dir, &["survey", "sir.camdl", "--data", "cases.tsv",
        "--estimate", "beta=0.001:5.0", "--scenario", "base", "--fixed", "gamma=0.15",
        "--eval", "simulate", "--n-points", "4", "--seed", "1", "--output", "sv"]);
    let stderr = ok(&out, "survey");
    assert_eq!(override_lines(&stderr),
        vec!["info: --param/--fixed replaced gamma = 0.1 (from scenario `base`) with 0.15"]);
    let leaf = one_leaf(&dir.join("sv"));
    assert_eq!(leaf["inputs"]["fixed"]["gamma"], 0.15);
    let gamma = &leaf["inputs"]["parameters_provenance"]["gamma"];
    assert_eq!(gamma["value"], 0.15);
    assert_eq!(gamma["source"], "fixed_cli");
    assert_eq!(gamma["overrides"], serde_json::json!([
        { "replaced": { "source": "scenario", "value": 0.1, "scenario": "base" },
          "by": { "source": "fixed_cli", "value": 0.15 } },
    ]));
    assert_eq!(leaf["inputs"]["notices"].as_array().map(Vec::len), Some(1));
}

// ─── profile --starts: the starts rule's values are their own tier ─────────

/// Run `profile` on the profile fixture with `extra` flags; returns stderr and
/// every profile-point leaf's `parameters_provenance`.
fn profile_run(dir: &Path, store: &str, extra: &[&str]) -> (String, Vec<serde_json::Value>) {
    let mut argv = vec!["profile", "sir.camdl", "--data", "cases.tsv", "--obs", "cases",
        "--sweep", "beta=lin(0.2,0.4,2)", "--particles", "30", "--iterations", "3",
        "--n-starts", "1", "--rw-sd", "auto", "--output", "p.tsv", "--seed", "1"];
    argv.extend_from_slice(extra);
    let out = Command::new(binary()).args(&argv).current_dir(dir)
        .env("CAMDL_OUTPUT_DIR", dir.join(store)).output().expect("spawn");
    let stderr = ok(&out, &format!("profile {extra:?}"));
    let provs: Vec<serde_json::Value> = run_jsons(&dir.join(store)).into_iter()
        .filter(|l| l["inputs"]["provenance"].is_object())
        .map(|l| l["inputs"]["provenance"]["parameters_provenance"].clone())
        .collect();
    assert!(!provs.is_empty(), "no profile-point leaves");
    (stderr, provs)
}

#[test]
fn profile_starts_from_params_is_recorded_as_the_file_not_the_model_default() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    profile_fixture(dir);
    std::fs::write(dir.join("start.toml"), "beta = 0.3\ngamma = 0.2\nN0 = 1000\n").unwrap();
    let (stderr, provs) = profile_run(dir, "out", &["--starts", "from_params=start.toml"]);
    assert!(override_lines(&stderr).is_empty(), "{stderr}");
    for prov in provs {
        for (name, v) in [("gamma", 0.2), ("N0", 1000.0)] {
            assert_eq!(prov[name]["value"], v, "{name}");
            assert_eq!(prov[name]["source"], "starts_from_params", "{name}");
        }
    }
}

#[test]
fn profile_scenario_over_a_starts_file_is_reported_and_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    profile_fixture(dir);
    std::fs::write(dir.join("start.toml"), "beta = 0.3\ngamma = 0.2\nN0 = 1000\n").unwrap();
    let (stderr, provs) = profile_run(dir, "out",
        &["--starts", "from_params=start.toml", "--scenario", "base"]);
    // `base` sets gamma 0.1 over the file's 0.2, and beta and N0 to the
    // file's own 0.3 and 1000 (equal values, not replacements).
    assert_eq!(override_lines(&stderr), vec![
        "info: scenario `base` replaced gamma = 0.2 (from --starts from_params=start.toml) \
         with 0.1",
    ]);
    for prov in provs {
        assert_eq!(prov["gamma"]["value"], 0.1);
        assert_eq!(prov["gamma"]["source"], "scenario");
        assert_eq!(prov["gamma"]["overrides"], serde_json::json!([
            { "replaced": { "source": "starts_from_params", "value": 0.2, "path": "start.toml" },
              "by": { "source": "scenario", "value": 0.1, "scenario": "base" } },
        ]));
        assert_eq!(prov["N0"]["source"], "scenario");
        assert!(prov["N0"].get("overrides").is_none(), "{}", prov["N0"]);
    }
}

#[test]
fn profile_replacing_a_from_mle_start_is_a_warning() {
    // A stored fit's point estimate, planted as a method leaf: `from_mle`
    // reads its `fit_state.toml` `start_values`.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    profile_fixture(dir);
    let leaf = dir.join("fitleaf");
    std::fs::create_dir_all(&leaf).unwrap();
    std::fs::write(leaf.join("fit_state.toml"),
        "method = \"if2\"\nseed = 1\ntimestamp = \"2026-01-01T00:00:00Z\"\n\
         best_loglik = -20.0\ninitial_loglik = -30.0\nbest_chain = 0\nn_chains = 1\n\n\
         [start_values]\nbeta = 0.3\ngamma = 0.2\nN0 = 1000\n\n[rw_sd]\n").unwrap();
    let handle = leaf.to_string_lossy().into_owned();
    let starts = format!("from_mle={handle}");
    let (stderr, provs) = profile_run(dir, "out",
        &["--starts", &starts, "--fixed", "gamma=0.15"]);
    assert_eq!(override_lines(&stderr), vec![format!(
        "warning: --param/--fixed replaced gamma = 0.2 (from --starts from_mle={handle}, \
         a fit's estimate) with 0.15")]);
    for prov in provs {
        assert_eq!(prov["gamma"]["overrides"], serde_json::json!([
            { "replaced": { "source": "starts_from_mle", "value": 0.2, "fit": handle },
              "by": { "source": "fixed_cli", "value": 0.15 } },
        ]));
        assert_eq!(prov["N0"]["source"], "starts_from_mle");
    }
}

// ─── From a posterior: simulate --draws posterior and fit predict ───────────

/// A closed SIR with a weekly negative-binomial observation; `k` (the
/// dispersion) is fixed in the fit, and `k_alt` sets it to 12.
const POSTERIOR_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability
  k     : positive
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
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
  k_alt { set = { k = 12.0 } }
}
simulate { from = 0 'days  to = 56 'days }
"#;

const POSTERIOR_FIT: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 40
burn_in = 10
thin = 1
"#;

/// Fit the posterior fixture in `dir`; returns the PGAS method leaf.
fn posterior_fit(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("model.camdl"), POSTERIOR_MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"),
        "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n")
        .unwrap();
    std::fs::write(dir.join("fit.toml"), POSTERIOR_FIT).unwrap();
    ok(&camdl(dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
    let fits = dir.join("results/fits");
    let fit = std::fs::read_dir(&fits).unwrap().flatten().next().expect("fit dir").path();
    let method = std::fs::read_dir(&fit).unwrap().flatten().map(|e| e.path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("pgas-"))
        .expect("pgas leaf");
    std::fs::read_dir(&method).unwrap().flatten().map(|e| e.path())
        .find(|p| p.is_dir()).expect("seed leaf")
}

#[test]
fn simulate_from_a_posterior_param_and_scenario_set_give_one_output() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let leaf = posterior_fit(dir);
    let leaf = leaf.to_string_lossy();
    let run = |name: &str, extra: &[&str]| -> (String, String, Vec<String>) {
        let (traj, obs) = (format!("traj_{name}.tsv"), format!("obs_{name}.tsv"));
        let store = format!("st_{name}");
        let mut argv = vec!["simulate", "model.camdl", "--draws", "posterior", "--fit", &leaf,
            "-n", "5", "--seed", "4", "--output-dir", &store, "-o", &traj, "--obs", &obs];
        argv.extend_from_slice(extra);
        let stderr = ok(&camdl(dir, &argv), &format!("simulate {name}"));
        (read(&dir.join(&traj)), read(&dir.join(&obs)), override_lines(&stderr))
    };
    let (t_param, o_param, n_param) = run("param", &["--param", "k=12"]);
    let (t_scen, o_scen, n_scen) = run("scenario", &["--scenario", "k_alt"]);
    let (t_none, o_none, _) = run("none", &[]);
    assert_eq!(t_param, t_scen, "trajectories differ");
    assert_eq!(o_param, o_scen, "observations differ");
    // k is observation-only: the trajectory is the fitted one, and the
    // observations move — so the comparison above is not vacuous.
    assert_eq!(t_param, t_none);
    assert_ne!(o_param, o_none, "k=12 left the observations unchanged");
    assert_eq!(n_param,
        vec!["info: --param/--fixed replaced k = 10 (from the draw or sweep point) with 12"]);
    assert_eq!(n_scen,
        vec!["info: scenario `k_alt` replaced k = 10 (from the draw or sweep point) with 12"]);
}

/// The `free_forward` rows of a `fit predict` predictive table, with the
/// `scenario` and `sweep:k` columns removed, for the rows whose scenario is
/// `arm` and whose `sweep:k` (when present) is `k`.
fn predictive_rows(table: &str, arm: &str, k: Option<&str>) -> Vec<String> {
    let mut lines = table.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name);
    let scen = col("scenario").expect("scenario column");
    let sweep = col("sweep:k");
    lines.filter_map(|l| {
        let cells: Vec<&str> = l.split('\t').collect();
        if cells[scen] != arm { return None; }
        if let (Some(i), Some(k)) = (sweep, k) { if cells[i] != k { return None; } }
        Some(cells.iter().enumerate()
            .filter(|(i, _)| *i != scen && Some(*i) != sweep)
            .map(|(_, c)| *c).collect::<Vec<_>>().join("\t"))
    }).collect()
}

#[test]
fn fit_predict_scenario_set_and_one_point_sweep_give_one_band() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    posterior_fit(dir);
    let predict = |extra: &[&str]| -> (String, Vec<String>) {
        let mut argv = vec!["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward",
            "--seed", "3"];
        argv.extend_from_slice(extra);
        let stderr = ok(&camdl(dir, &argv), &format!("fit predict {extra:?}"));
        let fit = std::fs::read_dir(dir.join("results/fits")).unwrap().flatten().next()
            .unwrap().path();
        (read(&predict_out(&fit).join("predictive/weekly_cases.tsv")), override_lines(&stderr))
    };
    let (by_scenario, n_scenario) = predict(&["--scenario", "k_alt"]);
    let (by_sweep, n_sweep) = predict(&["--sweep", "k=12"]);
    let scenario_rows = predictive_rows(&by_scenario, "k_alt", None);
    let sweep_rows = predictive_rows(&by_sweep, "fitted", Some("12"));
    assert!(!scenario_rows.is_empty());
    assert_eq!(scenario_rows, sweep_rows, "the k = 12 bands differ by route");
    // And they are not the fitted (k = 10) band.
    assert_ne!(scenario_rows, predictive_rows(&by_scenario, "fitted", None));

    // The scenario replaces the posterior row's k, and says so. A `--sweep`
    // value is written into the draw row itself (`fit/predict.rs`, the same
    // draw/sweep tier), so `fit predict` reports that replacement of the
    // draws' values itself, once per swept parameter.
    assert_eq!(n_scenario,
        vec!["info: scenario `k_alt` replaced k = 10 (from the draw or sweep point) with 12"]);
    assert_eq!(n_sweep,
        vec!["info: --sweep replaced k = 10 (from the posterior draws) with 12"]);
}

#[test]
fn fit_predict_sweep_reports_the_replaced_draws_once_as_a_range() {
    // Two sweep points, each replaying every draw: one line, both ranges.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    posterior_fit(dir);
    let out = camdl(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward",
        "--seed", "3", "--sweep", "beta=0.3,0.4"]);
    let lines = override_lines(&ok(&out, "fit predict --sweep"));
    assert_eq!(lines.len(), 1, "{lines:?}");
    let line = &lines[0];
    let prefix = "info: --sweep replaced beta = ";
    assert!(line.starts_with(prefix), "{line}");
    // The posterior's beta varies by draw: a range, then the swept values and
    // the draw count.
    let rest = &line[prefix.len()..];
    let (lo_hi, tail) = rest.split_once(" (from the posterior draws) with 0.3 to 0.4, across ")
        .unwrap_or_else(|| panic!("{line}"));
    let (lo, hi) = lo_hi.split_once(" to ").unwrap_or_else(|| panic!("{line}"));
    let (lo, hi): (f64, f64) = (lo.parse().unwrap(), hi.parse().unwrap());
    assert!(lo < hi && (0.05..=1.0).contains(&lo) && (0.05..=1.0).contains(&hi), "{line}");
    assert!(tail.ends_with(" draws"), "{line}");
}

// ─── fit run: the resolver's decision in fit.meta.json, reported once ───────

#[test]
fn fit_run_records_the_resolver_decision_once_per_segment() {
    // A fit.toml scenario agreeing with `[fixed]` on k = 12 (a disagreeing one
    // is refused, `fit_run_scenario_over_fixed.rs`). fit.meta.json records the
    // resolver's value and source for every parameter, and an equal value is
    // not an override, so nothing is reported, for either of the two fit
    // seeds (two cells, one fit segment).
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), POSTERIOR_MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"),
        "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n")
        .unwrap();
    std::fs::write(dir.join("fit.toml"), r#"output_dir = "results"
scenario = "k_alt"
fit_seeds = [1, 2]
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 12.0
[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 50
"#).unwrap();
    let stderr = ok(&camdl(dir, &["fit", "run", "fit.toml"]), "fit run");
    assert_eq!(override_lines(&stderr), Vec::<String>::new());
    let fit = std::fs::read_dir(dir.join("results/fits")).unwrap().flatten().next()
        .expect("fit segment").path();
    let meta: serde_json::Value = serde_json::from_str(&read(&fit.join("fit.meta.json"))).unwrap();
    let k = &meta["parameters_provenance"]["k"];
    assert_eq!(k["value"], 12.0);
    assert_eq!(k["source"], "scenario");
    assert!(k.get("overrides").is_none(), "an equal value is not an override: {k}");
    assert_eq!(meta["parameters_provenance"]["rho"]["value"], 0.6);
    assert_eq!(meta["parameters_provenance"]["rho"]["source"], "fit_toml_fixed");
    assert_eq!(meta["parameters_provenance"]["beta"]["role"], "estimated");
}

// ─── contrasts: an arm's scenario over the draws is reported once ───────────

/// The posterior fixture's SIR with an SIA at day 28, a deaths quantity and
/// one contrast whose `with_sia` arm also sets the dispersion `k`. `fit
/// predict` runs no `--scenario` here, so the only resolutions under
/// `with_sia` are the contrast arm's, one per forkable draw.
const CONTRAST_MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia]  set = { k = 12.0 } }
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
  total = final(D)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

#[test]
fn fit_predict_reports_a_contrast_arms_scenario_over_the_draws_once() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), CONTRAST_MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"),
        "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n")
        .unwrap();
    std::fs::write(dir.join("fit.toml"), r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.5], start = 0.5 }
gamma = { bounds = [0.05, 0.5], start = 0.15 }
[fixed]
mu  = 0.05
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 30
burn_in = 10
thin = 1
"#).unwrap();
    ok(&camdl(dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
    let stderr = ok(&camdl(dir, &["fit", "predict", "--fit", "fit.toml",
        "--horizon", "free_forward", "--seed", "3"]), "fit predict");
    assert!(stderr.contains("contrast 'averted' — fork at"), "the contrast ran: {stderr}");
    assert_eq!(override_lines(&stderr),
        vec!["info: scenario `with_sia` replaced k = 10 (from the draw or sweep point) with 12"]);
}
