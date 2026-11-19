//! A case for every input of `camdl survey`, checked as its class predicts
//! (`rust/crates/cli/src/input_class.rs`, `SURVEY`; proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests"). The runner is `input_class_runner`.
//!
//! Each case is one test, named `<class>_<input>`; an environment variable's
//! case is `<class>_env_<field>`; `input_class::tests::every_survey_input_has_a_case`
//! fails if an input has none. What each class asserts, over the survey leaf:
//!
//! - `keyed(level)`: changed alone, the leaf's run id changes, and so does
//!   its hash at the level the class table names (`model`, `config`, `box` or
//!   `seed`). An environment variable keyed at a simulation's `model` level
//!   enters a survey at its `model` level too, which holds the engine.
//! - `execution`: the run id and the stored bytes stay the same.
//! - `presentation`, `provenance`: the leaf is unaffected.
//! - `child`: the leaf's run id and files are unchanged, and the child is at
//!   its address, declared in the leaf's record.
//!
//! These are regression evidence on small fixtures, not proof. Each survey
//! is four points of a six-day SIR model, scored by its ODE skeleton or by a
//! twenty-particle filter; every execution, child, presentation and
//! provenance case runs on both. The sampled box (beta 1.5 to 4 a day) is a
//! fast epidemic, whose skeleton at the survey's step of one day depends on
//! the step, so an input that leaked into the step would move the landscape.
//!
//! The fixtures: `sir.camdl`, an SIR model with a vaccination intervention, a
//! scenario that enables it, a parameter-only rate factor that loop-invariant
//! code motion hoists and one observed stream; `patch.camdl`, a two-patch
//! model whose coupling matrix has structural zeros, which the constant fold
//! drops; two fit configs over the SIR model.

mod input_class_runner;

use std::sync::{Arc, Mutex, Weak};

use input_class_runner::*;

const KIND: &str = "survey";

const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R, V }
parameters {
  beta  : rate        in [0.01, 5.0]
  gamma : rate        in [0.01, 1.0]
  k     : probability in [0.01, 0.99]
}
let N = S + I + R + V
transitions {
  infection : S --> I @ beta * exp(-k) * S * I / N
  recovery  : I --> R @ gamma * I
}
interventions {
  sia : transfer(fraction = 0.05, from = S, to = V) at [3 'days]
}
scenarios {
  with_sia { enable = [sia] }
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected + 0.5)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0] }
tables {
  W : patch × patch = [[0.0, 0.5],
                       [0.5, 0.0]]
}
let N[l in patch] = S[l] + I[l] + R[l]
transitions {
  infection[l in patch] : S[l] --> I[l] @ beta * S[l] * sum(q in patch, W[l, q] * I[q] / N[q])
  recovery[l in patch]  : I[l] --> R[l] @ gamma * I[l]
}
init {
  S[p0] = 990  I[p0] = 10
  S[p1] = 1000  I[p1] = 5
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = sum(l in patch, I[l])
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = projected + 0.5)
  }
}
simulate { from = 0 'days  to = 6 'days }
"#;

/// A fit config over `sir.ir.json`'s `cases` stream, `beta` estimated.
fn fit_toml(d: &std::path::Path, gamma: f64) -> String {
    format!(
        "[model]\ncamdl = \"{}\"\n[data.observations]\ncases = \"{}\"\n\
         [estimate]\nbeta = {{ bounds = [0.1, 0.9], start = 0.4 }}\n[fixed]\ngamma = {gamma}\nk = 0.3\n",
        d.join("sir.ir.json").display(),
        d.join("cases.tsv").display(),
    )
}

fn files(d: &std::path::Path) -> Vec<(&'static str, String)> {
    let cases = |base: u32| (1..=6).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", base + 2 * t));
    vec![
        ("sir.camdl", SIR.to_string()),
        ("sir_other.camdl", SIR.replace("S = 990  I = 10", "S = 980  I = 20")),
        ("patch.camdl", PATCH.to_string()),
        ("cases.tsv", cases(10)),
        ("cases2.tsv", cases(11)),
        ("fit_a.toml", fit_toml(d, 0.2)),
        ("fit_b.toml", fit_toml(d, 0.25)),
    ]
}

fn fx() -> Arc<Fixture> {
    static LIVE: Mutex<Weak<Fixture>> = Mutex::new(Weak::new());
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(fx) = live.upgrade() {
        return fx;
    }
    let fx = Arc::new(Fixture::create("survey_input_classes", files, &["sir.camdl", "sir_other.camdl", "patch.camdl"]));
    *live = Arc::downgrade(&fx);
    fx
}

/// `camdl survey <model> --data cases.tsv --estimate beta=1.5:4.0` over four
/// points, scored by the ODE skeleton, seed 1.
fn sv_on(fx: &Arc<Fixture>, model: &str, extra: &[&str]) -> Invocation {
    let (m, d) = (fx.path(model), fx.path("cases.tsv"));
    Invocation::new(fx, &["survey", &m, "--data", &d, "--estimate", "beta=1.5:4.0", "--fixed", "gamma=0.2", "--fixed", "k=0.3"])
        .with(&["--n-points", "4", "--eval", "simulate", "--seed", "1"])
        .with(extra)
}

fn sv(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    sv_on(fx, "sir.ir.json", extra)
}

/// The same scored by a twenty-particle filter, one replicate.
fn sv_pf(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    sv(fx, &[]).set("--eval", "pfilter").with(&["--eval-particles", "20", "--eval-replicates", "1"]).with(extra)
}

/// A survey of `sir.ir.json` from the fit config `config`.
fn sv_fit(fx: &Arc<Fixture>, config: &str, extra: &[&str]) -> Invocation {
    let (m, f) = (fx.path("sir.ir.json"), fx.path(config));
    Invocation::new(fx, &["survey", &m, "--fit", &f, "--n-points", "4", "--eval", "simulate", "--seed", "1"]).with(extra)
}

fn patch(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    let (m, d) = (fx.path("patch.camdl"), fx.path("cases.tsv"));
    Invocation::new(fx, &["survey", &m, "--data", &d, "--estimate", "beta=0.1:0.9", "--fixed", "gamma=0.2"])
        .with(&["--n-points", "4", "--eval", "simulate", "--seed", "1"])
        .with(extra)
}

fn level(input: &str) -> String {
    declared_level("pub const SURVEY", input, "model")
}

fn assert_rekeys(input: &str, a: &Invocation, b: &Invocation) {
    assert_rekeyed(KIND, input, &level(input), &run(a), &run(b));
}

fn assert_same(a: &Invocation, b: &Invocation) {
    assert_same_result(KIND, &run(a), &run(b));
}

fn assert_unaffected(input: &str, a: &Invocation, b: &Invocation) {
    assert_leaf_unaffected(KIND, input, &run(a), &run(b));
}

// ─── keyed ───────────────────────────────────────────────────────────────────

#[test]
fn keyed_model() {
    let fx = fx();
    assert_rekeys("model", &sv(&fx, &[]), &sv_on(&fx, "sir_other.ir.json", &[]));
}

#[test]
fn keyed_allow_degenerate_rates() {
    let fx = fx();
    assert_rekeys("allow_degenerate_rates", &sv(&fx, &[]), &sv(&fx, &["--allow-degenerate-rates"]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    let fx = fx();
    assert_rekeys("allow_forcing_extrapolation", &sv(&fx, &[]), &sv(&fx, &["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_fit() {
    let fx = fx();
    assert_rekeys("fit", &sv_fit(&fx, "fit_a.toml", &[]), &sv_fit(&fx, "fit_b.toml", &[]));
}

#[test]
fn keyed_estimate() {
    let fx = fx();
    assert_rekeys("estimate", &sv(&fx, &[]), &sv(&fx, &[]).set("--estimate", "beta=1.6:4.0"));
}

#[test]
fn keyed_data() {
    let fx = fx();
    assert_rekeys("data", &sv(&fx, &[]), &sv(&fx, &[]).set("--data", &fx.path("cases2.tsv")));
}

#[test]
fn keyed_fixed() {
    let fx = fx();
    assert_rekeys("fixed", &sv(&fx, &[]), &sv(&fx, &[]).set("--fixed", "gamma=0.25"));
}

#[test]
fn keyed_scenario() {
    let fx = fx();
    assert_rekeys("scenario", &sv(&fx, &[]), &sv(&fx, &["--scenario", "with_sia"]));
}

#[test]
fn keyed_n_points() {
    let fx = fx();
    assert_rekeys("n_points", &sv(&fx, &[]), &sv(&fx, &[]).set("--n-points", "5"));
}

#[test]
fn keyed_eval() {
    let fx = fx();
    assert_rekeys("eval", &sv(&fx, &["--eval-particles", "20", "--eval-replicates", "1"]), &sv_pf(&fx, &[]));
}

#[test]
fn keyed_eval_particles() {
    let fx = fx();
    assert_rekeys("eval_particles", &sv_pf(&fx, &[]), &sv_pf(&fx, &[]).set("--eval-particles", "30"));
}

#[test]
fn keyed_eval_replicates() {
    let fx = fx();
    assert_rekeys("eval_replicates", &sv_pf(&fx, &[]), &sv_pf(&fx, &[]).set("--eval-replicates", "2"));
}

#[test]
fn keyed_seed() {
    let fx = fx();
    assert_rekeys("seed", &sv(&fx, &[]), &sv(&fx, &[]).set("--seed", "2"));
}

/// `--no-licm` changes the IR the compiler emits for a model with a
/// parameter-only rate factor; the compile is given it, so the model level
/// moves.
#[test]
fn keyed_no_licm() {
    let fx = fx();
    assert_rekeys("no_licm", &sv_on(&fx, "sir.camdl", &[]), &sv_on(&fx, "sir.camdl", &["--no-licm"]));
}

#[test]
fn keyed_env_no_licm() {
    let fx = fx();
    assert_rekeys("env:no_licm", &sv_on(&fx, "sir.camdl", &[]), &sv_on(&fx, "sir.camdl", &[]).env("CAMDL_NO_LICM", "1"));
}

#[test]
fn keyed_env_no_constant_fold() {
    let fx = fx();
    assert_rekeys("env:no_constant_fold", &patch(&fx, &[]), &patch(&fx, &[]).env("CAMDL_NO_CONSTANT_FOLD", "1"));
}

/// An engine switch keys the survey, and the leaf's record names it beside
/// the build; the landscape is the default path's.
#[test]
fn keyed_env_eval_unresolved() {
    let fx = fx();
    // Kept stores: the landscapes are read after the runs.
    let (sa, sb) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a = run_into(sa.path(), &sv_pf(&fx, &[]));
    let b = run_into(sb.path(), &sv_pf(&fx, &[]).env("CAMDL_EVAL_UNRESOLVED", "1"));
    assert_rekeyed(KIND, "env:eval_unresolved", &level("env:eval_unresolved"), &a, &b);
    let (plain, switched) = (a.of(KIND).into_values().next().unwrap().clone(), b.of(KIND).into_values().next().unwrap().clone());
    assert_eq!(switched.record["engine_version"], plain.record["engine_version"], "engine_version is the build");
    assert_eq!(switched.record["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
    assert!(plain.record.get("engine_switches").is_none(), "a default run's record has no switch list");
    // `landscape.tsv` names the run id in its header, which the switch moves.
    let body = |l: &Leaf| -> String {
        std::fs::read_to_string(l.dir.join("landscape.tsv")).unwrap().lines().filter(|x| !x.starts_with('#')).collect::<Vec<_>>().join("\n")
    };
    assert_eq!(body(&switched), body(&plain), "the same landscape under the alternative path");
}

#[test]
fn keyed_env_no_binding_cache() {
    let fx = fx();
    assert_rekeys("env:no_binding_cache", &sv(&fx, &[]), &sv(&fx, &[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let fx = fx();
    let (a, b) = (run(&sv(&fx, &[])), run(&sv(&fx, &[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1")));
    assert_rekeyed(KIND, "env:experimental", &level("env:experimental"), &a, &b);
    let leaf = b.of(KIND).into_values().next().unwrap().clone();
    assert_eq!(
        leaf.record["engine_switches"],
        serde_json::json!([{ "name": "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "value": "1" }])
    );
}

// ─── execution ───────────────────────────────────────────────────────────────
//
// Every execution, child, presentation and provenance case runs on both
// evaluation methods, the ODE skeleton and the particle filter ([`bases_on`]).

/// Both evaluation methods on `model`.
fn bases_on(fx: &Arc<Fixture>, model: &str) -> [Invocation; 2] {
    let simulate = sv_on(fx, model, &[]);
    let pfilter = simulate.set("--eval", "pfilter").with(&["--eval-particles", "20", "--eval-replicates", "1"]);
    [simulate, pfilter]
}

fn bases(fx: &Arc<Fixture>) -> [Invocation; 2] {
    bases_on(fx, "sir.ir.json")
}

/// The points are scored on the scoped pool `--parallel` sizes: by the ODE
/// skeleton and by the particle filter.
#[test]
fn execution_parallel() {
    let fx = fx();
    for b in bases(&fx) {
        assert_same(&b.with(&["--parallel", "1"]), &b.with(&["--parallel", "4"]));
    }
}

#[test]
fn execution_force() {
    let fx = fx();
    for b in bases(&fx) {
        let store = tempfile::tempdir().unwrap();
        let first = run_into(store.path(), &b);
        let forced = run_into(store.path(), &b.with(&["--force"]));
        assert_same_result(KIND, &first, &forced);
        assert!(store.path().join("results/.quarantine").is_dir(), "--force displaced the incumbent");
    }
}

/// `--output` is the store root: the same leaf under another root.
#[test]
fn execution_output() {
    let fx = fx();
    for b in bases(&fx) {
        let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let root = y.path().join("elsewhere");
        let (sx, sy) = (run_into(x.path(), &b), run_into(y.path(), &b.with(&["--output", &root.to_string_lossy()])));
        assert!(sy.of(KIND).values().all(|l| l.dir.starts_with(&root)), "the store root moved");
        assert_same_result(KIND, &sx, &sy);
    }
}

#[test]
fn execution_no_ir_cache() {
    let fx = fx();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.with(&["--no-ir-cache"]));
    }
}

#[test]
fn execution_env_no_ir_cache() {
    let fx = fx();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDL_NO_IR_CACHE", "1"));
    }
}

#[test]
fn execution_env_ir_cache_dir() {
    let fx = fx();
    let elsewhere = tempfile::tempdir().unwrap();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDL_IR_CACHE_DIR", &elsewhere.path().to_string_lossy()));
    }
}

#[test]
fn execution_env_xdg_cache_home() {
    let fx = fx();
    let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for b in bases_on(&fx, "sir.camdl") {
        let under = |d: &std::path::Path| b.clone().without("CAMDL_IR_CACHE_DIR").env("XDG_CACHE_HOME", &d.to_string_lossy());
        assert_same(&under(x.path()), &under(y.path()));
    }
}

#[test]
fn execution_env_home() {
    let fx = fx();
    let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for b in bases_on(&fx, "sir.camdl") {
        let under = |d: &std::path::Path| b.clone().without("CAMDL_IR_CACHE_DIR").env("HOME", &d.to_string_lossy());
        assert_same(&under(x.path()), &under(y.path()));
    }
}

#[test]
fn execution_env_camdlc_path() {
    let fx = fx();
    let camdlc = fx.camdlc.to_string_lossy().into_owned();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDLC_PATH", &camdlc));
    }
}

#[test]
fn execution_env_camdlc() {
    let fx = fx();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.camdlc, bin.path().join("camdlc")).unwrap();
    let path = format!("{}:{}", bin.path().display(), std::env::var("PATH").unwrap_or_default());
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().without("CAMDLC").env("PATH", &path));
    }
}

#[test]
fn execution_env_output_dir() {
    let fx = fx();
    for b in bases(&fx) {
        let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let elsewhere = y.path().join("elsewhere");
        let moved = b.clone().env("CAMDL_OUTPUT_DIR", &elsewhere.to_string_lossy());
        let (sx, sy) = (run_into(x.path(), &b), run_into(y.path(), &moved));
        assert!(sy.of(KIND).values().all(|l| l.dir.starts_with(&elsewhere)), "the store root moved");
        assert_same_result(KIND, &sx, &sy);
    }
}

// ─── child ───────────────────────────────────────────────────────────────────

/// `--render` asks for the leaf's child `landscape_html/`: the leaf's run id
/// and files are unchanged, and the child is at its address, declared.
#[test]
fn child_render() {
    let fx = fx();
    for b in bases(&fx) {
        let (sa, sb) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (x, y) = (run_into(sa.path(), &b), run_into(sb.path(), &b.with(&["--render"])));
        assert_eq!(x.result(KIND), y.result(KIND), "the leaf's run id and files are unchanged");
        let leaf = y.of(KIND).into_values().next().unwrap().clone();
        assert!(leaf.dir.join("landscape_html/landscape.html").is_file(), "the child is at its address");
        assert!(leaf.dir.join("landscape_html/landscape_html.json").is_file(), "with its record");
        assert_eq!(leaf.record["children"]["landscape_html"].as_array().map(Vec::len), Some(1), "declared");
    }
}

// ─── presentation and provenance ────────────────────────────────────────────

#[test]
fn provenance_label() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--label", &b, &b.with(&["--label", "a-label"]));
    }
}

#[test]
fn presentation_verbosity() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--verbosity", &b, &b.with(&["--verbosity", "debug"]));
    }
}

#[test]
fn presentation_progress() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--progress", &b.with(&["--progress", "none"]), &b.with(&["--progress", "plain"]));
    }
}

#[test]
fn presentation_no_progress() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--no-progress", &b.with(&["--progress", "plain"]), &b.with(&["--no-progress"]));
    }
}

#[test]
fn presentation_env_trace_steps() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("CAMDL_TRACE_STEPS", &b, &b.clone().env("CAMDL_TRACE_STEPS", "1"));
    }
}

#[test]
fn presentation_env_no_color() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("NO_COLOR", &b, &b.clone().env("NO_COLOR", "1"));
    }
}

#[test]
fn presentation_env_clicolor_force() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("CLICOLOR_FORCE", &b, &b.clone().env("CLICOLOR_FORCE", "1"));
    }
}

// ─── the --fit handle ────────────────────────────────────────────────────────

/// `--fit` takes the handle forms `fit predict` takes: a stored fit of
/// `fit_a.toml`, named by its leaf directory, problem folder or run-id
/// prefix, names the problem its archived config states, the file's, as the
/// `fit.toml` does, so every form surveys the one computation.
#[test]
fn every_fit_handle_names_the_fits_problem() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let config = fx.path("fit_a.toml");
    let with_method = format!(
        "{}[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n",
        std::fs::read_to_string(&config).unwrap()
    );
    let fit_path = store.path().join("fit_a.toml");
    std::fs::write(&fit_path, with_method).unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx, &["fit", "run", &fit_path.to_string_lossy()]));
    let leaf = fitted.of("fit_method").into_values().next().expect("the fit's leaf").clone();
    let segment = leaf.dir.parent().unwrap().parent().unwrap().to_path_buf();
    let handles = [
        fit_path.to_string_lossy().into_owned(),
        leaf.dir.to_string_lossy().into_owned(),
        segment.to_string_lossy().into_owned(),
        leaf.run_id[..12].to_string(),
    ];
    let mut ids = Vec::new();
    for h in &handles {
        let s = run_into(store.path(), &sv_fit(&fx, "fit_a.toml", &[]).set("--fit", h));
        ids.push(s.ids(KIND));
    }
    assert!(!ids[0].is_empty(), "the survey stored its leaf");
    assert!(ids.iter().all(|i| *i == ids[0]), "every handle names one problem: {ids:?}");
}

/// A point of `fit run --sweep` named by its leaf directory, its problem
/// folder or its run-id prefix is surveyed at the point's own value, which
/// its fit folder's `fit.resolved.toml` holds: every form stores the bytes a
/// survey of `fit_b.toml`, whose file writes the point's value, stores, and
/// not those of the file's own value. The forms name one computation.
#[test]
fn a_sweep_point_handle_surveys_at_the_points_value() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let with_method = format!(
        "{}[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n",
        std::fs::read_to_string(fx.path("fit_a.toml")).unwrap()
    );
    let fit_path = store.path().join("fit_a.toml");
    std::fs::write(&fit_path, with_method).unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx, &["fit", "run", &fit_path.to_string_lossy(), "--sweep", "gamma=0.2,0.25"]));
    let (_, other) = sweep_points(&fitted, "gamma", 0.2);
    assert_eq!(segment_fixed(&other.dir, "gamma"), Some(0.25));
    let written = run(&sv_fit(&fx, "fit_b.toml", &[])).bytes(KIND);
    let unswept = run(&sv_fit(&fx, "fit_a.toml", &[])).bytes(KIND);
    assert!(!written.is_empty());
    assert_ne!(written, unswept, "the swept value changes what a survey stores");
    let segment = other.dir.parent().unwrap().parent().unwrap().to_string_lossy().into_owned();
    let mut ids = Vec::new();
    for h in [other.dir.to_string_lossy().into_owned(), segment, other.run_id[..12].to_string()] {
        let s = run_into(store.path(), &sv_fit(&fx, "fit_a.toml", &[]).set("--fit", &h));
        assert_eq!(s.bytes(KIND), written, "--fit {h}: the survey of the point's value");
        ids.push(s.ids(KIND));
    }
    assert!(ids.iter().all(|i| *i == ids[0]), "every handle names one computation: {ids:?}");
}
